from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from src.utils import TorchJSONDecoder, TorchJSONEncoder
import json

import logging
logger = logging.getLogger(__name__)

DINOv2_PRETRAINED_MODEL = "facebook/dinov2-small"

# Base Neural Network
class TrackmasterNetwork(nn.Module):
    def __init__(self, is_critic=False):
        super().__init__()

        logger.info(f"Initialized TrackmasterNetwork as critic={is_critic}")
        self.is_critic = is_critic
        # todo: need smaller vision model, currently getting timeout warnings
        self.vision_model = AutoModel.from_pretrained(
            DINOv2_PRETRAINED_MODEL, 
            device_map=self.device, 
        )
        self.vision_preprocessor = AutoImageProcessor.from_pretrained(DINOv2_PRETRAINED_MODEL)

        # todo: action sampling, etc - See action_space
        if self.is_critic:
            non_image_feature_len = 12
        else:
            non_image_feature_len = 9
        self.mlp = nn.Sequential(
            nn.Linear(4 * 384 + non_image_feature_len, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 3)
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, obs):
        if self.is_critic:
            speed, gear, rpm, images, act1, act2, next_act = obs
        else:
            speed, gear, rpm, images, act1, act2 = obs

        # Images are RGB with shape (4, 3, 224, 224) or (4, 224, 224, 3)
        # DiNOv2 expects list of images in (H, W, C) format
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        # Ensure values are in 0-255 range for uint8
        images = np.clip(images, 0, 255).astype(np.uint8)
        # Convert to list of (H, W, C) format images
        image_list = []
        for i in range(images.shape[0]):
            img = images[i]  # Shape: (3, 224, 224) or (224, 224, 3)
            # If channels first (3, 224, 224), transpose to (224, 224, 3)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            image_list.append(img)


        inputs = self.vision_preprocessor(images=image_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.vision_model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0]

        # Build MLP input based on whether this is critic or actor
        if self.is_critic:
            mlp_input = torch.cat((speed.squeeze(0), gear.squeeze(0), rpm.squeeze(0), act1.flatten(), act2.flatten(), next_act.flatten(), cls_token.flatten()))
        else:
            mlp_input = torch.cat((speed.squeeze(0), gear.squeeze(0), rpm.squeeze(0), act1.flatten(), act2.flatten(), cls_token.flatten()))

        mlp_output = self.mlp(mlp_input)
        output = self.tanh(mlp_output)
        return output

# TMRL required TorchActorModule using the base Neural Net
class TrackmasterActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        self.model = TrackmasterNetwork(is_critic=False)

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Forward pass for PPO actor.

        Args:
            obs: observation from environment
            test: if True, return deterministic actions (mean)
            compute_logprob: if True, compute log probabilities for training

        Returns:
            action: sampled or deterministic action
            logp: log probability of the action (None if compute_logprob=False)
        """
        # Get the action mean from the network (tanh already applied, output in [-1, 1])
        action_mean = self.model(obs)

        # For PPO, we use a fixed std or learnable std
        # Here we use a simple fixed log std
        log_std = torch.ones_like(action_mean) * -0.5  # std = exp(-0.5) ≈ 0.6
        std = torch.exp(log_std)

        # Create Normal distribution
        pi_distribution = torch.distributions.Normal(action_mean, std)

        if test:
            # Deterministic action at test time
            pi_action = action_mean
        else:
            # Sample action during training
            pi_action = pi_distribution.rsample()

        # Compute log probability if requested
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        else:
            logp_pi = None

        # Clamp action to valid range [-1, 1]
        pi_action = torch.clamp(pi_action, -1.0, 1.0)

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test)
            return a.cpu().numpy()
    
    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))
        return self

# Unified class for the Trainer
class TrackmasterActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # Actor
        self.actor = TrackmasterActorModule(observation_space, action_space)
        # Critic - Note: for PPO, critic doesn't need the action, just observation
        # But we're reusing the TrackmasterNetwork which expects action for Q-learning
        # For PPO value function, we'll use a separate simple critic
        self.critic = TrackmasterNetwork(is_critic=True)  # Value function doesn't need action

        