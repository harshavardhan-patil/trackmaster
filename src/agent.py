import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from src.network import TrackMasterCNN

class Actor(nn.Module):
    """Policy network for action selection using CNN"""

    def __init__(self, action_space: int = 3):
        super().__init__()
        self.action_space = action_space

        # CNN backbone
        self.backbone = TrackMasterCNN()
        mlp_output_size = self.backbone.mlp_layers[-1]

        # Action mean layer (outputs 3 actions: steering, throttle, brake)
        self.action_mean = nn.Sequential(
            nn.Linear(mlp_output_size, action_space),
            nn.Tanh()  # Bound actions to [-1, 1]
        )

        # Action log variance layer (single shared variance for all action dimensions)
        self.actor_logvar = nn.Linear(mlp_output_size, 1)

    def sample_action_with_logprobs(self, observation):
        """Sample action and compute log probability"""
        dist = self(observation)
        sample_action = dist.sample()
        return sample_action, dist.log_prob(sample_action)

    def mean_only(self, observation):
        """Get mean action without sampling (for evaluation)"""
        with torch.no_grad():
            # Forward through backbone
            features = self.backbone(observation)
            return self.action_mean(features)

    def get_action_log_prob(self, observation, action):
        """Get log probability of given action"""
        dist = self(observation)
        return dist.log_prob(action)

    def forward(self, observation):
        """
        Forward pass to create action distribution

        Args:
            observation: tuple of (speed, gear, rpm, images, act1, act2)

        Returns:
            MultivariateNormal distribution over actions
        """
        features = self.backbone(observation)

        # Get action means
        means = self.action_mean(features)
        means = torch.stack([
          torch.sigmoid(means[:, 0]),  # Gas: 0 to 1
          torch.sigmoid(means[:, 1]),  # Brake: 0 to 1
          torch.tanh(means[:, 2])      # Steering: -1 to 1
      ], dim=1)

        log_vars = self.actor_logvar(features)
        # Reduce std to 20% by adding log(0.04) ≈ -3.219 to log_vars
        # Since std = exp(log_var/2), reducing std by 0.2x means log_var' = log_var + 2*log(0.2)
        log_vars = log_vars + np.log(0.04)
        # Clamp log_variance to a safe range
        # min -23.219 (std ~0.000009), max -1.219 (std ~0.544)
        log_vars = torch.clamp(log_vars, min=-23.219, max=-1.219)
        
        vars = torch.zeros(features.shape[0], self.action_space).to(features.device)
        vars[:, :] = log_vars.exp().view(-1, 1)
        covar_mat = torch.zeros(features.shape[0], self.action_space, self.action_space).to(features.device)
        covar_mat[:, np.arange(self.action_space), np.arange(self.action_space)] = vars
        dist = torch.distributions.MultivariateNormal(means, covar_mat)
        return dist


class Critic(nn.Module):
    """Critic network for state value estimation using CNN"""

    def __init__(self):
        super().__init__()

        # CNN backbone (q_net=False since we estimate V(s) not Q(s,a) for PPO)
        self.backbone = TrackMasterCNN()

        # Output layer to produce single value
        # The backbone outputs 256 features
        mlp_output_size = 256  # Last layer of backbone MLP
        self.value_head = nn.Linear(mlp_output_size, 1)

    def forward(self, observation):
        """
        Forward pass to estimate state value

        Args:
            observation: tuple of (speed, gear, rpm, images, act1, act2)

        Returns:
            State value V(s) as a tensor
        """
        # The backbone expects (speed, gear, rpm, images, act1, act2)
        features = self.backbone(observation)
        value = self.value_head(features)
        return value


class Agent(nn.Module):
    """Agent combining Policy and Critic"""

    def __init__(self, action_space: int = 3):
        super().__init__()
        self.policy = Actor(action_space=action_space)
        self.critic = Critic()

    def forward(self, x):
        raise SyntaxError('Propagate through Agent.policy and Agent.critic individually')
