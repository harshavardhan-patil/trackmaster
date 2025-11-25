import torch
import numpy as np
import json
from typing import Tuple


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


def obs_preprocessor(obs):
    """
    Preprocess observation.

    Args:
        obs: tuple from environment

    Returns:
        preprocessed observation
    """
    speed, gear, rpm, images, act1, act2 = obs
    speed = speed / 1000.0
    images = images.astype(np.float32) / 255.0

    return speed, gear, rpm, images, act1, act2


def env_obs_to_tensor(obs, device='cpu') -> Tuple[torch.Tensor, ...]:
    """
    Convert environment observation to tensors for PPO training.
    Based on TrackManiaPPO.ipynb approach.

    Args:
        obs: tuple from environment (speed, gear, rpm, images, act1, act2)
        device: device to place tensors on

    Returns:
        tuple of tensors
    """
    speed, gear, rpm, images, act1, act2 = obs

    # Convert to tensors and move to device
    speed_t = torch.tensor(speed, dtype=torch.float32, device=device).view(1, -1)
    gear_t = torch.tensor(gear, dtype=torch.float32, device=device).view(1, -1)
    rpm_t = torch.tensor(rpm, dtype=torch.float32, device=device).view(1, -1)
    images_t = torch.tensor(images, dtype=torch.float32, device=device)
    act1_t = torch.tensor(act1, dtype=torch.float32, device=device).view(1, -1)
    act2_t = torch.tensor(act2, dtype=torch.float32, device=device).view(1, -1)

    return speed_t, gear_t, rpm_t, images_t, act1_t, act2_t


def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    This is an optional but commonly used improvement for PPO.

    Args:
        rewards: list of rewards
        values: tensor of state values
        gamma: discount factor
        lambda_: GAE lambda parameter

    Returns:
        advantages: tensor of advantages
        returns: tensor of returns
    """
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)

    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def normalize_observations(obs, mean=None, std=None, epsilon=1e-8):
    """
    Normalize observations for better training stability.
    Optional utility for PPO training.

    Args:
        obs: observation tensor
        mean: running mean (if None, compute from obs)
        std: running std (if None, compute from obs)
        epsilon: small value to avoid division by zero

    Returns:
        normalized observation, mean, std
    """
    if mean is None:
        mean = obs.mean(dim=0, keepdim=True)
    if std is None:
        std = obs.std(dim=0, keepdim=True)

    normalized_obs = (obs - mean) / (std + epsilon)
    return normalized_obs, mean, std