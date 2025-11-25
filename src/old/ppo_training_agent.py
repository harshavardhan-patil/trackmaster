"""
PPO TrainingAgent for use with TrainingOnline.

This module provides a PPO (Proximal Policy Optimization) implementation
that is compatible with TMRL's TrainingAgent interface and the TrainingOnline
class.
"""

# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# local imports
from tmrl.training import TrainingAgent


__docformat__ = "google"


class PPOTrainingAgent(TrainingAgent):
    """
    PPO (Proximal Policy Optimization) training agent.

    This agent implements the PPO algorithm for on-policy reinforcement learning.
    It maintains both a policy network and a value network, and optimizes them
    using the PPO clipped objective.

    Args:
        observation_space: observation space of the environment
        action_space: action space of the environment
        device (str): device for training
        policy_network (nn.Module): policy network (actor)
        value_network (nn.Module): value network (critic)
        policy_lr (float): learning rate for policy
        value_lr (float): learning rate for value function
        clip_epsilon (float): PPO clipping parameter
        value_loss_coef (float): coefficient for value loss
        entropy_coef (float): coefficient for entropy bonus
        max_grad_norm (float): maximum gradient norm for clipping
        normalize_advantages (bool): whether to normalize advantages
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 policy_network=None,
                 value_network=None,
                 policy_lr=3e-4,
                 value_lr=3e-4,
                 clip_epsilon=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 normalize_advantages=True):
        """
        Initialize PPO training agent.
        """
        super().__init__(observation_space, action_space, device)

        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

        # Create or use provided networks
        if policy_network is None:
            self.policy = self._create_default_policy().to(device)
        else:
            self.policy = policy_network.to(device)

        if value_network is None:
            self.value = self._create_default_value().to(device)
        else:
            self.value = value_network.to(device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)

        # Training stats
        self.train_step = 0

    def _create_default_policy(self):
        """
        Create a default policy network.

        Override this method to create custom policy architectures.

        Returns:
            nn.Module: policy network
        """
        # This is a placeholder - you should provide your own policy
        # For continuous actions
        raise NotImplementedError(
            "You must provide a policy_network or override _create_default_policy()"
        )

    def _create_default_value(self):
        """
        Create a default value network.

        Override this method to create custom value architectures.

        Returns:
            nn.Module: value network
        """
        # This is a placeholder - you should provide your own value network
        raise NotImplementedError(
            "You must provide a value_network or override _create_default_value()"
        )

    def train(self, batch):
        """
        Execute a PPO training step.

        Args:
            batch (dict): batch of trajectory data containing:
                - observations: tensor of observations
                - actions: tensor of actions
                - log_probs: tensor of old log probabilities
                - advantages: tensor of advantages
                - returns: tensor of returns
                - values: tensor of old state values

        Returns:
            dict: training statistics
        """
        # Extract batch data
        observations = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass through policy
        new_log_probs, entropy = self._compute_policy_stats(observations, actions)

        # Compute policy loss (PPO clipped objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()

        # Total policy loss
        total_policy_loss = policy_loss + self.entropy_coef * entropy_loss

        # Forward pass through value network
        predicted_values = self.value(observations).squeeze(-1)

        # Compute value loss
        value_loss = F.mse_loss(predicted_values, returns)

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        # Compute additional statistics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

        self.train_step += 1

        # Return statistics
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'total_loss': (total_policy_loss + self.value_loss_coef * value_loss).item()
        }

    def _compute_policy_stats(self, observations, actions):
        """
        Compute log probabilities and entropy for given observations and actions.

        This method should be overridden based on your policy architecture.

        Args:
            observations: batch of observations
            actions: batch of actions

        Returns:
            tuple: (log_probs, entropy)
        """
        # This is a placeholder - implement based on your policy type
        raise NotImplementedError(
            "You must override _compute_policy_stats() for your specific policy"
        )

    def get_actor(self):
        """
        Returns the current policy network to be broadcast to workers.

        Returns:
            nn.Module: current policy network
        """
        return self.policy


class PPOTrainingAgentContinuous(PPOTrainingAgent):
    """
    PPO agent for continuous action spaces using Gaussian policies.

    This is a concrete implementation for continuous control tasks.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 policy_network,
                 value_network,
                 policy_lr=3e-4,
                 value_lr=3e-4,
                 clip_epsilon=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 normalize_advantages=True):
        """
        Initialize PPO agent for continuous actions.
        """
        super().__init__(
            observation_space,
            action_space,
            device,
            policy_network,
            value_network,
            policy_lr,
            value_lr,
            clip_epsilon,
            value_loss_coef,
            entropy_coef,
            max_grad_norm,
            normalize_advantages
        )

    def _compute_policy_stats(self, observations, actions):
        """
        Compute log probabilities and entropy for continuous actions.

        Assumes the policy network outputs a Gaussian distribution.

        Args:
            observations: batch of observations
            actions: batch of actions taken

        Returns:
            tuple: (log_probs, entropy)
        """
        # Get distribution from policy
        dist = self.policy(observations)

        # Compute log probabilities
        log_probs = dist.log_prob(actions)

        # For multivariate actions, sum over action dimensions
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(dim=-1)

        # Compute entropy
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)

        return log_probs, entropy


class PPOTrainingAgentDiscrete(PPOTrainingAgent):
    """
    PPO agent for discrete action spaces using categorical policies.

    This is a concrete implementation for discrete action tasks.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 policy_network,
                 value_network,
                 policy_lr=3e-4,
                 value_lr=3e-4,
                 clip_epsilon=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 normalize_advantages=True):
        """
        Initialize PPO agent for discrete actions.
        """
        super().__init__(
            observation_space,
            action_space,
            device,
            policy_network,
            value_network,
            policy_lr,
            value_lr,
            clip_epsilon,
            value_loss_coef,
            entropy_coef,
            max_grad_norm,
            normalize_advantages
        )

    def _compute_policy_stats(self, observations, actions):
        """
        Compute log probabilities and entropy for discrete actions.

        Assumes the policy network outputs logits for a Categorical distribution.

        Args:
            observations: batch of observations
            actions: batch of actions taken

        Returns:
            tuple: (log_probs, entropy)
        """
        # Get logits from policy
        logits = self.policy(observations)

        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)

        # Compute log probabilities
        log_probs = dist.log_prob(actions)

        # Compute entropy
        entropy = dist.entropy()

        return log_probs, entropy
