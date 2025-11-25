"""
Standalone PPO Training for TrackMania
Based on TrackManiaPPO.ipynb approach, adapted for custom TrackmasterNetwork
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
import tmrl

from src.actor import TrackmasterActorModule, TrackmasterNetwork

logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PPOStandaloneTrainer:
    """
    Standalone PPO trainer that doesn't use TMRL's distributed architecture.
    Directly interfaces with the environment and collects data in episodes.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # PPO hyperparameters
        policy_lr=1e-5,
        critic_lr=1e-5,
        gamma=0.996,
        clip_coef=0.2,
        critic_coef=0.1,
        entropy_coef=0.1,
        batch_size=256,
        num_updates=10000,
        epochs_per_update=100,
        max_episode_steps=2400,
        norm_advantages=True,
        grad_clip_val=0.1
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        # Hyperparameters
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.clip_coef = clip_coef
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.epochs_per_update = epochs_per_update
        self.max_episode_steps = max_episode_steps
        self.norm_advantages = norm_advantages
        self.grad_clip_val = grad_clip_val

        # Initialize actor and critic
        # NOTE: For PPO, the actor outputs actions and the critic outputs state values (not Q-values)
        # The critic should take only observations as input, not (observation, action) pairs
        self.actor = TrackmasterActorModule(observation_space, action_space).to(device)

        # NOTE: Your current TrackmasterNetwork is designed for Q-learning (critic takes action as input)
        # For PPO, you need a value network that only takes observations
        # TODO: Create a separate value network or modify TrackmasterNetwork to support PPO value function
        # For now, we'll use the existing network in critic mode
        self.critic = TrackmasterNetwork(is_critic=True).to(device)

        # Optimizers
        self.policy_optim = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        logger.info(f"Initialized PPO Standalone Trainer on device: {device}")
        logger.info(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        logger.info(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")

    def obs_to_tensor(self, obs) -> Tuple[torch.Tensor, ...]:
        """
        Convert environment observation to tensors.

        Args:
            obs: tuple from environment (speed, gear, rpm, images, act1, act2)

        Returns:
            tuple of tensors suitable for the network
        """
        speed, gear, rpm, images, act1, act2 = obs

        # Convert to tensors
        speed_t = torch.tensor(speed, dtype=torch.float32).view(1, -1)
        gear_t = torch.tensor(gear, dtype=torch.float32).view(1, -1)
        rpm_t = torch.tensor(rpm, dtype=torch.float32).view(1, -1)
        images_t = torch.tensor(images, dtype=torch.float32)
        act1_t = torch.tensor(act1, dtype=torch.float32).view(1, -1)
        act2_t = torch.tensor(act2, dtype=torch.float32).view(1, -1)

        return speed_t, gear_t, rpm_t, images_t, act1_t, act2_t

    def collect_episode(self, env) -> Dict:
        """
        Collect one episode of data.

        Args:
            env: the TrackMania environment

        Returns:
            Dictionary containing episode data (observations, actions, logprobs, rewards, state_values)
        """
        # Storage for episode data
        observations = []
        actions = []
        logprobs = []
        rewards = []
        state_values = []

        # Reset environment
        obs = env.reset()[0]  # Get observation from reset

        self.actor.eval()
        self.critic.eval()

        step_count = 0
        done = False

        while not done and step_count < self.max_episode_steps:
            # Convert observation to tensors
            obs_tensors = self.obs_to_tensor(obs)

            # Get action and log probability from actor
            with torch.no_grad():
                action, logprob = self.actor.forward(
                    tuple(t.to(self.device) for t in obs_tensors),
                    test=False,
                    compute_logprob=True
                )

                # Get state value from critic
                # NOTE: For PPO, critic should only take observation, not (obs, action)
                # TODO: Modify critic to accept only observation for PPO value function
                # For now, we append action to make it compatible with current architecture
                state_value = self.critic(
                    tuple(t.to(self.device) for t in obs_tensors) + (action,)
                )

            # Store data
            observations.append(obs_tensors)
            actions.append(action.cpu())
            logprobs.append(logprob.cpu())
            state_values.append(state_value.cpu())

            # Clip action to valid range and convert to numpy
            clamped_action = np.clip(action.cpu().numpy(), -1, 1)

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(clamped_action)

            rewards.append(reward)
            obs = next_obs

            done = terminated or truncated
            step_count += 1

        # Pause environment (as per TMRL docs)
        env.wait()

        logger.info(f"Episode collected: {step_count} steps, total reward: {sum(rewards):.2f}")

        return {
            'observations': observations,
            'actions': actions,
            'logprobs': logprobs,
            'rewards': rewards,
            'state_values': state_values,
            'step_count': step_count,
            'terminated': done
        }

    def compute_returns_and_advantages(self, episode_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages for PPO.

        Args:
            episode_data: dictionary containing episode data

        Returns:
            returns: tensor of returns for each timestep
            advantages: tensor of advantages for each timestep
        """
        rewards = episode_data['rewards']
        state_values = torch.stack(episode_data['state_values']).squeeze()
        step_count = episode_data['step_count']
        terminated = episode_data['terminated']

        returns = torch.zeros(step_count)

        # Compute returns (discounted cumulative rewards)
        with torch.no_grad():
            for t in range(step_count - 1, -1, -1):
                if t == step_count - 1:
                    # Last timestep
                    if not terminated:
                        # Bootstrap with critic if episode was truncated
                        # NOTE: Would need final observation for proper bootstrapping
                        returns[t] = rewards[t]
                    else:
                        returns[t] = rewards[t]
                else:
                    returns[t] = rewards[t] + self.gamma * returns[t + 1]

            # Compute advantages (returns - value estimates)
            advantages = returns - state_values

        return returns, advantages

    def train_step(self, episode_data: Dict, returns: torch.Tensor, advantages: torch.Tensor) -> Dict:
        """
        Perform PPO training update.

        Args:
            episode_data: dictionary containing episode data
            returns: computed returns
            advantages: computed advantages

        Returns:
            Dictionary of training metrics
        """
        step_count = episode_data['step_count']
        observations = episode_data['observations']
        actions = torch.stack(episode_data['actions']).squeeze()
        old_logprobs = torch.stack(episode_data['logprobs']).squeeze()

        # Random permutation for mini-batch sampling
        indices = np.random.permutation(step_count)

        actor_losses = []
        critic_losses = []
        total_losses = []

        self.actor.train()
        self.critic.train()

        # Multiple epochs of updates
        for epoch in range(self.epochs_per_update):
            # Mini-batch updates
            for batch_start in range(0, step_count, self.batch_size):
                batch_end = min(batch_start + self.batch_size, step_count)
                batch_indices = indices[batch_start:batch_end]

                # Get batch data
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = actions[batch_indices].to(self.device)
                batch_old_logprobs = old_logprobs[batch_indices].to(self.device)
                batch_returns = returns[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)

                # Normalize advantages
                if self.norm_advantages:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # ========== PPO Actor Update ==========
                # TODO: Implement PPO clipped surrogate objective
                # 1. Get new log probabilities for the batch actions under current policy
                # 2. Compute probability ratio: ratio = exp(new_logprob - old_logprob)
                # 3. Compute surrogate losses:
                #    - unclipped_loss = ratio * advantages
                #    - clipped_loss = clip(ratio, 1-epsilon, 1+epsilon) * advantages
                # 4. Take minimum (for maximization, or maximum for minimization)
                # 5. Add entropy bonus for exploration

                # Placeholder for actor loss
                actor_loss = torch.tensor(0.0).to(self.device)

                # ========== PPO Critic Update ==========
                # TODO: Implement value function loss
                # 1. Get new state value estimates for batch observations
                # 2. Compute MSE loss: (new_values - returns)^2

                # Placeholder for critic loss
                critic_loss = torch.tensor(0.0).to(self.device)

                # Total loss
                total_loss = actor_loss + self.critic_coef * critic_loss

                # Optimizer steps
                self.policy_optim.zero_grad()
                self.critic_optim.zero_grad()

                total_loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=self.grad_clip_val)
                nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=self.grad_clip_val)

                self.policy_optim.step()
                self.critic_optim.step()

                # Record losses
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                total_losses.append(total_loss.item())

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'total_loss': np.mean(total_losses)
        }

    def train(self, env):
        """
        Main training loop.

        Args:
            env: the TrackMania environment
        """
        logger.info("Starting PPO training...")

        cum_rewards = []
        actor_losses = []
        critic_losses = []
        total_losses = []

        for update in range(self.num_updates):
            logger.info(f"Update {update + 1}/{self.num_updates}")

            # Collect episode
            episode_data = self.collect_episode(env)

            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(episode_data)

            # Perform training update
            metrics = self.train_step(episode_data, returns, advantages)

            # Log metrics
            cum_reward = sum(episode_data['rewards'])
            cum_rewards.append(cum_reward)
            actor_losses.append(metrics['actor_loss'])
            critic_losses.append(metrics['critic_loss'])
            total_losses.append(metrics['total_loss'])

            logger.info(f"  Total Reward: {cum_reward:.2f}")
            logger.info(f"  Actor Loss: {metrics['actor_loss']:.4f}")
            logger.info(f"  Critic Loss: {metrics['critic_loss']:.4f}")
            logger.info(f"  Total Loss: {metrics['total_loss']:.4f}")

            # Save model periodically
            if cum_reward > 200 or (update + 1) % 50 == 0:
                self.save_model(f"checkpoint_update_{update+1}_reward_{cum_reward:.2f}.pt")

        return {
            'cum_rewards': cum_rewards,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses,
            'total_losses': total_losses
        }

    def save_model(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
        }
        torch.save(checkpoint, filename)
        logger.info(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        logger.info(f"Model loaded from {filename}")

    def evaluate(self, env, num_episodes: int = 5):
        """
        Evaluate the current policy.

        Args:
            env: the TrackMania environment
            num_episodes: number of episodes to evaluate

        Returns:
            average reward across episodes
        """
        logger.info(f"Evaluating for {num_episodes} episodes...")

        self.actor.eval()
        total_rewards = []

        for episode in range(num_episodes):
            obs = env.reset()[0]
            done = False
            episode_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                obs_tensors = self.obs_to_tensor(obs)

                with torch.no_grad():
                    action, _ = self.actor.forward(
                        tuple(t.to(self.device) for t in obs_tensors),
                        test=True,  # Deterministic actions
                        compute_logprob=False
                    )

                clamped_action = np.clip(action.cpu().numpy(), -1, 1)
                next_obs, reward, terminated, truncated, info = env.step(clamped_action)

                episode_reward += reward
                obs = next_obs
                done = terminated or truncated
                step_count += 1

            env.wait()
            total_rewards.append(episode_reward)
            logger.info(f"  Episode {episode + 1}: {episode_reward:.2f} reward, {step_count} steps")

        avg_reward = np.mean(total_rewards)
        logger.info(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

        return avg_reward
