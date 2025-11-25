"""
OnlineMemory class for on-policy algorithms like PPO.

This module provides a trajectory-based memory buffer designed for online,
on-policy RL algorithms. Unlike replay buffers that sample randomly, this
memory stores complete trajectories and provides them in order for training.
"""

# standard library imports
import logging
from collections import deque

# third-party imports
import numpy as np
import torch

# local imports
from tmrl.util import collate_torch


__docformat__ = "google"


class OnlineMemory:
    """
    Trajectory buffer for on-policy algorithms.

    This memory stores complete trajectories (episodes) and provides them
    for training. After training, the buffer should be cleared to maintain
    the on-policy constraint.

    Key features:
    - Stores complete trajectories, not individual transitions
    - Supports batching of trajectory data
    - Designed for PPO-style algorithms that need:
      * Observations
      * Actions
      * Rewards
      * Log probabilities
      * Advantages
      * Returns
    """

    def __init__(self,
                 device,
                 batch_size=256,
                 max_trajectories=100):
        """
        Initialize the online memory buffer.

        Args:
            device (str): device to collate tensors onto
            batch_size (int): size of batches to yield during iteration
            max_trajectories (int): maximum number of trajectories to store
        """
        self.device = device
        self.batch_size = batch_size
        self.max_trajectories = max_trajectories

        # Storage for trajectories
        self.trajectories = deque(maxlen=max_trajectories)

        # Flattened storage for efficient batching
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

        # Statistics
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0

        self._flattened = False

    def append(self, buffer):
        """
        Append a buffer of trajectories to memory.

        Args:
            buffer: Buffer object containing trajectory data
        """
        # Extract trajectory data from buffer
        # The buffer format depends on your TMRL setup
        # This is a generic implementation

        if hasattr(buffer, 'memory'):
            # Buffer contains trajectory data
            for trajectory in buffer.memory:
                self.trajectories.append(trajectory)
        else:
            # Buffer is a single trajectory
            self.trajectories.append(buffer)

        # Update statistics
        if hasattr(buffer, 'stat_test_return'):
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_steps = buffer.stat_test_steps
            self.stat_train_steps = buffer.stat_train_steps

        self._flattened = False
        logging.debug(f"Buffer appended. Total trajectories: {len(self.trajectories)}")

    def flatten_trajectories(self):
        """
        Flatten all stored trajectories into arrays for efficient batching.

        This should be called before training begins to prepare the data.
        """
        if self._flattened:
            return

        # Clear previous flattened data
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

        # Flatten all trajectories
        for trajectory in self.trajectories:
            # Each trajectory should contain:
            # - observations, actions, rewards, log_probs, values, dones
            # The exact format depends on your data collection

            if isinstance(trajectory, dict):
                self.observations.extend(trajectory.get('observations', []))
                self.actions.extend(trajectory.get('actions', []))
                self.rewards.extend(trajectory.get('rewards', []))
                self.log_probs.extend(trajectory.get('log_probs', []))
                self.values.extend(trajectory.get('values', []))
                self.dones.extend(trajectory.get('dones', []))
                self.advantages.extend(trajectory.get('advantages', []))
                self.returns.extend(trajectory.get('returns', []))
            else:
                # Assume trajectory is a tuple/list
                # (observations, actions, rewards, log_probs, values, dones)
                if len(trajectory) >= 6:
                    obs, acts, rews, logp, vals, dones = trajectory[:6]
                    self.observations.extend(obs)
                    self.actions.extend(acts)
                    self.rewards.extend(rews)
                    self.log_probs.extend(logp)
                    self.values.extend(vals)
                    self.dones.extend(dones)

        self._flattened = True
        logging.debug(f"Trajectories flattened: {len(self.observations)} timesteps")

    def __len__(self):
        """
        Return the number of complete trajectories stored.

        Returns:
            int: number of trajectories
        """
        return len(self.trajectories)

    def __iter__(self):
        """
        Iterate over batches of trajectory data.

        Yields:
            tuple: Batched tensors (observations, actions, log_probs, advantages, returns, values)
        """
        # Flatten trajectories if not already done
        self.flatten_trajectories()

        total_timesteps = len(self.observations)
        if total_timesteps == 0:
            logging.warning("No data in memory to iterate over")
            return

        # Generate random indices for shuffling
        indices = np.random.permutation(total_timesteps)

        # Yield batches
        for start_idx in range(0, total_timesteps, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_timesteps)
            batch_indices = indices[start_idx:end_idx]

            # Gather batch data
            batch_obs = [self.observations[i] for i in batch_indices]
            batch_acts = [self.actions[i] for i in batch_indices]
            batch_logp = [self.log_probs[i] for i in batch_indices]
            batch_vals = [self.values[i] for i in batch_indices]

            # Handle advantages and returns (may be empty initially)
            batch_adv = [self.advantages[i] for i in batch_indices] if self.advantages else [0] * len(batch_indices)
            batch_ret = [self.returns[i] for i in batch_indices] if self.returns else [0] * len(batch_indices)

            # Collate to tensors
            batch = self.collate(
                batch_obs, batch_acts, batch_logp, batch_adv, batch_ret, batch_vals
            )

            yield batch

    def collate(self, observations, actions, log_probs, advantages, returns, values):
        """
        Collate lists of data into tensors on the device.

        Args:
            observations: list of observations
            actions: list of actions
            log_probs: list of log probabilities
            advantages: list of advantages
            returns: list of returns
            values: list of state values

        Returns:
            dict: Dictionary of tensors
        """
        # Convert to tensors and move to device
        obs_tensor = self._to_tensor(observations)
        act_tensor = self._to_tensor(actions)
        logp_tensor = self._to_tensor(log_probs)
        adv_tensor = self._to_tensor(advantages)
        ret_tensor = self._to_tensor(returns)
        val_tensor = self._to_tensor(values)

        return {
            'observations': obs_tensor,
            'actions': act_tensor,
            'log_probs': logp_tensor,
            'advantages': adv_tensor,
            'returns': ret_tensor,
            'values': val_tensor
        }

    def _to_tensor(self, data):
        """
        Convert data to tensor and move to device.

        Args:
            data: list or array of data

        Returns:
            torch.Tensor: tensor on the specified device
        """
        if isinstance(data[0], torch.Tensor):
            return torch.stack(data).to(self.device)
        elif isinstance(data[0], np.ndarray):
            return torch.from_numpy(np.array(data)).to(self.device)
        else:
            return torch.tensor(data).to(self.device)

    def clear(self):
        """
        Clear all stored trajectories.

        This should be called after training to maintain the on-policy constraint.
        """
        self.trajectories.clear()
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self._flattened = False
        logging.debug("Memory cleared")

    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation (GAE) for all trajectories.

        This should be called before training to compute advantages and returns.

        Args:
            gamma (float): discount factor
            gae_lambda (float): GAE lambda parameter

        Note:
            This assumes trajectories are stored with rewards, values, and dones.
        """
        self.advantages = []
        self.returns = []

        for trajectory in self.trajectories:
            if isinstance(trajectory, dict):
                rewards = trajectory.get('rewards', [])
                values = trajectory.get('values', [])
                dones = trajectory.get('dones', [])
            else:
                # Assume tuple format
                rewards = trajectory[2] if len(trajectory) > 2 else []
                values = trajectory[4] if len(trajectory) > 4 else []
                dones = trajectory[5] if len(trajectory) > 5 else []

            if not rewards or not values:
                continue

            # Compute GAE
            traj_advantages = []
            traj_returns = []
            gae = 0
            next_value = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = 0
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]

                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae

                traj_advantages.insert(0, gae)
                traj_returns.insert(0, gae + values[t])

            self.advantages.extend(traj_advantages)
            self.returns.extend(traj_returns)

        logging.debug(f"GAE computed: {len(self.advantages)} advantages")

    def get_stats(self):
        """
        Get statistics about the current buffer.

        Returns:
            dict: Statistics dictionary
        """
        return {
            'num_trajectories': len(self.trajectories),
            'num_timesteps': len(self.observations) if self._flattened else sum(len(t.get('rewards', [])) if isinstance(t, dict) else len(t[2]) for t in self.trajectories),
            'test_return': self.stat_test_return,
            'train_return': self.stat_train_return,
            'test_steps': self.stat_test_steps,
            'train_steps': self.stat_train_steps
        }
