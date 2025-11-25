"""
Example usage of TrainingOnline with PPO for TrackMania.

This example shows how to integrate your existing PPO implementation
from TrackManiaPPO.ipynb with the TMRL distributed training framework
using the custom TrainingOnline class.
"""

import torch
import torch.nn as nn
import numpy as np
import tmrl

from training_online import TorchTrainingOnline
from online_memory import OnlineMemory
from ppo_training_agent import PPOTrainingAgentContinuous


# ============================================================================
# STEP 1: Define your Policy and Value networks
# (adapted from your notebook)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network for PPO (returns a Gaussian distribution).
    """
    def __init__(self, observation_dim, action_dim, hidden_dim=512, avg_ray=400, initial_std=1.0):
        super().__init__()
        self.avg_ray = avg_ray

        # Mean network
        self.action_mean = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Log std network (learns variance)
        self.actor_logvar = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        """
        Forward pass returns a MultivariateNormal distribution.

        Args:
            observation: batch of observations

        Returns:
            torch.distributions.MultivariateNormal: action distribution
        """
        # Normalize observation
        observation = observation / self.avg_ray

        # Compute mean
        means = self.action_mean(observation)

        # Compute covariance matrix (diagonal)
        action_dim = means.shape[-1]
        vars = torch.zeros(observation.shape[0], action_dim).to(observation.device)
        vars[:, :] = self.actor_logvar(observation).exp().view(-1, 1)

        # Create diagonal covariance matrix
        covar_mat = torch.zeros(observation.shape[0], action_dim, action_dim).to(observation.device)
        covar_mat[:, np.arange(action_dim), np.arange(action_dim)] = vars

        # Return distribution
        dist = torch.distributions.MultivariateNormal(means, covar_mat)
        return dist


class ValueNetwork(nn.Module):
    """
    Value network for PPO (estimates state value).
    """
    def __init__(self, observation_dim, hidden_dim=512, avg_ray=400):
        super().__init__()
        self.avg_ray = avg_ray

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observation):
        """
        Forward pass returns state value.

        Args:
            observation: batch of observations

        Returns:
            torch.Tensor: state values
        """
        observation = observation / self.avg_ray
        return self.network(observation)


# ============================================================================
# STEP 2: Create custom PPO agent for your specific networks
# ============================================================================

class TrackManiaPPOAgent(PPOTrainingAgentContinuous):
    """
    Custom PPO agent using your TrackMania networks.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 hidden_dim=512,
                 avg_ray=400,
                 policy_lr=1e-5,
                 value_lr=1e-5,
                 clip_epsilon=0.2,
                 value_loss_coef=0.1,
                 entropy_coef=0.1,
                 max_grad_norm=0.1):

        # Calculate observation dimension
        observation_dim = np.sum([np.prod(value.shape) for value in observation_space])
        action_dim = action_space.shape[0]

        # Create networks
        policy_network = PolicyNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            avg_ray=avg_ray
        )

        value_network = ValueNetwork(
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            avg_ray=avg_ray
        )

        # Initialize parent
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            policy_network=policy_network,
            value_network=value_network,
            policy_lr=policy_lr,
            value_lr=value_lr,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm
        )


# ============================================================================
# STEP 3: Set up and run distributed training
# ============================================================================

def main():
    """
    Main training function.
    """
    # Hyperparameters (from your notebook)
    hyper_params = {
        'policy_lr': 1e-5,
        'value_lr': 1e-5,  # Renamed from critic_lr
        'gamma': 0.996,
        'clip_coef': 0.2,
        'value_coef': 0.1,  # Renamed from critic_coef
        'entropy_coef': 0.1,
        'batch_size': 256,
        'num_updates': 10000,  # Total epochs
        'update_epochs': 4,  # PPO epochs per batch (renamed from epochs_per_update)
        'hidden_dim': 512,
        'avg_ray': 400,
        'num_trajectories': 10,  # Number of trajectories to collect before training
        'max_grad_norm': 0.1  # Renamed from grad_clip_val
    }

    # Get environment to extract spaces
    env = tmrl.get_environment()
    observation_space = env.observation_space
    action_space = env.action_space

    print(f"Observation Space: {observation_space}")
    print(f"Action Space: {action_space}")

    # Create training configuration
    training = TorchTrainingOnline(
        env_cls=(observation_space, action_space),  # Pass as tuple
        memory_cls=OnlineMemory,
        training_agent_cls=lambda obs_space, act_space, dev: TrackManiaPPOAgent(
            observation_space=obs_space,
            action_space=act_space,
            device=dev,
            hidden_dim=hyper_params['hidden_dim'],
            avg_ray=hyper_params['avg_ray'],
            policy_lr=hyper_params['policy_lr'],
            value_lr=hyper_params['value_lr'],
            clip_epsilon=hyper_params['clip_coef'],
            value_loss_coef=hyper_params['value_coef'],
            entropy_coef=hyper_params['entropy_coef'],
            max_grad_norm=hyper_params['max_grad_norm']
        ),
        epochs=hyper_params['num_updates'],
        rounds=50,  # Number of rounds per epoch
        update_epochs=hyper_params['update_epochs'],  # PPO epochs per batch
        num_trajectories=hyper_params['num_trajectories'],
        update_model_interval=1,  # Update model after each round (on-policy)
        profiling=False,
        device=None,  # Auto-select
        clear_buffer_after_training=True,  # Important for on-policy
        min_trajectories_before_training=5
    )

    print("\n" + "="*70)
    print("TrainingOnline Configuration")
    print("="*70)
    print(f"Device: {training.device}")
    print(f"Total Epochs: {training.epochs}")
    print(f"Rounds per Epoch: {training.rounds}")
    print(f"PPO Update Epochs: {training.update_epochs}")
    print(f"Trajectories per Round: {training.num_trajectories}")
    print(f"Batch Size: {hyper_params['batch_size']}")
    print("="*70 + "\n")

    # Note: To actually run this, you need to set up the TMRL server/client infrastructure
    # This would involve:
    # 1. Starting a TMRL server (on Colab or local)
    # 2. Starting TMRL workers (local with TrackMania running)
    # 3. Running the training loop with the interface

    # Example (pseudo-code):
    # from tmrl import Server
    # server = Server()
    # interface = server.get_interface()
    #
    # for epoch in range(training.epochs):
    #     stats = training.run_epoch(interface)
    #     # Save model, log stats, etc.

    print("Training configuration created successfully!")
    print("\nNext steps:")
    print("1. Set up TMRL server (on Google Colab with GPU)")
    print("2. Configure TMRL workers (on local machine with TrackMania)")
    print("3. Run the training loop using training.run_epoch(interface)")

    return training


if __name__ == "__main__":
    training = main()
