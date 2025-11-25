import argparse
import time
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import tmrl


# ============================================================
# CNN Architecture Configuration
# ============================================================
# Default TMRL configuration: 4 grayscale images of 64 x 64 pixels
imgs_buf_len = 4  # Number of stacked grayscale images
img_height = 64   # Image height in pixels
img_width = 64    # Image width in pixels


# ============================================================
# Helper Functions for CNN
# ============================================================

def conv2d_out_dims(conv_layer, h_in, w_in):
    """
    Calculate output dimensions of a Conv2d layer

    Args:
        conv_layer: nn.Conv2d layer
        h_in: input height
        w_in: input width

    Returns:
        (h_out, w_out): output dimensions
    """
    h_out = (h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) // conv_layer.stride[0] + 1
    w_out = (w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) // conv_layer.stride[1] + 1
    return h_out, w_out


def num_flat_features(x):
    """
    Calculate number of features in a flattened tensor

    Args:
        x: input tensor

    Returns:
        number of features
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Create a multi-layer perceptron

    Args:
        sizes: list of layer sizes
        activation: activation function
        output_activation: activation for output layer

    Returns:
        nn.Sequential MLP
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class TrackMasterCNN(nn.Module):
    """
    CNN (Convolutional Neural Network) model for processing TMRL observations

    Args:
        q_net (bool): indicates whether this neural net is a critic network
    """

    def __init__(self, q_net):
        super(TrackMasterCNN, self).__init__()

        self.q_net = q_net

        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the MLP input:
        # The MLP input will be formed of:
        # - the flattened CNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        # MLP layers:
        # (when using the model as a policy, we will sample from a multivariate gaussian defined later;
        # thus, the output dimensionality is 1 for the critic, and we will define the output layer of policies later)
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: input tuple (speed, gear, rpm, images, act1, act2) or
               (speed, gear, rpm, images, act1, act2, act) for critic

        Returns:
            output tensor
        """
        if self.q_net:
            # The critic takes the next action (act) as additional input
            # act1 and act2 are the actions in the action buffer (real-time RL):
            speed, gear, rpm, images, act1, act2, act = x
        else:
            # For the policy, the next action (act) is what we are computing, so we don't have it:
            speed, gear, rpm, images, act1, act2 = x

        speed = speed / 1000.0  # Max speed approx 1000
        rpm = rpm / 10000.0     # Max RPM approx 10000
        gear = gear / 6.0       # Max gear 6
        images = images / 255.0

        # we will stack these greyscale images along the channel dimension of our input tensor)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"

        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Finally, we can feed the result along our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            # print(speed.size(), gear.size(), rpm.size(), x.size(), act1.size(), act2.size())
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x


class Policy(nn.Module):
    """Policy network for action selection using CNN"""

    def __init__(self, action_space: int = 3):
        super().__init__()
        self.action_space = action_space

        # CNN backbone (q_net=False for policy)
        self.backbone = TrackMasterCNN(q_net=False)

        # Output layers for action distribution
        # The backbone outputs 256 features
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
        # Forward through CNN backbone
        features = self.backbone(observation)

        # Get action means
        means = self.action_mean(features)

        # Get log variance
        log_vars = self.actor_logvar(features)
        # Clamp log_variance to a safe range
        # min -20 (std ~0.00004), max 2 (std ~2.7)
        log_vars = torch.clamp(log_vars, min=-20, max=2)
        
        # Calculate variance (shared across action dimensions)
        vars = torch.zeros(features.shape[0], self.action_space).to(features.device)
        vars[:, :] = log_vars.exp().view(-1, 1)

        # Create covariance matrix (diagonal)
        covar_mat = torch.zeros(features.shape[0], self.action_space, self.action_space).to(features.device)
        covar_mat[:, np.arange(self.action_space), np.arange(self.action_space)] = vars

        # Create and return distribution
        dist = torch.distributions.MultivariateNormal(means, covar_mat)
        return dist


class Critic(nn.Module):
    """Critic network for state value estimation using CNN"""

    def __init__(self):
        super().__init__()

        # CNN backbone (q_net=False since we estimate V(s) not Q(s,a) for PPO)
        self.backbone = TrackMasterCNN(q_net=False)

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
        self.policy = Policy(action_space=action_space)
        self.critic = Critic()

    def forward(self, x):
        raise SyntaxError('Propagate through Agent.policy and Agent.critic individually')


def env_obs_to_tensor(observation, device='cpu'):
    """
    Convert TMRL observation tuple to tensor tuple for CNN

    Args:
        observation: TMRL observation tuple (speed, gear, rpm, images, act1, act2)
        device: torch device to place tensors on

    Returns:
        tuple of tensors: (speed, gear, rpm, images, act1, act2)
    """
    speed, gear, rpm, images, act1, act2 = observation

    # Convert each component to tensor with appropriate shape
    speed_t = torch.tensor(speed, dtype=torch.float32, device=device).unsqueeze(1)
    gear_t = torch.tensor(gear, dtype=torch.float32, device=device).unsqueeze(1)
    rpm_t = torch.tensor(rpm, dtype=torch.float32, device=device).unsqueeze(1)

    # Images: should be [batch, channels, height, width]
    # TMRL gives [4, 64, 64] (4 grayscale images)
    images_t = torch.tensor(images, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 4, 64, 64]

    # Actions: flatten to [batch, 3]
    act1_t = torch.tensor(act1, dtype=torch.float32, device=device).unsqueeze(0)      # [batch, 3]
    act2_t = torch.tensor(act2, dtype=torch.float32, device=device).unsqueeze(0)      # [batch, 3]

    return (speed_t, gear_t, rpm_t, images_t, act1_t, act2_t)


def batch_obs_to_tensor(observations, device='cpu'):
    """
    Convert batch of TMRL observations to tensor tuple for CNN

    Args:
        observations: list of TMRL observation tuples
        device: torch device to place tensors on

    Returns:
        tuple of batched tensors: (speed, gear, rpm, images, act1, act2)
    """
    batch_size = len(observations)

    # Separate components
    speeds, gears, rpms, images_list, act1s, act2s = [], [], [], [], [], []

    for obs in observations:
        speed, gear, rpm, images, act1, act2 = obs
        speeds.append(speed)
        gears.append(gear)
        rpms.append(rpm)
        images_list.append(images)
        act1s.append(act1)
        act2s.append(act2)

    # Convert to batched tensors
    speed_t = torch.tensor(speeds, dtype=torch.float32, device=device)
    gear_t = torch.tensor(gears, dtype=torch.float32, device=device)
    rpm_t = torch.tensor(rpms, dtype=torch.float32, device=device)

    # Images: stack to [batch, channels, height, width]
    images_t = torch.tensor(images_list, dtype=torch.float32, device=device)  # [batch, 4, 64, 64]

    # Actions
    act1_t = torch.tensor(act1s, dtype=torch.float32, device=device)     # [batch, 3]
    act2_t = torch.tensor(act2s, dtype=torch.float32, device=device)  # [batch, 3]

    return (speed_t, gear_t, rpm_t, images_t, act1_t, act2_t)


class RewardFunction:
    """Trajectory-based rewards for guided learning"""

    def __init__(self, trajectory, scale=0.01, max_dist=60.0,
                 check_forward=10, check_backward=10,
                 failure_countdown=10, min_steps=70):
        self.trajectory = trajectory
        self.scale = scale
        self.max_dist = max_dist
        self.check_forward = check_forward
        self.check_backward = check_backward
        self.failure_countdown = failure_countdown
        self.min_steps = min_steps
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0

    def compute_reward(self, position):
        """Compute reward based on progress along trajectory"""
        self.step_counter += 1
        min_dist = float('inf')
        best_idx = self.cur_idx

        # Search forward for best matching position
        for i in range(self.cur_idx, min(len(self.trajectory), self.cur_idx + self.check_forward)):
            dist = np.linalg.norm(position - self.trajectory[i])
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # If too far from trajectory, no reward
        if min_dist > self.max_dist:
            best_idx = self.cur_idx

        # Reward proportional to progress
        reward = (best_idx - self.cur_idx) * self.scale

        # If no progress, check backward and count failures
        if best_idx == self.cur_idx:
            for i in range(max(0, self.cur_idx - self.check_backward), self.cur_idx):
                dist = np.linalg.norm(position - self.trajectory[i])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            if self.step_counter > self.min_steps:
                self.failure_counter += 1
        else:
            self.failure_counter = 0

        self.cur_idx = best_idx
        terminated = self.failure_counter > self.failure_countdown
        return reward, terminated

    def reset(self):
        """Reset for new episode"""
        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0


class FullyLocalTrainer:
    """Fully local trainer - collects episodes and trains PPO on local hardware"""

    def __init__(
        self,
        max_episode_steps: int = 2400,
        checkpoint_dir: str = "./checkpoints",
        trajectory_path: str = None,
        device: str = None,
        # Hyperparameters
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.996,
        clip_coef: float = 0.2,
        critic_coef: float = 0.1,
        entropy_coef: float = 0.1,
        batch_size: int = 128,
        epochs_per_update: int = 50,
        hidden_dim: int = 32,
        norm_advantages: bool = True,
        grad_clip_val: float = 0.5,
        initial_std: float = 1.0,
        avg_ray: float = 400.0,
        # Trajectory reward parameters
        reward_scale: float = 0.01,
        max_dist_from_traj: float = 60.0,
        check_forward: int = 10,
        check_backward: int = 10,
        failure_countdown: int = 10,
        min_steps_before_failure: int = 70
    ):
        """
        Initialize fully local trainer

        Args:
            max_episode_steps: Maximum steps per episode
            checkpoint_dir: Directory to save model checkpoints
            trajectory_path: Path to trajectory file for guided learning (optional)
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            policy_lr: Learning rate for policy network
            critic_lr: Learning rate for critic network
            gamma: Discount factor
            clip_coef: PPO clipping coefficient
            critic_coef: Critic loss coefficient
            entropy_coef: Entropy bonus coefficient
            batch_size: Mini-batch size for training
            epochs_per_update: Number of epochs per episode
            hidden_dim: Hidden layer dimension
            norm_advantages: Whether to normalize advantages
            grad_clip_val: Gradient clipping value
            initial_std: Initial standard deviation for policy
            avg_ray: Average ray value for normalization
            reward_scale: Scale factor for trajectory progress rewards
            max_dist_from_traj: Max distance from trajectory before reward = 0
            check_forward: Allow cuts up to N positions ahead
            check_backward: Allow rewinding up to N positions back
            failure_countdown: Terminate after N steps with no progress
            min_steps_before_failure: Minimum steps before termination
        """
        self.max_episode_steps = max_episode_steps
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Hyperparameters
        self.hyper_params = {
            'policy_lr': policy_lr,
            'critic_lr': critic_lr,
            'gamma': gamma,
            'clip_coef': clip_coef,
            'critic_coef': critic_coef,
            'entropy_coef': entropy_coef,
            'batch_size': batch_size,
            'epochs_per_update': epochs_per_update,
            'hidden_dim': hidden_dim,
            'norm_advantages': norm_advantages,
            'grad_clip_val': grad_clip_val,
            'initial_std': initial_std,
            'avg_ray': avg_ray
        }

        # Initialize TMRL environment
        print("Initializing TMRL environment...")
        self.env = tmrl.get_environment()
        print(f"Observation Space: {self.env.observation_space}")
        print(f"Action Space: {self.env.action_space}")

        # Observation and action space sizes
        self.observation_space = 49161  # TMRL default
        self.action_space = 3  # Steering, throttle, brake

        # Initialize agent
        print("Initializing agent...")
        self.agent = Agent(action_space=self.action_space).to(self.device)

        # Optimizers
        self.policy_optim = torch.optim.Adam(
            self.agent.policy.parameters(),
            lr=self.hyper_params['policy_lr']
        )
        self.critic_optim = torch.optim.Adam(
            self.agent.critic.parameters(),
            lr=self.hyper_params['critic_lr']
        )

        print(f"Policy parameters: {sum(p.numel() for p in self.agent.policy.parameters()):,}")
        print(f"Critic parameters: {sum(p.numel() for p in self.agent.critic.parameters()):,}")

        # Load trajectory if provided
        self.trajectory_reward_fn = None
        if trajectory_path is not None and os.path.exists(trajectory_path):
            print(f"\n{'='*60}")
            print("Loading Trajectory for Guided Learning")
            print(f"{'='*60}")
            try:
                with open(trajectory_path, 'rb') as f:
                    trajectory_data = pickle.load(f)
                print(f"✓ Loaded trajectory with {len(trajectory_data)} positions")

                self.trajectory_reward_fn = RewardFunction(
                    trajectory_data,
                    scale=reward_scale,
                    max_dist=max_dist_from_traj,
                    check_forward=check_forward,
                    check_backward=check_backward,
                    failure_countdown=failure_countdown,
                    min_steps=min_steps_before_failure
                )

                print(f"✓ Trajectory reward function initialized")
                print(f"  Reward scale: {reward_scale}")
                print(f"  Max distance from trajectory: {max_dist_from_traj}m")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"✗ Failed to load trajectory: {e}")
                print("  Continuing without trajectory-based rewards")
                self.trajectory_reward_fn = None
        else:
            print(f"\n⚠ No trajectory provided - using standard rewards only\n")

        # Statistics
        self.update_count = 0
        self.total_steps = 0
        self.training_history = []

    def get_action(self, observation: Tuple) -> Tuple[np.ndarray, float, float]:
        """
        Get action from policy

        Args:
            observation: TMRL observation tuple

        Returns:
            action: numpy array [3] with values in [-1, 1]
            logprob: log probability of the action
            state_value: critic's estimate of state value
        """
        # Convert observation to tensor tuple
        obs_tensor = env_obs_to_tensor(observation, device=self.device)

        # Get action from policy
        self.agent.eval()
        with torch.no_grad():
            action, logprob = self.agent.policy.sample_action_with_logprobs(obs_tensor)
            state_value = self.agent.critic(obs_tensor)

        return action[0].cpu().numpy(), logprob[0].cpu().item(), state_value[0, 0].cpu().item()

    def collect_episode(self) -> Dict[str, Any]:
        """
        Collect one complete episode using current policy

        Returns:
            episode_data: Dictionary containing observations, actions, logprobs, rewards, state_values, positions
        """
        print("\n  Collecting episode...")

        # Buffers for episode data
        observations = []
        actions = []
        logprobs = []
        rewards = []
        state_values = []
        positions = []  # Store positions for trajectory-based rewards

        # Reset environment
        obs = self.env.reset()[0]

        step_count = 0
        done = False
        episode_reward = 0.0

        start_time = time.time()

        while not done and step_count < self.max_episode_steps:
            # Get action from policy
            action, logprob, state_value = self.get_action(obs)

            # Store trajectory data
            observations.append(obs)
            actions.append(action)
            logprobs.append(logprob)
            state_values.append(state_value)

            # Extract position from TMRL client (for trajectory-based rewards)
            try:
                data = self.env.unwrapped.interface.client.retrieve_data(sleep_if_empty=0.01, timeout=0.1)
                position = np.array([data[2], data[3], data[4]], dtype=np.float32)  # x, y, z
                positions.append(position)
            except Exception as e:
                # If position unavailable, append zeros (only warn on first occurrence)
                if step_count == 0:
                    print(f"    Warning: Could not retrieve position data: {e}")
                    print(f"    Trajectory rewards will not be available for this episode")
                positions.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))

            # Clip action to valid range
            clamped_action = np.clip(action, -1, 1)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(clamped_action)

            # Custom termination: stuck on rail (low LIDAR readings)
            # next_obs[2] is RPM/LIDAR data
            if next_obs[2][next_obs[2] <= 40].sum() > 0:
                terminated = True

            rewards.append(reward)
            episode_reward += reward

            obs = next_obs
            done = terminated or truncated
            step_count += 1

            # Print progress every 500 steps
            if step_count % 500 == 0:
                elapsed = time.time() - start_time
                print(f"    Step {step_count}/{self.max_episode_steps} - Reward: {episode_reward:.2f} - Time: {elapsed:.1f}s")

        # Pause environment (TMRL requirement)
        self.env.unwrapped.wait()  # type: ignore

        elapsed = time.time() - start_time
        self.total_steps += step_count

        print(f"  ✓ Episode collected: {step_count} steps, reward: {episode_reward:.2f}, time: {elapsed:.1f}s")

        return {
            'observations': observations,
            'actions': actions,
            'logprobs': logprobs,
            'rewards': rewards,
            'state_values': state_values,
            'positions': positions,
            'episode_length': step_count,
            'episode_reward': episode_reward
        }

    def train_on_episode(self, episode_data: Dict[str, Any]):
        """
        Train PPO for multiple epochs on one episode

        Args:
            episode_data: Dictionary containing episode trajectory
        """
        print(f"\n{'='*60}")
        print(f"Training Update {self.update_count + 1}")
        print(f"{'='*60}")

        episode_length = episode_data['episode_length']
        episode_reward = episode_data['episode_reward']

        print(f"  Episode length: {episode_length} steps")
        print(f"  Episode reward: {episode_reward:.2f}")

        # Extract episode data
        observations = episode_data['observations']
        actions = torch.tensor(np.array(episode_data['actions']), dtype=torch.float32)
        old_logprobs = torch.tensor(episode_data['logprobs'], dtype=torch.float32)
        rewards = torch.tensor(episode_data['rewards'], dtype=torch.float32)
        state_values = torch.tensor(episode_data['state_values'], dtype=torch.float32)

        # Add trajectory-based rewards if available
        if self.trajectory_reward_fn is not None and 'positions' in episode_data:
            print(f"Computing trajectory-based rewards...")
            self.trajectory_reward_fn.reset()
            trajectory_rewards = []

            for position in episode_data['positions']:
                traj_reward, _ = self.trajectory_reward_fn.compute_reward(position)
                trajectory_rewards.append(traj_reward)

            trajectory_rewards = torch.tensor(trajectory_rewards, dtype=torch.float32)
            total_traj_reward = trajectory_rewards.sum().item()

            # Combine original rewards with trajectory rewards
            rewards = rewards + trajectory_rewards

            print(f"Trajectory reward contribution: {total_traj_reward:.4f}")
            print(f"Combined episode reward: {rewards.sum().item():.2f}")
            episode_data['rewards'] = rewards.sum().item()

        # Compute returns (discounted cumulative rewards)
        returns = torch.zeros(episode_length)
        with torch.no_grad():
            for t in range(episode_length - 1, -1, -1):
                if t == episode_length - 1:
                    returns[t] = rewards[t]
                else:
                    returns[t] = rewards[t] + self.hyper_params['gamma'] * returns[t + 1]

            # Compute advantages
            advantages = returns - state_values

        print(f"  Mean advantage: {advantages.mean():.4f}")
        print(f"  Mean return: {returns.mean():.4f}")

        # Training metrics
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_total_losses = []

        # Set models to training mode
        self.agent.train()

        # Train for multiple epochs
        for epoch in range(self.hyper_params['epochs_per_update']):
            # Random permutation for mini-batches
            rand_idxs = np.random.permutation(episode_length)

            # Mini-batch updates
            for batch_start in range(0, episode_length, self.hyper_params['batch_size']):
                batch_end = min(batch_start + self.hyper_params['batch_size'], episode_length)
                batch_idxs = rand_idxs[batch_start:batch_end]

                # Extract batch
                batch_obs = [observations[i] for i in batch_idxs]
                batch_obs_tensor = batch_obs_to_tensor(batch_obs, device=self.device)
                batch_actions = actions[batch_idxs].to(self.device)
                batch_old_logprobs = old_logprobs[batch_idxs].to(self.device)
                batch_returns = returns[batch_idxs].to(self.device)
                batch_advantages = advantages[batch_idxs].to(self.device)

                # Normalize advantages
                if self.hyper_params['norm_advantages']:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # ========== PPO Actor Loss ==========
                # Get new log probabilities
                batch_new_logprobs = self.agent.policy.get_action_log_prob(batch_obs_tensor, batch_actions)

                # Importance sampling ratio
                log_ratio = batch_new_logprobs - batch_old_logprobs
                ratio = torch.exp(log_ratio)

                # PPO clipped surrogate objective
                unclipped_obj = -ratio * batch_advantages
                clipped_obj = -torch.clamp(
                    ratio,
                    1 - self.hyper_params['clip_coef'],
                    1 + self.hyper_params['clip_coef']
                ) * batch_advantages
                ppo_loss = torch.max(unclipped_obj, clipped_obj).sum() / len(batch_idxs) # max because we are calculating as loss so sign flips

                # ========== Critic Loss ==========
                new_values = self.agent.critic(batch_obs_tensor).squeeze()
                v_loss = ((new_values - batch_returns) ** 2).sum() / len(batch_idxs)

                # ========== Combined Loss ==========
                total_loss = ppo_loss + self.hyper_params['critic_coef'] * v_loss

                # ========== Optimization ==========
                self.policy_optim.zero_grad()
                self.critic_optim.zero_grad()

                total_loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_value_(
                    self.agent.policy.parameters(),
                    clip_value=self.hyper_params['grad_clip_val']
                )

                # Handle NaN gradients
                for param in self.agent.policy.parameters():
                    if param.grad is not None:
                        mask = torch.isnan(param.grad)
                        param.grad[mask] = 0.0

                nn.utils.clip_grad_value_(
                    self.agent.critic.parameters(),
                    clip_value=self.hyper_params['grad_clip_val']
                )

                self.policy_optim.step()
                self.critic_optim.step()

                # Record losses
                epoch_actor_losses.append(ppo_loss.item())
                epoch_critic_losses.append(v_loss.item())
                epoch_total_losses.append(total_loss.item())

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{self.hyper_params['epochs_per_update']} - Loss: {np.mean(epoch_total_losses[-10:]):.4f}")

        # Update statistics
        self.update_count += 1

        metrics = {
            'update': int(self.update_count),
            'reward': float(episode_reward),
            'steps': int(episode_length),
            'actor_loss': float(np.mean(epoch_actor_losses)),
            'critic_loss': float(np.mean(epoch_critic_losses)),
            'total_loss': float(np.mean(epoch_total_losses))
        }
        self.training_history.append(metrics)

        print(f"\n  Training complete:")
        print(f"    Actor loss: {metrics['actor_loss']:.4f}")
        print(f"    Critic loss: {metrics['critic_loss']:.4f}")
        print(f"    Total loss: {metrics['total_loss']:.4f}")

        # Save checkpoint if good reward
        if episode_reward > 200:
            checkpoint_name = f"Y{episode_reward:.2f}RewardRacer{self.update_count}Update.pt"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            torch.save(self.agent.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_name}")

        # Periodic checkpoint
        if self.update_count % 50 == 0:
            checkpoint_name = f"checkpoint_update{self.update_count}.pt"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            torch.save({
                'agent_state_dict': self.agent.state_dict(),
                'policy_optim_state_dict': self.policy_optim.state_dict(),
                'critic_optim_state_dict': self.critic_optim.state_dict(),
                'update_count': self.update_count,
                'training_history': self.training_history,
                'hyper_params': self.hyper_params
            }, checkpoint_path)
            print(f"  ✓ Periodic checkpoint saved: {checkpoint_name}")

        print(f"{'='*60}\n")

        # Synchronize CUDA if using GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'agent_state_dict' in checkpoint:
            # Full checkpoint with optimizer states
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
            self.update_count = checkpoint['update_count']
            self.training_history = checkpoint['training_history']
            print(f"  ✓ Loaded checkpoint from update {self.update_count}")
        else:
            # Simple checkpoint with just model weights
            self.agent.load_state_dict(checkpoint)
            print(f"  ✓ Loaded model weights")

    def train_loop(self, num_updates: int = 10000):
        """
        Main training loop

        Args:
            num_updates: Number of episodes to collect and train on
        """
        print(f"\n{'='*60}")
        print(f"Starting Fully Local Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Number of updates: {num_updates}")
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Trajectory-based rewards: {'ENABLED' if self.trajectory_reward_fn else 'DISABLED'}")
        print(f"{'='*60}\n")

        # Training loop
        for update in range(num_updates):
            print(f"\n{'='*60}")
            print(f"Update {update + 1}/{num_updates}")
            print(f"{'='*60}")

            start_time = time.time()

            # 1. Collect episode locally
            episode_data = self.collect_episode()

            # 2. Train on episode (synchronous - no network delay!)
            self.train_on_episode(episode_data)

            elapsed = time.time() - start_time

            print(f"\n{'='*60}")
            print(f"Update {self.update_count} complete")
            print(f"  Episode reward: {episode_data['episode_reward']:.2f}")
            print(f"  Episode length: {episode_data['episode_length']} steps")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Update time: {elapsed:.1f}s")
            print(f"{'='*60}")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total updates: {self.update_count}")
        print(f"  Total steps: {self.total_steps}")
        print(f"{'='*60}\n")

        # Save final checkpoint
        final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_checkpoint.pt")
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'update_count': self.update_count,
            'training_history': self.training_history,
            'hyper_params': self.hyper_params
        }, final_checkpoint_path)
        print(f"✓ Final checkpoint saved: {final_checkpoint_path}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fully local TrackMania PPO trainer")

    # Training parameters
    parser.add_argument(
        '--num-updates',
        type=int,
        default=10000,
        help='Number of episodes to collect and train on (default: 10000)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=2400,
        help='Maximum steps per episode (default: 2400)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints (default: ./checkpoints)'
    )
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (optional)'
    )
    parser.add_argument(
        '--trajectory-path',
        type=str,
        default=None,
        help='Path to trajectory file for guided learning (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    # Hyperparameters
    parser.add_argument('--policy-lr', type=float, default=3e-4, help='Policy learning rate')
    parser.add_argument('--critic-lr', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.996, help='Discount factor')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clipping coefficient')
    parser.add_argument('--critic-coef', type=float, default=0.1, help='Critic loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.1, help='Entropy coefficient')
    parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--epochs-per-update', type=int, default=50, help='Epochs per update')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden layer dimension')
    parser.add_argument('--grad-clip-val', type=float, default=0.5, help='Gradient clipping value')

    # Trajectory reward parameters
    parser.add_argument('--reward-scale', type=float, default=0.01, help='Trajectory reward scale')
    parser.add_argument('--max-dist-from-traj', type=float, default=60.0, help='Max distance from trajectory')
    parser.add_argument('--check-forward', type=int, default=10, help='Check forward positions')
    parser.add_argument('--check-backward', type=int, default=10, help='Check backward positions')
    parser.add_argument('--failure-countdown', type=int, default=10, help='Failure countdown')
    parser.add_argument('--min-steps-before-failure', type=int, default=70, help='Min steps before failure')

    args = parser.parse_args()

    # Create trainer
    trainer = FullyLocalTrainer(
        max_episode_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        trajectory_path=args.trajectory_path,
        device=args.device,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        clip_coef=args.clip_coef,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size,
        epochs_per_update=args.epochs_per_update,
        hidden_dim=args.hidden_dim,
        grad_clip_val=args.grad_clip_val,
        reward_scale=args.reward_scale,
        max_dist_from_traj=args.max_dist_from_traj,
        check_forward=args.check_forward,
        check_backward=args.check_backward,
        failure_countdown=args.failure_countdown,
        min_steps_before_failure=args.min_steps_before_failure
    )

    # Load checkpoint if provided
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    # Start training loop
    try:
        trainer.train_loop(num_updates=args.num_updates)
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")

        # Save interrupt checkpoint
        interrupt_checkpoint_path = os.path.join(args.checkpoint_dir, "interrupt_checkpoint.pt")
        torch.save({
            'agent_state_dict': trainer.agent.state_dict(),
            'policy_optim_state_dict': trainer.policy_optim.state_dict(),
            'critic_optim_state_dict': trainer.critic_optim.state_dict(),
            'update_count': trainer.update_count,
            'training_history': trainer.training_history,
            'hyper_params': trainer.hyper_params
        }, interrupt_checkpoint_path)
        print(f"✓ Interrupt checkpoint saved: {interrupt_checkpoint_path}")
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
