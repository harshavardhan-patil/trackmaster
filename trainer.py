import argparse
import time
import pickle
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import tmrl
from src.agent import Agent

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
    def __init__(self, trajectory, scale, max_dist,
                 check_forward, check_backward,
                 failure_countdown, min_steps, debug=False):
        self.trajectory = trajectory
        self.scale = scale
        self.max_dist = max_dist
        self.check_forward = check_forward  
        self.check_backward = check_backward
        self.failure_countdown = failure_countdown
        self.min_steps = min_steps
        self.debug = debug

        # State tracking
        self.cur_idx = 0
        self.last_checkpoint = 0  # Track last valid checkpoint reached
        self.step_counter = 0
        self.failure_counter = 0
        self.prev_position = None
        self.last_progress_position = None  # Track position for distance-based failure detection
        self.distance_since_last_progress = 0.0 

    def compute_reward(self, position, speed, action, gear=1):
        """
        Args:
            position: [x, y, z] numpy array
            speed: current speed (float, approx km/h)
            action: [steering, throttle, brake] numpy array
            gear: current gear (0 = reverse, 1+ = forward gears)
        """
        self.step_counter += 1

        # 1. FIND TRAJECTORY INDEX (CHECKPOINT-BASED)
        # ------------------------------------------------
        # Only search from last_checkpoint to last_checkpoint + check_forward
        # This prevents reward hacking by cutting the track or reversing to nearby checkpoints
        min_dist = float('inf')
        best_idx = self.last_checkpoint

        # Search only in the next few checkpoints from last valid checkpoint
        search_start = self.last_checkpoint
        search_end = min(len(self.trajectory), self.last_checkpoint + self.check_forward)

        if self.debug and self.step_counter % 100 == 0:
            print(f"\n[DEBUG Step {self.step_counter}] Checkpoint Search:")
            print(f"  Searching checkpoints {search_start} to {search_end} (range: {search_end - search_start})")
            print(f"  Car position: {position}")
            print(f"  Current checkpoint idx: {self.cur_idx}, Last valid: {self.last_checkpoint}")

        for i in range(search_start, search_end):
            dist = np.linalg.norm(position - self.trajectory[i])
            if self.debug and self.step_counter % 100 == 0:
                print(f"  Checkpoint {i}: dist={dist:.2f}m")
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        if self.debug and self.step_counter % 100 == 0:
            print(f"  Best checkpoint: {best_idx}, distance: {min_dist:.2f}m (max allowed: {self.max_dist}m)")

        # If we found a valid checkpoint ahead, update last_checkpoint
        old_last_checkpoint = self.last_checkpoint
        if best_idx > self.last_checkpoint and min_dist < self.max_dist:
            self.last_checkpoint = best_idx

        # Use last_checkpoint as current index (prevents backwards movement)
        best_idx = self.last_checkpoint

        if self.debug and self.step_counter % 100 == 0 and self.last_checkpoint != old_last_checkpoint:
            print(f"  ✓ Progress! Checkpoint {old_last_checkpoint} -> {self.last_checkpoint}") 

        # 2. CALCULATE VECTORS
        # ------------------------------------------------
        # Track Vector (Where we SHOULD be going)
        # Look ahead 10 checkpoints to get meaningful direction (handles high-res trajectories)
        lookahead = min(10, len(self.trajectory) - best_idx - 1)
        if lookahead > 0:
            track_vec = self.trajectory[best_idx + lookahead] - self.trajectory[best_idx]
        else:
            track_vec = np.array([0, 0, 0])

        # Car Vector (Where we ARE going)
        if self.prev_position is not None:
            car_vec = position - self.prev_position
        else:
            car_vec = np.array([0, 0, 0])

        # 3. CALCULATE REWARDS
        # ------------------------------------------------

        # A. Progress Reward (Base)
        progress_reward = (best_idx - self.cur_idx) * self.scale

        if self.debug and self.step_counter % 100 == 0:
            print(f"\n[DEBUG Step {self.step_counter}] Reward Calculation:")
            print(f"  Progress: (best_idx={best_idx} - cur_idx={self.cur_idx}) * scale={self.scale}")
            print(f"  Progress reward (raw): {progress_reward:.4f}")

        # B. Effective Speed Reward (Velocity along the line)
        alignment = 0.0
        norm_track = np.linalg.norm(track_vec)
        norm_car = np.linalg.norm(car_vec)

        if norm_track > 0 and norm_car > 0:
            # Cosine similarity
            alignment = np.dot(track_vec, car_vec) / (norm_track * norm_car)

        # Scale down speed (0-1000) to reasonable reward size (e.g. 0-2.0)
        # Only reward speed if alignment is positive
        if alignment > 0:
            effective_speed_reward = (speed * alignment) * 0.001
        else:
            effective_speed_reward = 0.0

        if self.debug and self.step_counter % 100 == 0:
            print(f"  Track vector: {track_vec} (norm: {norm_track:.2f})")
            print(f"  Car vector: {car_vec} (norm: {norm_car:.2f})")
            print(f"  Alignment (cosine): {alignment:.4f}")
            print(f"  Speed: {speed:.2f} km/h")
            print(f"  Speed reward (raw): {effective_speed_reward:.4f}")

        # C. Gas Bias
        # If throttle (action[1]) > 0.8
        gas_bonus = 0.0
        if action[1] > 0.8:
            gas_bonus = 0.5 * self.scale

        # D. Reverse Tax
        # Penalty for reversing to avoid reward hacking
        # gear == 0 means reverse gear
        reverse_tax = 0.0
        if gear == 0:
            reverse_tax = speed * self.scale

        # E. Parking Fine
        # If we are 2 seconds into the race and stopped, punish idleness.
        parking_fine = 0.0
        if self.step_counter > 40 and abs(speed) < 1.0:
            parking_fine = 0.05

        if self.debug and self.step_counter % 100 == 0:
            print(f"  Gas bonus: {gas_bonus:.4f} (throttle={action[1]:.2f})")
            print(f"  Reverse tax: {reverse_tax:.4f} (gear={gear})")
            print(f"  Parking fine: {parking_fine:.4f}")
            total_raw = progress_reward + effective_speed_reward + gas_bonus - reverse_tax - parking_fine
            print(f"  Total (alpha): {total_raw:.4f}")

        # 4. TOTAL & UPDATES
        # ------------------------------------------------
        total_reward = (progress_reward
                        + effective_speed_reward
                        + gas_bonus
                        - reverse_tax
                        - parking_fine)

        # Package individual components for logging
        reward_components = {
            'progress_reward': progress_reward,
            'effective_speed_reward': effective_speed_reward,
            'gas_bonus': gas_bonus,
            'reverse_tax': reverse_tax,
            'parking_fine': parking_fine
        }

        # Update state
        self.cur_idx = best_idx
        self.prev_position = position.copy()

        # Continue is step count less than exploratory min steps
        if self.step_counter < self.min_steps:
            return total_reward, False, reward_components

        # Failure check (Stuck logic)
        # Track if car is making progress via checkpoint advancement OR distance traveled
        made_progress = False

        # Check 1: Advanced to a new checkpoint?
        if best_idx > self.cur_idx:
            made_progress = True
            self.last_progress_position = position.copy()
            self.distance_since_last_progress = 0.0
        else:
            # Check 2: Traveled significant distance (>1m) without checkpoint advancement?
            # This handles slow movement through high-res trajectories
            if self.last_progress_position is not None:
                distance_traveled = np.linalg.norm(position - self.last_progress_position)
                if distance_traveled > 1.0:  # Reset if moved >1m
                    made_progress = True
                    self.last_progress_position = position.copy()
                    self.distance_since_last_progress = 0.0

        if made_progress:
            self.failure_counter = 0
        else:
            self.failure_counter += 1

        terminated = self.failure_counter > self.failure_countdown

        # Hard terminate if hopelessly lost
        if min_dist > (self.max_dist * 2.0):
            terminated = True

        return total_reward, terminated, reward_components

    def update_curriculum_params(self, failure_countdown: int, min_steps: int):
        """Update curriculum learning parameters"""
        self.failure_countdown = failure_countdown
        self.min_steps = min_steps

    def reset(self):
        self.cur_idx = 0
        self.last_checkpoint = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.prev_position = None
        self.last_progress_position = None
        self.distance_since_last_progress = 0.0


class FullyLocalTrainer:
    """Fully local trainer - collects episodes and trains PPO on local hardware"""

    def __init__(
        self,
        max_episode_steps: int = 2400,
        checkpoint_dir: str = "./checkpoints",
        trajectory_path: Optional[str] = None,
        device: Optional[str] = None,
        # Hyperparameters
        policy_lr: float = 1e-5,
        critic_lr: float = 1e-5,
        gamma: float = 0.996,
        clip_coef: float = 0.2,
        critic_coef: float = 0.1,
        entropy_coef: float = 0.1,
        batch_size: int = 128,
        epochs_per_update: int = 50,
        episodes_per_update: int = 4,
        hidden_dim: int = 32,
        norm_advantages: bool = True,
        grad_clip_val: float = 0.5,
        gae_lambda: float = 0.95,
        initial_std: float = 1.0,
        avg_ray: float = 400.0,
        # Trajectory reward parameters
        reward_scale: float = 0.1,
        max_dist_from_traj: float = 60.0,
        check_forward: int = 1000,
        check_backward: int = 10,
        # Curriculum learning parameters
        initial_failure_countdown: int = 1000,
        final_failure_countdown: int = 1500,
        initial_min_steps: int = 500,
        final_min_steps: int = 1500,
        curriculum_updates: int = 1000
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
            epochs_per_update: Number of epochs per training update
            episodes_per_update: Number of episodes to collect before each training update
            hidden_dim: Hidden layer dimension
            norm_advantages: Whether to normalize advantages
            grad_clip_val: Gradient clipping value
            gae_lambda: GAE lambda parameter for advantage estimation (0.0 = no GAE, 1.0 = Monte Carlo)
            initial_std: Initial standard deviation for policy
            avg_ray: Average ray value for normalization
            reward_scale: Scale factor for trajectory progress rewards
            max_dist_from_traj: Max distance from trajectory before reward = 0
            check_forward: Allow cuts up to N positions ahead
            check_backward: Allow rewinding up to N positions back
            failure_countdown: Terminate after N steps with no progress (used if no curriculum)
            min_steps_before_failure: Minimum steps before termination (used if no curriculum)
            initial_failure_countdown: Starting value for curriculum learning
            final_failure_countdown: Final value for curriculum learning
            initial_min_steps: Starting min steps for curriculum learning
            final_min_steps: Final min steps for curriculum learning
            curriculum_updates: Number of updates to linearly schedule curriculum over
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
            'episodes_per_update': episodes_per_update,
            'hidden_dim': hidden_dim,
            'norm_advantages': norm_advantages,
            'grad_clip_val': grad_clip_val,
            'gae_lambda': gae_lambda,
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
            print("Loading Custom Reward Function")
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
                    failure_countdown=initial_failure_countdown,
                    min_steps=initial_min_steps,
                    debug=False  # Disable debug logging
                )

                print(f"✓ Custom reward function initialized")
                print(f"  ⚠ TMRL rewards will be COMPLETELY IGNORED")
                print(f"  ⚠ Using ONLY custom trajectory-based rewards")
                print(f"  Reward scale: {reward_scale}")
                print(f"  Max distance from trajectory: {max_dist_from_traj}m")
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"✗ Failed to load trajectory: {e}")
                print("  Continuing with TMRL rewards")
                self.trajectory_reward_fn = None
        else:
            print(f"\n⚠ No trajectory provided - using TMRL rewards only\n")

        # Curriculum learning parameters
        self.curriculum_config = {
            'initial_failure_countdown': initial_failure_countdown,
            'final_failure_countdown': final_failure_countdown,
            'initial_min_steps': initial_min_steps,
            'final_min_steps': final_min_steps,
            'curriculum_updates': curriculum_updates
        }

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

    def update_curriculum(self):
        """
        Update curriculum learning parameters based on training progress.
        Linearly interpolates between initial and final values over curriculum_updates.
        """
        if self.trajectory_reward_fn is None:
            return  # No reward function to update

        # Calculate progress (clamped between 0 and 1)
        progress = min(1.0, self.update_count / self.curriculum_config['curriculum_updates'])

        # Linear interpolation
        current_failure_countdown = int(
            self.curriculum_config['initial_failure_countdown'] +
            progress * (self.curriculum_config['final_failure_countdown'] -
                       self.curriculum_config['initial_failure_countdown'])
        )

        current_min_steps = int(
            self.curriculum_config['initial_min_steps'] +
            progress * (self.curriculum_config['final_min_steps'] -
                       self.curriculum_config['initial_min_steps'])
        )

        # Update reward function
        self.trajectory_reward_fn.update_curriculum_params(
            failure_countdown=current_failure_countdown,
            min_steps=current_min_steps
        )

        # Log every 10 updates
        if self.update_count % 10 == 0:
            print(f"  Curriculum update: failure_countdown={current_failure_countdown}, min_steps={current_min_steps}")

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
        speeds = []

        # Track individual reward components
        reward_component_totals = {
            'progress_reward': 0.0,
            'effective_speed_reward': 0.0,
            'gas_bonus': 0.0,
            'reverse_tax': 0.0,
            'parking_fine': 0.0
        }

        # Reset environment and custom reward function
        obs = self.env.reset()[0]
        if self.trajectory_reward_fn is not None:
            self.trajectory_reward_fn.reset()

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

            # Extract position, speed, and gear for custom reward function
            # Observation format: (speed, gear, rpm, images, act1, act2)
            speed = float(obs[0]) if hasattr(obs[0], '__iter__') else obs[0]
            gear = float(obs[1]) if hasattr(obs[1], '__iter__') else obs[1]
            position = None

            if self.trajectory_reward_fn is not None:
                try:
                    data = self.env.unwrapped.interface.client.retrieve_data(sleep_if_empty=0.01, timeout=0.1)
                    position = np.array([data[2], data[3], data[4]], dtype=np.float32)  # x, y, z
                    positions.append(position)
                except Exception as e:
                    # If position unavailable, append zeros (only warn on first occurrence)
                    if step_count == 0:
                        print(f"    Warning: Could not retrieve position data: {e}")
                        print(f"    Custom reward function will not be available for this episode")
                    position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    positions.append(position)

            # Clip action to valid range
            clamped_action = np.clip(action, -1, 1)

            # Step environment (we ignore TMRL reward when using custom reward function)
            next_obs, tmrl_reward, terminated, truncated, info = self.env.step(clamped_action)

            # Compute reward using custom reward function if available
            if self.trajectory_reward_fn is not None and position is not None:
                custom_reward, custom_terminated, reward_components = self.trajectory_reward_fn.compute_reward(
                    position, speed, action, gear
                )
                reward = float(custom_reward)  # Use ONLY custom reward, ignore TMRL reward
                terminated = custom_terminated or terminated

                # Accumulate individual components
                for key in reward_component_totals:
                    reward_component_totals[key] += reward_components[key]
            else:
                reward = float(tmrl_reward)  # Fall back to TMRL reward if no custom function

            rewards.append(reward)
            episode_reward += reward
            speeds.append(speed)

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
            'episode_reward': episode_reward,
            'speeds': speeds,
            'reward_components': reward_component_totals
        }

    def train_on_episodes(self, episodes_data: List[Dict[str, Any]]):
        """
        Train PPO for multiple epochs on multiple collected episodes

        Args:
            episodes_data: List of episode dictionaries
        """
        print(f"\n{'='*60}")
        print(f"Training Update {self.update_count + 1}")
        print(f"{'='*60}")

        # Calculate aggregate statistics
        total_steps = sum(ep['episode_length'] for ep in episodes_data)
        total_reward = sum(ep['episode_reward'] for ep in episodes_data)
        mean_reward = total_reward / len(episodes_data)

        print(f"  Training on {len(episodes_data)} episodes")
        print(f"  Total steps: {total_steps}")
        print(f"  Mean episode reward: {mean_reward:.2f}")

        # Combine all episodes' data
        all_observations = []
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_state_values = []
        all_returns = []
        all_advantages = []

        with torch.no_grad():
            # Process each episode separately for returns and advantages
            for episode_data in episodes_data:
                episode_length = episode_data['episode_length']

                # Extract episode data
                observations = episode_data['observations']
                actions = torch.tensor(np.array(episode_data['actions']), dtype=torch.float32)
                logprobs = torch.tensor(episode_data['logprobs'], dtype=torch.float32)
                rewards = torch.tensor(episode_data['rewards'], dtype=torch.float32)
                state_values = torch.tensor(episode_data['state_values'], dtype=torch.float32)

                # Compute returns and GAE advantages for this episode
                returns = torch.zeros(episode_length)
                advantages = torch.zeros(episode_length)

                # Compute returns (for value function training)
                for t in range(episode_length - 1, -1, -1):
                    if t == episode_length - 1:
                        returns[t] = rewards[t]
                    else:
                        returns[t] = rewards[t] + self.hyper_params['gamma'] * returns[t + 1]

                # Compute GAE advantages
                # GAE formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
                # where δ_t = r_t + γV(s_{t+1}) - V(s_t)
                gae = 0
                for t in range(episode_length - 1, -1, -1):
                    if t == episode_length - 1:
                        # Terminal state: V(s_{t+1}) = 0
                        delta = rewards[t] - state_values[t]
                    else:
                        # δ_t = r_t + γV(s_{t+1}) - V(s_t)
                        delta = rewards[t] + self.hyper_params['gamma'] * state_values[t + 1] - state_values[t]

                    # GAE accumulation: A_t = δ_t + γλ * A_{t+1}
                    gae = delta + self.hyper_params['gamma'] * self.hyper_params['gae_lambda'] * gae
                    advantages[t] = gae

                # Accumulate all data
                all_observations.extend(observations)
                all_actions.append(actions)
                all_logprobs.append(logprobs)
                all_rewards.append(rewards)
                all_state_values.append(state_values)
                all_returns.append(returns)
                all_advantages.append(advantages)

        # Concatenate all episodes
        combined_actions = torch.cat(all_actions, dim=0)
        combined_logprobs = torch.cat(all_logprobs, dim=0)
        combined_returns = torch.cat(all_returns, dim=0)
        combined_advantages = torch.cat(all_advantages, dim=0)

        print(f"  Mean advantage: {combined_advantages.mean():.4f}")
        print(f"  Mean return: {combined_returns.mean():.4f}")

        # Training metrics
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_total_losses = []

        # Set models to training mode
        self.agent.train()

        # Train for multiple epochs
        for epoch in range(self.hyper_params['epochs_per_update']):
            # Random permutation for mini-batches
            rand_idxs = np.random.permutation(total_steps)

            # Mini-batch updates
            for batch_start in range(0, total_steps, self.hyper_params['batch_size']):
                batch_end = min(batch_start + self.hyper_params['batch_size'], total_steps)
                batch_idxs = rand_idxs[batch_start:batch_end]

                # Extract batch
                batch_obs = [all_observations[i] for i in batch_idxs]
                batch_obs_tensor = batch_obs_to_tensor(batch_obs, device=self.device)
                batch_actions = combined_actions[batch_idxs].to(self.device)
                batch_old_logprobs = combined_logprobs[batch_idxs].to(self.device)
                batch_returns = combined_returns[batch_idxs].to(self.device)
                batch_advantages = combined_advantages[batch_idxs].to(self.device)

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
            'reward': float(mean_reward),
            'total_reward': float(total_reward),
            'steps': int(total_steps / len(episodes_data)),  # Mean episode length
            'total_steps': int(total_steps),
            'num_episodes': len(episodes_data),
            'actor_loss': float(np.mean(epoch_actor_losses)),
            'critic_loss': float(np.mean(epoch_critic_losses)),
            'total_loss': float(np.mean(epoch_total_losses))
        }

        # Add averaged reward components to metrics if available
        if 'reward_components' in episodes_data[0]:
            avg_comps = {key: 0.0 for key in episodes_data[0]['reward_components'].keys()}
            for ep_data in episodes_data:
                for key, value in ep_data['reward_components'].items():
                    avg_comps[key] += value
            for key in avg_comps:
                avg_comps[key] /= len(episodes_data)

            metrics['reward_components'] = {
                'progress_reward': float(avg_comps['progress_reward']),
                'effective_speed_reward': float(avg_comps['effective_speed_reward']),
                'gas_bonus': float(avg_comps['gas_bonus']),
                'reverse_tax': float(avg_comps['reverse_tax']),
                'parking_fine': float(avg_comps['parking_fine'])
            }

        self.training_history.append(metrics)

        print(f"\n  Training complete:")
        print(f"    Actor loss: {metrics['actor_loss']:.4f}")
        print(f"    Critic loss: {metrics['critic_loss']:.4f}")
        print(f"    Total loss: {metrics['total_loss']:.4f}")

        # Save checkpoint if good mean reward
        if mean_reward > 200:
            checkpoint_name = f"Y{mean_reward:.2f}RewardRacer{self.update_count}Update.pt"
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

        # Save training metadata to JSON after each update
        metadata_path = self.save_metadata_to_json()
        if self.update_count == 1 or self.update_count % 10 == 0:
            print(f"  ✓ Training metadata saved: {os.path.basename(metadata_path)}")

        print(f"{'='*60}\n")

        # Synchronize CUDA if using GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'agent_state_dict' in checkpoint:
            # Checkpoint with agent state dict (may or may not have optimizer states)
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            print(f"  [OK] Loaded agent state dict")

            # Try to load optimizer states if available
            if 'policy_optim_state_dict' in checkpoint:
                self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
                print(f"  [OK] Loaded policy optimizer state")
            else:
                print(f"  [INFO] Policy optimizer state not found, using fresh optimizer")

            if 'critic_optim_state_dict' in checkpoint:
                self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
                print(f"  [OK] Loaded critic optimizer state")
            else:
                print(f"  [INFO] Critic optimizer state not found, using fresh optimizer")

            # Load training metadata if available
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']
                print(f"  [OK] Resuming from update {self.update_count}")
            else:
                print(f"  [INFO] Update count not found, starting from 0")

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
                print(f"  [OK] Loaded training history ({len(self.training_history)} entries)")
            else:
                print(f"  [INFO] Training history not found, starting fresh")

        else:
            # Simple checkpoint with just model weights (state dict directly)
            self.agent.load_state_dict(checkpoint)
            print(f"  [OK] Loaded model weights (no metadata)")

        print(f"  [OK] Checkpoint loaded successfully!")

    def save_metadata_to_json(self, filepath: Optional[str] = None):
        """
        Save training metadata to a JSON file

        Args:
            filepath: Path to save JSON file. If None, saves to checkpoint_dir/training_metadata.json
        """
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "training_metadata.json")

        # Prepare metadata dictionary
        metadata = {
            'training_info': {
                'update_count': int(self.update_count),
                'total_steps': int(self.total_steps),
                'device': str(self.device),
                'max_episode_steps': int(self.max_episode_steps)
            },
            'hyperparameters': self.hyper_params,
            'training_history': self.training_history,
            'model_info': {
                'policy_parameters': sum(p.numel() for p in self.agent.policy.parameters()),
                'critic_parameters': sum(p.numel() for p in self.agent.critic.parameters())
            }
        }

        # Add trajectory info if available
        if self.trajectory_reward_fn is not None:
            metadata['trajectory_info'] = {
                'enabled': True,
                'scale': float(self.trajectory_reward_fn.scale),
                'max_dist': float(self.trajectory_reward_fn.max_dist),
                'trajectory_length': len(self.trajectory_reward_fn.trajectory),
                'current_failure_countdown': int(self.trajectory_reward_fn.failure_countdown),
                'current_min_steps': int(self.trajectory_reward_fn.min_steps)
            }
        else:
            metadata['trajectory_info'] = {
                'enabled': False
            }

        # Add curriculum learning info
        metadata['curriculum_learning'] = self.curriculum_config

        # Calculate summary statistics if training history exists
        if self.training_history:
            rewards = [h['reward'] for h in self.training_history]
            steps = [h['steps'] for h in self.training_history]
            actor_losses = [h['actor_loss'] for h in self.training_history]

            metadata['summary_statistics'] = {
                'mean_reward': float(np.mean(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'mean_episode_length': float(np.mean(steps)),
                'mean_actor_loss': float(np.mean(actor_losses)),
                'total_episodes': len(self.training_history)
            }

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        return filepath

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
        print(f"{'='*60}\n")

        # Training loop
        for update in range(num_updates):
            print(f"\n{'='*60}")
            print(f"Update {update + 1}/{num_updates}")
            print(f"{'='*60}")

            start_time = time.time()

            # Update curriculum learning parameters
            self.update_curriculum()

            # 1. Collect multiple episodes locally
            print(f"\nCollecting {self.hyper_params['episodes_per_update']} episodes...")
            episodes_data = []
            for ep in range(self.hyper_params['episodes_per_update']):
                episode_data = self.collect_episode()
                episodes_data.append(episode_data)
                print(f"  Episode {ep+1}/{self.hyper_params['episodes_per_update']}: {episode_data['episode_length']} steps, reward: {episode_data['episode_reward']:.2f}")

            # 2. Train on collected episodes (synchronous - no network delay!)
            self.train_on_episodes(episodes_data)

            elapsed = time.time() - start_time

            # Calculate aggregate statistics
            total_reward = sum(ep['episode_reward'] for ep in episodes_data)
            mean_reward = total_reward / len(episodes_data)
            total_episode_steps = sum(ep['episode_length'] for ep in episodes_data)
            mean_episode_length = total_episode_steps / len(episodes_data)

            print(f"\n{'='*60}")
            print(f"Update {self.update_count} complete")
            print(f"  Episodes collected: {len(episodes_data)}")
            print(f"  Mean episode reward: {mean_reward:.2f} (total: {total_reward:.2f})")
            print(f"  Mean episode length: {mean_episode_length:.0f} steps (total: {total_episode_steps})")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Update time: {elapsed:.1f}s")

            # Show aggregate reward component summary if available
            if 'reward_components' in episodes_data[0]:
                # Average components across all episodes
                avg_comps = {key: 0.0 for key in episodes_data[0]['reward_components'].keys()}
                for ep_data in episodes_data:
                    for key, value in ep_data['reward_components'].items():
                        avg_comps[key] += value
                for key in avg_comps:
                    avg_comps[key] /= len(episodes_data)

                print(f"\n  Average reward breakdown:")
                print(f"    Progress: {avg_comps['progress_reward']:.2f} | Speed: {avg_comps['effective_speed_reward']:.2f} | Gas: {avg_comps['gas_bonus']:.2f}")
                print(f"    Reverse: -{avg_comps['reverse_tax']:.2f} | Parking: -{avg_comps['parking_fine']:.2f}")

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
        print(f"✓ Final checkpoint saved: {final_checkpoint_path}")

        # Save final training metadata
        metadata_path = self.save_metadata_to_json()
        print(f"✓ Final training metadata saved: {metadata_path}\n")


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
    parser.add_argument('--episodes-per-update', type=int, default=4, help='Episodes to collect per update')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden layer dimension')
    parser.add_argument('--grad-clip-val', type=float, default=0.5, help='Gradient clipping value')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda for advantage estimation (0.0=no GAE, 1.0=Monte Carlo)')

    # Trajectory reward parameters
    parser.add_argument('--reward-scale', type=float, default=0.01, help='Trajectory reward scale')
    parser.add_argument('--max-dist-from-traj', type=float, default=60.0, help='Max distance from trajectory')
    parser.add_argument('--check-forward', type=int, default=1000, help='Check forward positions (prevents cutting track)')
    parser.add_argument('--check-backward', type=int, default=10, help='Check backward positions')

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
        episodes_per_update=args.episodes_per_update,
        hidden_dim=args.hidden_dim,
        grad_clip_val=args.grad_clip_val,
        gae_lambda=args.gae_lambda,
        reward_scale=args.reward_scale,
        max_dist_from_traj=args.max_dist_from_traj,
        check_forward=args.check_forward,
        check_backward=args.check_backward
        # Using function defaults for curriculum learning parameters
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

        # Save training metadata
        metadata_path = trainer.save_metadata_to_json()
        print(f"✓ Training metadata saved: {metadata_path}")
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Save training metadata even on error
        try:
            metadata_path = trainer.save_metadata_to_json()
            print(f"✓ Training metadata saved: {metadata_path}")
        except Exception as meta_error:
            print(f"✗ Failed to save metadata: {meta_error}")


if __name__ == "__main__":
    main()
