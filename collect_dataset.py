"""
uv run collect_dataset.py --checkpoint ./checkpoints/base.pt --output pretrain.h5 --trajectory trajectory_data.pkl
"""

import argparse
import time
import os
import pickle
import numpy as np
import torch
import h5py
from typing import List, Tuple, Dict, Any, Optional
import tmrl
from src.agent import Agent
from trainer_ppo import RewardFunction


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


class DataCollector:
    """Collects expert demonstrations from a pre-trained model"""

    def __init__(
        self,
        checkpoint_path: str,
        max_episode_steps: int = 2000,
        device: str = None,
        trajectory_path: Optional[str] = None,
        reward_scale: float = 0.1,
        max_dist_from_traj: float = 30.0,
        check_forward: int = 1000,
        check_backward: int = 10,
        failure_countdown: int = 1000,
        min_steps: int = 500
    ):
        """
        Initialize data collector

        Args:
            checkpoint_path: Path to pre-trained model checkpoint
            max_episode_steps: Maximum steps per episode
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            trajectory_path: Path to trajectory file for custom reward function (optional)
            reward_scale: Scale factor for trajectory progress rewards
            max_dist_from_traj: Max distance from trajectory before reward = 0
            check_forward: Allow cuts up to N positions ahead
            check_backward: Allow rewinding up to N positions back
            failure_countdown: Terminate after N steps with no progress
            min_steps: Minimum steps before termination
        """
        self.max_episode_steps = max_episode_steps

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Initialize environment
        print("\nInitializing TMRL environment...")
        self.env = tmrl.get_environment()
        print(f"Observation Space: {self.env.observation_space}")
        print(f"Action Space: {self.env.action_space}")

        # Initialize agent
        print("\nInitializing agent...")
        self.agent = Agent(action_space=3).to(self.device)

        # Load checkpoint
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'agent_state_dict' in checkpoint:
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            print("  ✓ Loaded agent from checkpoint (with metadata)")
        else:
            self.agent.load_state_dict(checkpoint)
            print("  ✓ Loaded agent from checkpoint (weights only)")

        # Set to evaluation mode
        self.agent.eval()

        # Load trajectory if provided for custom reward function
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
                    failure_countdown=failure_countdown,
                    min_steps=min_steps,
                    debug=False
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

        print(f"\n✓ Data collector initialized successfully!")

    def get_action(self, observation: Tuple) -> np.ndarray:
        """
        Get deterministic action from policy (mean only, no sampling)

        Args:
            observation: TMRL observation tuple

        Returns:
            action: numpy array [3] with values in [-1, 1]
        """
        # Convert observation to tensor tuple
        obs_tensor = env_obs_to_tensor(observation, device=self.device)

        # Get deterministic action (mean only, no sampling)
        with torch.no_grad():
            action = self.agent.policy.mean_only(obs_tensor)

        return action[0].cpu().numpy()

    def collect_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Collect one complete episode

        Args:
            episode_num: Episode number (for logging)

        Returns:
            episode_data: Dictionary containing observations and actions
        """
        print(f"\n  Episode {episode_num}:")

        # Buffers for episode data
        observations = []
        actions = []

        # Reset environment and custom reward function
        obs = self.env.reset()[0]
        if self.trajectory_reward_fn is not None:
            self.trajectory_reward_fn.reset()

        step_count = 0
        episode_reward = 0.0
        done = False

        start_time = time.time()

        while not done and step_count < self.max_episode_steps:
            # Get deterministic action from policy
            action = self.get_action(obs)

            # Store data
            # Observation format: (speed, gear, rpm, images, act1, act2)
            # We need to flatten this for HDF5 storage
            speed = float(obs[0].item()) if hasattr(obs[0], 'item') else float(obs[0])
            gear = float(obs[1].item()) if hasattr(obs[1], 'item') else float(obs[1])
            rpm = float(obs[2].item()) if hasattr(obs[2], 'item') else float(obs[2])
            images = np.array(obs[3], dtype=np.float32)  # [4, 64, 64]
            act1 = np.array(obs[4], dtype=np.float32)    # [3]
            act2 = np.array(obs[5], dtype=np.float32)    # [3]

            # Store as dictionary for easier HDF5 conversion
            obs_dict = {
                'speed': speed,
                'gear': gear,
                'rpm': rpm,
                'images': images,
                'act1': act1,
                'act2': act2
            }

            observations.append(obs_dict)
            actions.append(action.copy())

            # Clip action to valid range
            clamped_action = np.clip(action, -1, 1)

            # Step environment
            next_obs, tmrl_reward, terminated, truncated, info = self.env.step(clamped_action)

            # Compute reward using custom reward function if available
            if self.trajectory_reward_fn is not None:
                try:
                    # Get position data from environment
                    data = self.env.unwrapped.interface.client.retrieve_data(sleep_if_empty=0.01, timeout=0.1)
                    position = np.array([data[2], data[3], data[4]], dtype=np.float32)  # x, y, z

                    # Compute custom reward
                    custom_reward, custom_terminated, reward_components = self.trajectory_reward_fn.compute_reward(
                        position, speed, action, gear
                    )
                    reward = float(custom_reward)  # Use ONLY custom reward, ignore TMRL reward
                    terminated = custom_terminated or terminated
                except Exception as e:
                    # If position unavailable, fall back to TMRL reward (only warn on first occurrence)
                    if step_count == 0:
                        print(f"    Warning: Could not retrieve position data: {e}")
                        print(f"    Falling back to TMRL reward for this episode")
                    reward = float(tmrl_reward)
            else:
                reward = float(tmrl_reward)  # Use TMRL reward if no custom function

            episode_reward += reward
            obs = next_obs
            done = terminated or truncated
            step_count += 1

            # Print progress every 500 steps
            if step_count % 500 == 0:
                elapsed = time.time() - start_time
                print(f"    Step {step_count}/{self.max_episode_steps} - Reward: {episode_reward:.2f} - Time: {elapsed:.1f}s")

        # Pause environment (TMRL requirement)
        self.env.unwrapped.wait()

        elapsed = time.time() - start_time

        print(f"    ✓ Completed: {step_count} steps, reward: {episode_reward:.2f}, time: {elapsed:.1f}s")

        return {
            'observations': observations,
            'actions': actions,
            'episode_length': step_count,
            'episode_reward': episode_reward
        }

    def collect_dataset(self, num_episodes: int = 100, output_path: str = None, save_every: int = 10) -> Dict[str, Any]:
        """
        Collect dataset from multiple episodes with incremental saving

        Args:
            num_episodes: Number of episodes to collect
            output_path: Path to HDF5 file (if provided, saves incrementally)
            save_every: Save to disk every N episodes to free memory (default: 10)

        Returns:
            dataset: Dictionary containing summary statistics
        """
        print(f"\n{'='*60}")
        print(f"Starting Data Collection")
        print(f"{'='*60}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Max episode steps: {self.max_episode_steps}")
        if output_path:
            print(f"Saving incrementally every {save_every} episodes to: {output_path}")
        print(f"{'='*60}\n")

        batch_episodes = []
        total_steps = 0
        total_reward = 0.0
        episodes_collected = 0

        start_time = time.time()

        for ep in range(num_episodes):
            episode_data = self.collect_episode(ep + 1)
            batch_episodes.append(episode_data)
            total_steps += episode_data['episode_length']
            total_reward += episode_data['episode_reward']
            episodes_collected += 1

            # Save incrementally every 'save_every' episodes
            if output_path and (episodes_collected % save_every == 0 or episodes_collected == num_episodes):
                print(f"\n  💾 Saving batch ({len(batch_episodes)} episodes) to disk...")
                self.append_to_hdf5(batch_episodes, output_path)
                print(f"  ✓ Batch saved, freeing memory...")
                batch_episodes = []  # Clear memory

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Data Collection Complete!")
        print(f"{'='*60}")
        print(f"Total episodes: {num_episodes}")
        print(f"Total steps: {total_steps}")
        print(f"Mean episode length: {total_steps / num_episodes:.1f}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Mean episode reward: {total_reward / num_episodes:.2f}")
        print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
        print(f"{'='*60}\n")

        return {
            'total_steps': total_steps,
            'num_episodes': num_episodes,
            'mean_episode_length': total_steps / num_episodes,
            'mean_episode_reward': total_reward / num_episodes
        }

    def append_to_hdf5(self, episodes: List[Dict[str, Any]], output_path: str):
        """
        Append episode data to HDF5 file (creates file if doesn't exist)

        Args:
            episodes: List of episode dictionaries
            output_path: Path to HDF5 file
        """
        # Extract data from episodes
        all_speeds = []
        all_gears = []
        all_rpms = []
        all_images = []
        all_act1s = []
        all_act2s = []
        all_actions = []
        episode_lengths = []
        episode_rewards = []

        for episode in episodes:
            for obs_dict, action in zip(episode['observations'], episode['actions']):
                all_speeds.append(obs_dict['speed'])
                all_gears.append(obs_dict['gear'])
                all_rpms.append(obs_dict['rpm'])
                all_images.append(obs_dict['images'])
                all_act1s.append(obs_dict['act1'])
                all_act2s.append(obs_dict['act2'])
                all_actions.append(action)

            episode_lengths.append(episode['episode_length'])
            episode_rewards.append(episode['episode_reward'])

        # Convert to numpy arrays
        speeds_array = np.array(all_speeds, dtype=np.float32)
        gears_array = np.array(all_gears, dtype=np.float32)
        rpms_array = np.array(all_rpms, dtype=np.float32)
        images_array = np.array(all_images, dtype=np.float32)
        act1s_array = np.array(all_act1s, dtype=np.float32)
        act2s_array = np.array(all_act2s, dtype=np.float32)
        actions_array = np.array(all_actions, dtype=np.float32)

        batch_steps = len(all_speeds)

        # Check if file exists
        file_exists = os.path.exists(output_path)

        if not file_exists:
            # Create new file
            with h5py.File(output_path, 'w') as f:
                # Create observations group
                obs_group = f.create_group('observations')
                obs_group.create_dataset('speed', data=speeds_array, maxshape=(None,), compression='gzip')
                obs_group.create_dataset('gear', data=gears_array, maxshape=(None,), compression='gzip')
                obs_group.create_dataset('rpm', data=rpms_array, maxshape=(None,), compression='gzip')
                obs_group.create_dataset('images', data=images_array, maxshape=(None, 4, 64, 64), compression='gzip')
                obs_group.create_dataset('act1', data=act1s_array, maxshape=(None, 3), compression='gzip')
                obs_group.create_dataset('act2', data=act2s_array, maxshape=(None, 3), compression='gzip')

                # Create actions dataset
                f.create_dataset('actions', data=actions_array, maxshape=(None, 3), compression='gzip')

                # Store metadata
                metadata = f.create_group('metadata')
                metadata.attrs['num_episodes'] = len(episodes)
                metadata.attrs['total_steps'] = batch_steps

                # Store episode boundaries
                metadata.create_dataset('episode_lengths', data=np.array(episode_lengths, dtype=np.int32), maxshape=(None,), compression='gzip')
                metadata.create_dataset('episode_rewards', data=np.array(episode_rewards, dtype=np.float32), maxshape=(None,), compression='gzip')

            print(f"    Created new file: {batch_steps} transitions, {len(episodes)} episodes")
        else:
            # Append to existing file
            with h5py.File(output_path, 'a') as f:
                # Resize and append to observation datasets
                obs_group = f['observations']

                for name, data in [
                    ('speed', speeds_array),
                    ('gear', gears_array),
                    ('rpm', rpms_array),
                    ('images', images_array),
                    ('act1', act1s_array),
                    ('act2', act2s_array)
                ]:
                    dataset = obs_group[name]
                    old_size = dataset.shape[0]
                    new_size = old_size + batch_steps
                    dataset.resize(new_size, axis=0)
                    dataset[old_size:new_size] = data

                # Resize and append actions
                actions_dataset = f['actions']
                old_size = actions_dataset.shape[0]
                new_size = old_size + batch_steps
                actions_dataset.resize(new_size, axis=0)
                actions_dataset[old_size:new_size] = actions_array

                # Update metadata
                metadata = f['metadata']
                old_num_episodes = metadata.attrs['num_episodes']
                old_total_steps = metadata.attrs['total_steps']

                metadata.attrs['num_episodes'] = old_num_episodes + len(episodes)
                metadata.attrs['total_steps'] = old_total_steps + batch_steps

                # Append episode metadata
                ep_lengths_dataset = metadata['episode_lengths']
                ep_rewards_dataset = metadata['episode_rewards']

                old_ep_count = ep_lengths_dataset.shape[0]
                new_ep_count = old_ep_count + len(episodes)

                ep_lengths_dataset.resize(new_ep_count, axis=0)
                ep_rewards_dataset.resize(new_ep_count, axis=0)

                ep_lengths_dataset[old_ep_count:new_ep_count] = np.array(episode_lengths, dtype=np.int32)
                ep_rewards_dataset[old_ep_count:new_ep_count] = np.array(episode_rewards, dtype=np.float32)

            print(f"    Appended: {batch_steps} transitions, {len(episodes)} episodes")

        # Print file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    File size: {file_size_mb:.2f} MB")

    def save_to_hdf5(self, dataset: Dict[str, Any], output_path: str):
        """
        Save dataset to HDF5 file

        Args:
            dataset: Dataset dictionary from collect_dataset()
            output_path: Path to save HDF5 file
        """
        print(f"Saving dataset to {output_path}...")

        episodes = dataset['episodes']

        # First pass: calculate total size and collect all data
        total_steps = 0
        all_speeds = []
        all_gears = []
        all_rpms = []
        all_images = []
        all_act1s = []
        all_act2s = []
        all_actions = []

        for episode in episodes:
            for obs_dict, action in zip(episode['observations'], episode['actions']):
                all_speeds.append(obs_dict['speed'])
                all_gears.append(obs_dict['gear'])
                all_rpms.append(obs_dict['rpm'])
                all_images.append(obs_dict['images'])
                all_act1s.append(obs_dict['act1'])
                all_act2s.append(obs_dict['act2'])
                all_actions.append(action)
                total_steps += 1

        # Convert lists to numpy arrays
        speeds_array = np.array(all_speeds, dtype=np.float32)
        gears_array = np.array(all_gears, dtype=np.float32)
        rpms_array = np.array(all_rpms, dtype=np.float32)
        images_array = np.array(all_images, dtype=np.float32)  # [total_steps, 4, 64, 64]
        act1s_array = np.array(all_act1s, dtype=np.float32)    # [total_steps, 3]
        act2s_array = np.array(all_act2s, dtype=np.float32)    # [total_steps, 3]
        actions_array = np.array(all_actions, dtype=np.float32) # [total_steps, 3]

        print(f"  Total transitions: {total_steps}")
        print(f"  Observations shape:")
        print(f"    - speeds: {speeds_array.shape}")
        print(f"    - gears: {gears_array.shape}")
        print(f"    - rpms: {rpms_array.shape}")
        print(f"    - images: {images_array.shape}")
        print(f"    - act1: {act1s_array.shape}")
        print(f"    - act2: {act2s_array.shape}")
        print(f"  Actions shape: {actions_array.shape}")

        # Save to HDF5
        with h5py.File(output_path, 'w') as f:
            # Create observations group
            obs_group = f.create_group('observations')
            obs_group.create_dataset('speed', data=speeds_array, compression='gzip')
            obs_group.create_dataset('gear', data=gears_array, compression='gzip')
            obs_group.create_dataset('rpm', data=rpms_array, compression='gzip')
            obs_group.create_dataset('images', data=images_array, compression='gzip')
            obs_group.create_dataset('act1', data=act1s_array, compression='gzip')
            obs_group.create_dataset('act2', data=act2s_array, compression='gzip')

            # Create actions dataset
            f.create_dataset('actions', data=actions_array, compression='gzip')

            # Store metadata
            metadata = f.create_group('metadata')
            metadata.attrs['num_episodes'] = dataset['num_episodes']
            metadata.attrs['total_steps'] = dataset['total_steps']
            metadata.attrs['mean_episode_length'] = dataset['mean_episode_length']
            metadata.attrs['mean_episode_reward'] = dataset['mean_episode_reward']

            # Store episode boundaries (useful for batching by episode)
            episode_lengths = [ep['episode_length'] for ep in episodes]
            episode_rewards = [ep['episode_reward'] for ep in episodes]
            metadata.create_dataset('episode_lengths', data=np.array(episode_lengths, dtype=np.int32))
            metadata.create_dataset('episode_rewards', data=np.array(episode_rewards, dtype=np.float32))

        # Verify file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✓ Dataset saved successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Collect dataset from pre-trained TrackMania model")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to pre-trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='trackmania_dataset.h5',
        help='Output HDF5 file path (default: trackmania_dataset.h5)'
    )
    parser.add_argument(
        '--trajectory',
        type=str,
        default=None,
        help='Path to trajectory file (.pkl) for custom reward function (optional)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=20,
        help='Number of episodes to collect (default: 120)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=2000,
        help='Maximum steps per episode (default: 2000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    # Reward function parameters
    parser.add_argument(
        '--reward-scale',
        type=float,
        default=0.1,
        help='Scale factor for trajectory progress rewards (default: 0.1)'
    )
    parser.add_argument(
        '--max-dist-from-traj',
        type=float,
        default=30.0,
        help='Max distance from trajectory before reward = 0 (default: 30.0)'
    )
    parser.add_argument(
        '--check-forward',
        type=int,
        default=1000,
        help='Check forward positions (prevents cutting track) (default: 1000)'
    )
    parser.add_argument(
        '--check-backward',
        type=int,
        default=10,
        help='Check backward positions (default: 10)'
    )
    parser.add_argument(
        '--failure-countdown',
        type=int,
        default=1000,
        help='Terminate after N steps with no progress (default: 1000)'
    )
    parser.add_argument(
        '--min-steps',
        type=int,
        default=500,
        help='Minimum steps before termination (default: 500)'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save to disk every N episodes to free memory (default: 10)'
    )

    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

    # Create data collector
    collector = DataCollector(
        checkpoint_path=args.checkpoint,
        max_episode_steps=args.max_steps,
        device=args.device,
        trajectory_path=args.trajectory,
        reward_scale=args.reward_scale,
        max_dist_from_traj=args.max_dist_from_traj,
        check_forward=args.check_forward,
        check_backward=args.check_backward,
        failure_countdown=args.failure_countdown,
        min_steps=args.min_steps
    )

    try:
        # Collect dataset with incremental saving
        dataset = collector.collect_dataset(
            num_episodes=args.num_episodes,
            output_path=args.output,
            save_every=args.save_every
        )

        print(f"\n{'='*60}")
        print("Data collection completed successfully!")
        print(f"{'='*60}")
        print(f"Dataset saved to: {args.output}")
        print(f"Total episodes: {dataset['num_episodes']}")
        print(f"Total transitions: {dataset['total_steps']}")
        print(f"\nTo load this dataset in Python:")
        print(f"  import h5py")
        print(f"  with h5py.File('{args.output}', 'r') as f:")
        print(f"      observations = f['observations']")
        print(f"      actions = f['actions']")
        print(f"      metadata = f['metadata']")
        print(f"      # Check total episodes: f['metadata'].attrs['num_episodes']")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\n✗ Data collection interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Data collection failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
