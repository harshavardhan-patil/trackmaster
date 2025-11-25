"""
Local Trainer for Distributed TrackMania PPO Training

This script runs on your local machine and handles:
1. TMRL environment interactions (env.reset(), env.step(), env.wait())
2. Episode collection (observations, actions, rewards)
3. Communication with remote Colab server via HTTP

Usage:
    python local_trainer.py --remote-url https://your-ngrok-url.ngrok.io --num-updates 10000
"""

import argparse
import time
import pickle
import gzip
import numpy as np
import requests
from typing import List, Tuple, Dict, Any
import tmrl


class LocalTrainer:
    """Collects episodes locally and syncs with remote trainer"""

    def __init__(
        self,
        remote_url: str,
        max_episode_steps: int = 2400,
        retry_attempts: int = 3,
        timeout_seconds: int = 600
    ):
        """
        Initialize local trainer

        Args:
            remote_url: URL of remote training server (ngrok URL)
            max_episode_steps: Maximum steps per episode
            retry_attempts: Number of retry attempts for failed requests
            timeout_seconds: Maximum time to wait for training completion
        """
        self.remote_url = remote_url.rstrip('/')
        self.max_episode_steps = max_episode_steps
        self.retry_attempts = retry_attempts
        self.timeout_seconds = timeout_seconds

        # Initialize TMRL environment
        print("Initializing TMRL environment...")
        self.env = tmrl.get_environment()
        print(f"Observation Space: {self.env.observation_space}")
        print(f"Action Space: {self.env.action_space}")

        # Statistics
        self.update_count = 0
        self.total_steps = 0

    def check_server_health(self) -> bool:
        """Check if remote server is reachable"""
        try:
            response = requests.get(f"{self.remote_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Server connected - Status: {status['status']}")
                return True
            return False
        except Exception as e:
            print(f"✗ Server connection failed: {e}")
            return False

    def reset_remote_episode(self) -> bool:
        """Signal remote server that new episode is starting"""
        try:
            response = requests.post(f"{self.remote_url}/reset", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"  Warning: Failed to signal episode reset: {e}")
            return False

    def get_action(self, observation: Tuple) -> Tuple[np.ndarray, float, float]:
        """
        Request action from remote policy

        Args:
            observation: TMRL observation tuple (speed, gear, rpm, images, act1, act2)

        Returns:
            action: numpy array [3] with values in [-1, 1]
            logprob: log probability of the action
            state_value: critic's estimate of state value
        """
        try:
            # Serialize observation
            data = pickle.dumps(observation)

            # Request action from remote
            response = requests.post(
                f"{self.remote_url}/action",
                data=data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=5
            )

            if response.status_code == 200:
                result = pickle.loads(response.content)
                return result['action'], result['logprob'], result['state_value']
            else:
                print(f"  Warning: Action request failed with status {response.status_code}")
                # Return random action as fallback
                return np.random.uniform(-1, 1, 3), 0.0, 0.0

        except Exception as e:
            print(f"  Warning: Action request error: {e}")
            # Return random action as fallback
            return np.random.uniform(-1, 1, 3), 0.0, 0.0

    def collect_episode(self) -> Dict[str, Any]:
        """
        Collect one complete episode using current policy

        Returns:
            episode_data: Dictionary containing observations, actions, logprobs, rewards, state_values
        """
        print("\n  Collecting episode...")

        # Signal remote to prepare for new episode
        self.reset_remote_episode()

        # Buffers for episode data
        observations = []
        actions = []
        logprobs = []
        rewards = []
        state_values = []

        # Reset environment
        obs = self.env.reset()[0]

        step_count = 0
        done = False
        episode_reward = 0.0

        start_time = time.time()

        while not done and step_count < self.max_episode_steps:
            # Get action from remote policy
            action, logprob, state_value = self.get_action(obs)

            # Store trajectory data
            observations.append(obs)
            actions.append(action)
            logprobs.append(logprob)
            state_values.append(state_value)

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
            'episode_length': step_count,
            'episode_reward': episode_reward
        }

    def upload_episode(self, episode_data: Dict[str, Any]) -> bool:
        """
        Upload episode to remote trainer with retry logic

        Args:
            episode_data: Dictionary containing episode trajectory

        Returns:
            success: True if upload succeeded
        """
        for attempt in range(self.retry_attempts):
            try:
                print(f"  Uploading episode (attempt {attempt + 1}/{self.retry_attempts})...")

                # Serialize and compress
                data = pickle.dumps(episode_data)
                compressed = gzip.compress(data, compresslevel=6)

                print(f"    Data size: {len(data) / 1e6:.1f} MB → {len(compressed) / 1e6:.1f} MB compressed")

                # HTTP POST with timeout
                start_time = time.time()
                response = requests.post(
                    f"{self.remote_url}/episode",
                    data=compressed,
                    headers={'Content-Encoding': 'gzip'},
                    timeout=60
                )

                elapsed = time.time() - start_time

                if response.status_code == 200:
                    print(f"  ✓ Episode uploaded successfully in {elapsed:.1f}s")
                    return True
                elif response.status_code == 503:
                    print(f"    Server busy, retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"    Upload failed with status {response.status_code}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                print(f"    Upload error: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)

        print("  ✗ Upload failed after all retries")
        return False

    def wait_for_training(self, poll_interval: int = 5) -> bool:
        """
        Poll remote status until training is complete

        Args:
            poll_interval: Seconds between status checks

        Returns:
            success: True if training completed
        """
        print("  Waiting for training to complete...")
        start_time = time.time()

        while time.time() - start_time < self.timeout_seconds:
            try:
                response = requests.get(f"{self.remote_url}/status", timeout=10)

                if response.status_code == 200:
                    status = response.json()

                    if status['status'] == 'idle':
                        elapsed = time.time() - start_time
                        print(f"  ✓ Training completed in {elapsed:.1f}s")
                        return True

                    elapsed = time.time() - start_time
                    print(f"    Training in progress... ({elapsed:.0f}s elapsed)")

            except requests.exceptions.RequestException as e:
                print(f"    Status check error: {e}")

            time.sleep(poll_interval)

        print(f"  ⚠ Training timeout after {self.timeout_seconds}s")
        return False

    def train_loop(self, num_updates: int = 10000):
        """
        Main training loop

        Args:
            num_updates: Number of episodes to collect and train on
        """
        print(f"\n{'='*60}")
        print(f"Starting Distributed Training")
        print(f"{'='*60}")
        print(f"Remote URL: {self.remote_url}")
        print(f"Number of updates: {num_updates}")
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"{'='*60}\n")

        # Check server connectivity
        if not self.check_server_health():
            print("\n✗ ERROR: Cannot connect to remote server!")
            print("  Make sure the Colab notebook is running and ngrok URL is correct.")
            return

        # Training loop
        for update in range(num_updates):
            print(f"\n{'='*60}")
            print(f"Update {update + 1}/{num_updates}")
            print(f"{'='*60}")

            start_time = time.time()

            # 1. Collect episode locally
            episode_data = self.collect_episode()

            # 2. Upload episode to remote
            if not self.upload_episode(episode_data):
                print("  Skipping this update due to upload failure")
                continue

            # 3. Wait for training to complete
            self.wait_for_training()

            # Update statistics
            self.update_count += 1
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Local trainer for distributed TrackMania PPO")
    parser.add_argument(
        '--remote-url',
        type=str,
        required=True,
        help='Remote server URL (ngrok URL from Colab)'
    )
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

    args = parser.parse_args()

    # Create trainer
    trainer = LocalTrainer(
        remote_url=args.remote_url,
        max_episode_steps=args.max_steps
    )

    # Start training loop
    try:
        trainer.train_loop(num_updates=args.num_updates)
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
