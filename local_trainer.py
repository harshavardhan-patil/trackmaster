import argparse
import time
import pickle
import gzip
import numpy as np
import requests
from typing import List, Tuple, Dict, Any
import tmrl

# _original_save_ghost = tmrl.custom.tm.utils.tools.save_ghost

# def _safe_save_ghost(*args, **kwargs):
#     try:
#         return _original_save_ghost(*args, **kwargs)
#     except ConnectionRefusedError:
#         # WinError 10061: Game refused connection. Not critical, just skip saving.
#         print("  ⚠ Warning: Could not save ghost (Game refused connection). Continuing...")
#     except Exception as e:
#         print(f"  ⚠ Warning: Ghost save failed: {e}")

# # Apply the patch
# tmrl.custom.tm.utils.tools.save_ghost = _safe_save_ghost

class LocalTrainer:
    """Collects episodes locally and syncs with remote trainer"""

    def __init__(
        self,
        remote_url: str,
        max_episode_steps: int = 2400,
        retry_attempts: int = 3,
        timeout_seconds: int = 600,
        use_last_action_on_timeout: bool = True,
        action_timeout_threshold: float = 0.15
    ):
        """
        Initialize local trainer

        Args:
            remote_url: URL of remote training server (ngrok URL)
            max_episode_steps: Maximum steps per episode
            retry_attempts: Number of retry attempts for failed requests
            timeout_seconds: Maximum time to wait for training completion
            use_last_action_on_timeout: If True, repeat last action when nearing timeout
            action_timeout_threshold: Time threshold (seconds) to trigger last action repetition
        """
        self.remote_url = remote_url.rstrip('/')
        self.max_episode_steps = max_episode_steps
        self.retry_attempts = retry_attempts
        self.timeout_seconds = timeout_seconds
        self.use_last_action_on_timeout = use_last_action_on_timeout
        self.action_timeout_threshold = action_timeout_threshold

        self.session = requests.Session()

        # Initialize TMRL environment
        print("Initializing TMRL environment...")
        self.env = tmrl.get_environment()
        print(f"Observation Space: {self.env.observation_space}")
        print(f"Action Space: {self.env.action_space}")

        # Statistics
        self.update_count = 0
        self.total_steps = 0

        # Timeout tracking
        self.last_action = None
        self.last_logprob = 0.0
        self.last_state_value = 0.0
        self.actual_action_count = 0
        self.timeout_action_count = 0
        self.episode_actual_actions = 0
        self.episode_timeout_actions = 0

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
        start_time = time.time()
        used_last_action = False

        try:
            # Serialize observation
            data = pickle.dumps(observation)

            # Request action from remote with timeout threshold
            response = self.session.post(
                f"{self.remote_url}/action",
                data=data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=self.action_timeout_threshold
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = pickle.loads(response.content)
                action = result['action']
                logprob = result['logprob']
                state_value = result['state_value']

                # Store as last action
                self.last_action = action
                self.last_logprob = logprob
                self.last_state_value = state_value

                # Track as actual action
                self.actual_action_count += 1
                self.episode_actual_actions += 1

                return action, logprob, state_value
            else:
                print(f"  Warning: Action request failed with status {response.status_code}")
                used_last_action = True

        except requests.exceptions.Timeout:
            # Timeout occurred - use last action if enabled
            elapsed = time.time() - start_time
            if self.use_last_action_on_timeout and self.last_action is not None:
                used_last_action = True
            else:
                print(f"  Warning: Action request timed out after {elapsed:.3f}s (no last action available)")
                # Return random action as fallback
                action = np.random.uniform(-1, 1, 3)
                self.last_action = action
                self.last_logprob = 0.0
                self.last_state_value = 0.0
                return action, 0.0, 0.0

        except Exception as e:
            print(f"  Warning: Action request error: {e}")
            used_last_action = True

        # If we need to use last action
        if used_last_action:
            if self.last_action is not None:
                self.timeout_action_count += 1
                self.episode_timeout_actions += 1
                return self.last_action, self.last_logprob, self.last_state_value
            else:
                # No last action available, return random
                action = np.random.uniform(-1, 1, 3)
                self.last_action = action
                self.last_logprob = 0.0
                self.last_state_value = 0.0
                self.actual_action_count += 1
                self.episode_actual_actions += 1
                return action, 0.0, 0.0

    def collect_episode(self) -> Dict[str, Any]:
        """
        Collect one complete episode using current policy

        Returns:
            episode_data: Dictionary containing observations, actions, logprobs, rewards, state_values, positions
        """
        print("\n  Collecting episode...")

        # Signal remote to prepare for new episode
        self.reset_remote_episode()

        # Reset episode timeout counters
        self.episode_actual_actions = 0
        self.episode_timeout_actions = 0

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
            # Get action from remote policy
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

        # Calculate timeout percentage
        total_episode_steps = self.episode_actual_actions + self.episode_timeout_actions
        timeout_percent = (self.episode_timeout_actions / total_episode_steps * 100) if total_episode_steps > 0 else 0

        print(f"  ✓ Episode collected: {step_count} steps, reward: {episode_reward:.2f}, time: {elapsed:.1f}s")
        print(f"    Action stats: {self.episode_actual_actions} actual, {self.episode_timeout_actions} repeated ({timeout_percent:.1f}% timeout)")
        if self.use_last_action_on_timeout:
            print(f"    Last-action repetition: ENABLED (threshold: {self.action_timeout_threshold}s)")
        else:
            print(f"    Last-action repetition: DISABLED")

        return {
            'observations': observations,
            'actions': actions,
            'logprobs': logprobs,
            'rewards': rewards,
            'state_values': state_values,
            'positions': positions,  # Include positions for trajectory rewards
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
        print(f"Last-action repetition: {'ENABLED' if self.use_last_action_on_timeout else 'DISABLED'}")
        if self.use_last_action_on_timeout:
            print(f"Action timeout threshold: {self.action_timeout_threshold}s")
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

            # Calculate cumulative timeout statistics
            total_actions = self.actual_action_count + self.timeout_action_count
            cumulative_timeout_percent = (self.timeout_action_count / total_actions * 100) if total_actions > 0 else 0

            print(f"\n{'='*60}")
            print(f"Update {self.update_count} complete")
            print(f"  Episode reward: {episode_data['episode_reward']:.2f}")
            print(f"  Episode length: {episode_data['episode_length']} steps")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Cumulative action stats: {self.actual_action_count} actual, {self.timeout_action_count} repeated ({cumulative_timeout_percent:.1f}%)")
            print(f"  Update time: {elapsed:.1f}s")
            print(f"{'='*60}")

        # Calculate final timeout statistics
        total_actions = self.actual_action_count + self.timeout_action_count
        final_timeout_percent = (self.timeout_action_count / total_actions * 100) if total_actions > 0 else 0

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total updates: {self.update_count}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Total actions: {self.actual_action_count} actual, {self.timeout_action_count} repeated")
        print(f"  Overall timeout rate: {final_timeout_percent:.1f}%")
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
    parser.add_argument(
        '--use-last-action-on-timeout',
        type=bool,
        default=True,
        help='If True, repeat last action when nearing timeout (default: True)'
    )
    parser.add_argument(
        '--action-timeout-threshold',
        type=float,
        default=0.15,
        help='Time threshold (seconds) to trigger last action repetition (default: 0.15)'
    )

    args = parser.parse_args()

    # Create trainer
    trainer = LocalTrainer(
        remote_url=args.remote_url,
        max_episode_steps=args.max_steps,
        use_last_action_on_timeout=args.use_last_action_on_timeout,
        action_timeout_threshold=args.action_timeout_threshold
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
