"""
Simple CLI wrapper for starting local training

Usage:
    python start_local.py

The script will prompt you for the ngrok URL and number of updates.
"""

import sys
import subprocess

def main():
    print("="*60)
    print("TrackMania Distributed PPO Training - Local Trainer")
    print("="*60)
    print()

    # Get remote URL
    print("Enter the ngrok URL from your Colab notebook:")
    print("(Example: https://abc123.ngrok.io)")
    remote_url = input("Remote URL: ").strip()

    if not remote_url:
        print("Error: Remote URL is required")
        sys.exit(1)

    # Get number of updates
    print("\nHow many training updates do you want to run?")
    num_updates_str = input("Number of updates [default: 10000]: ").strip()
    num_updates = int(num_updates_str) if num_updates_str else 10000

    # Get max steps per episode
    print("\nMaximum steps per episode?")
    max_steps_str = input("Max steps [default: 2400]: ").strip()
    max_steps = int(max_steps_str) if max_steps_str else 2400

    print()
    print("="*60)
    print("Configuration:")
    print(f"  Remote URL: {remote_url}")
    print(f"  Number of updates: {num_updates}")
    print(f"  Max steps per episode: {max_steps}")
    print("="*60)
    print()

    # Confirm
    confirm = input("Start training? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Training cancelled")
        sys.exit(0)

    # Run local trainer
    print("\nStarting local trainer...\n")

    cmd = [
        sys.executable,
        "local_trainer.py",
        "--remote-url", remote_url,
        "--num-updates", str(num_updates),
        "--max-steps", str(max_steps)
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n\nTraining failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
