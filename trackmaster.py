"""
TrackMaster - Load and run a trained model in TMRL environment
"""
import torch
import numpy as np
import argparse
import time
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
    images_t = torch.tensor(images, dtype=torch.float32, device=device).unsqueeze(0)

    # Previous actions
    act1_t = torch.tensor(act1, dtype=torch.float32, device=device).unsqueeze(0)
    act2_t = torch.tensor(act2, dtype=torch.float32, device=device).unsqueeze(0)

    return (speed_t, gear_t, rpm_t, images_t, act1_t, act2_t)


def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained model from a checkpoint file

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        Loaded Agent model
    """
    print(f"Loading model from: {checkpoint_path}")

    # Initialize agent
    agent = Agent(action_space=3)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if 'agent_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print("✓ Loaded agent_state_dict from checkpoint")

        # Print additional info if available
        if 'update_count' in checkpoint:
            print(f"  Update count: {checkpoint['update_count']}")
        if 'episode' in checkpoint:
            print(f"  Episode: {checkpoint['episode']}")
    elif 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded model_state_dict from checkpoint")
    else:
        # Try loading directly
        agent.load_state_dict(checkpoint)
        print("✓ Loaded model from checkpoint")

    agent.to(device)
    agent.eval()

    return agent


def get_action(agent, observation, device='cpu', deterministic=True):
    """
    Get action from the model

    Args:
        agent: Loaded Agent model
        observation: TMRL observation tuple
        device: Device to run inference on
        deterministic: If True, use mean action (no sampling)

    Returns:
        action: numpy array [3] with values for [gas, brake, steering]
    """
    # Convert observation to tensors
    obs_tensor = env_obs_to_tensor(observation, device=device)

    with torch.no_grad():
        if deterministic:
            # Use mean action (no sampling)
            action = agent.policy.mean_only(obs_tensor)
        else:
            # Sample action
            action, _ = agent.policy.sample_action_with_logprobs(obs_tensor)

    # Clip to valid range and return
    action = action[0].cpu().numpy()
    return np.clip(action, -1, 1)


def run_episode(agent, env, device='cpu', max_steps=2000, deterministic=True, verbose=True):
    """
    Run one episode in the TMRL environment

    Args:
        agent: Loaded Agent model
        env: TMRL environment
        device: Device to run inference on
        max_steps: Maximum steps per episode
        deterministic: If True, use mean action
        verbose: If True, print step information

    Returns:
        episode_reward: Total reward for the episode
        episode_length: Number of steps in the episode
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Starting episode...")
        print("=" * 60)

    # Reset environment
    obs = env.reset()[0]

    step_count = 0
    episode_reward = 0.0
    done = False

    start_time = time.time()

    while not done and step_count < max_steps:
        # Get action from policy
        action = get_action(agent, obs, device=device, deterministic=deterministic)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        obs = next_obs
        done = terminated or truncated
        step_count += 1

        # Print progress
        if verbose and step_count % 100 == 0:
            speed = float(obs[0].item()) if hasattr(obs[0], 'item') else float(obs[0])
            print(f"  Step {step_count}/{max_steps} - Reward: {episode_reward:.2f} - Speed: {speed:.1f} km/h")

    # Pause environment
    env.unwrapped.wait()

    elapsed = time.time() - start_time

    if verbose:
        print("=" * 60)
        print(f"Episode complete!")
        print(f"  Steps: {step_count}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Time: {elapsed:.1f}s")
        if terminated:
            print(f"  Status: TERMINATED")
        elif truncated or step_count >= max_steps:
            print(f"  Status: TRUNCATED (time limit)")
        print("=" * 60)

    return episode_reward, step_count


def main():
    parser = argparse.ArgumentParser(description='TrackMaster - Load and run a trained model')
    parser.add_argument('checkpoint', type=str, help='Path to the .pt checkpoint file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run inference on (default: cpu)')
    parser.add_argument('--max-steps', type=int, default=2000,
                        help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to run (default: 1)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (sample actions instead of mean)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print("=" * 60)
    print("""

     ___       ___/  \___
      |    __/_|_____|_\____
      |   /                 \
      | /     TRACKMASTER     \___________________
      .---.                                    .=====.
     | (@) | ---------------------------------|| (@) ||
      '---'                                    '====='     
""")
    print("=" * 60)

    # Load model
    agent = load_model(args.checkpoint, device=device)
    print(f"  Model loaded successfully on {device}")

    # Initialize TMRL environment
    print("\nInitializing TMRL environment...")
    env = tmrl.get_environment()
    print(f"  Environment initialized")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    try:
        reward, length = run_episode(
            agent,
            env,
            device=device,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            verbose=not args.quiet
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")



if __name__ == "__main__":
    main()
