"""
Hyperparameter configurations for TrackMania PPO training

Use these to quickly switch between different model sizes:
- TINY: ~1.5M parameters (for quick testing)
- SMALL: ~3M parameters (balanced)
- MEDIUM: ~7M parameters
- LARGE: ~26M parameters (original)
"""

# TINY MODEL - For rapid testing (16x smaller than original)
# Total params: ~1.5M (Policy: ~1M, Critic: ~500K)
HYPER_PARAMS_TINY = {
    'policy_lr': 3e-4,  # Higher LR for tiny model
    'critic_lr': 3e-4,
    'gamma': 0.996,
    'clip_coef': 0.2,
    'critic_coef': 0.1,
    'entropy_coef': 0.1,
    'batch_size': 128,  # Smaller batch for tiny model
    'num_updates': 10000,
    'epochs_per_update': 50,  # Fewer epochs for faster testing
    'hidden_dim': 32,  # Reduced from 512
    'max_episode_steps': 2400,
    'norm_advantages': True,
    'grad_clip_val': 0.5,  # Higher clip for faster training
    'initial_std': 1,
    'avg_ray': 400
}

# SMALL MODEL - Good for testing with reasonable performance
# Total params: ~3M (Policy: ~2M, Critic: ~1M)
HYPER_PARAMS_SMALL = {
    'policy_lr': 1e-4,
    'critic_lr': 1e-4,
    'gamma': 0.996,
    'clip_coef': 0.2,
    'critic_coef': 0.1,
    'entropy_coef': 0.1,
    'batch_size': 128,
    'num_updates': 10000,
    'epochs_per_update': 80,
    'hidden_dim': 64,  # Reduced from 512
    'max_episode_steps': 2400,
    'norm_advantages': True,
    'grad_clip_val': 0.3,
    'initial_std': 1,
    'avg_ray': 400
}

# MEDIUM MODEL - Balanced performance/speed
# Total params: ~7M (Policy: ~4.5M, Critic: ~2.2M)
HYPER_PARAMS_MEDIUM = {
    'policy_lr': 5e-5,
    'critic_lr': 5e-5,
    'gamma': 0.996,
    'clip_coef': 0.2,
    'critic_coef': 0.1,
    'entropy_coef': 0.1,
    'batch_size': 256,
    'num_updates': 10000,
    'epochs_per_update': 100,
    'hidden_dim': 128,  # Reduced from 512
    'max_episode_steps': 2400,
    'norm_advantages': True,
    'grad_clip_val': 0.2,
    'initial_std': 1,
    'avg_ray': 400
}

# LARGE MODEL - Original configuration
# Total params: ~26M (Policy: ~17M, Critic: ~9M)
HYPER_PARAMS_LARGE = {
    'policy_lr': 1e-5,
    'critic_lr': 1e-5,
    'gamma': 0.996,
    'clip_coef': 0.2,
    'critic_coef': 0.1,
    'entropy_coef': 0.1,
    'batch_size': 256,
    'num_updates': 10000,
    'epochs_per_update': 100,
    'hidden_dim': 512,  # Original size
    'max_episode_steps': 2400,
    'norm_advantages': True,
    'grad_clip_val': 0.1,
    'initial_std': 1,
    'avg_ray': 400
}


def get_param_count(hidden_dim, observation_space=16393, action_space=3):
    """Calculate approximate parameter count for given hidden_dim"""
    # Policy network
    policy_action_mean = observation_space * hidden_dim + hidden_dim * hidden_dim + hidden_dim * action_space
    policy_logvar = observation_space * hidden_dim + hidden_dim * hidden_dim + hidden_dim * 1
    policy_total = policy_action_mean + policy_logvar

    # Critic network
    critic_total = observation_space * hidden_dim + hidden_dim * hidden_dim + hidden_dim * 1

    total = policy_total + critic_total

    return {
        'policy': policy_total,
        'critic': critic_total,
        'total': total
    }


def print_model_sizes():
    """Print model sizes for all configurations"""
    configs = {
        'TINY': HYPER_PARAMS_TINY,
        'SMALL': HYPER_PARAMS_SMALL,
        'MEDIUM': HYPER_PARAMS_MEDIUM,
        'LARGE': HYPER_PARAMS_LARGE
    }

    print("Model Size Comparison:")
    print("=" * 70)
    print(f"{'Config':<10} {'Hidden':<10} {'Policy':<15} {'Critic':<15} {'Total':<15}")
    print("-" * 70)

    for name, config in configs.items():
        params = get_param_count(config['hidden_dim'])
        print(f"{name:<10} {config['hidden_dim']:<10} "
              f"{params['policy']:>12,}   {params['critic']:>12,}   {params['total']:>12,}")

    print("=" * 70)


if __name__ == "__main__":
    print_model_sizes()
    print("\nRecommendation for testing:")
    print("  Use TINY for quick connection/infrastructure testing")
    print("  Use SMALL for actual training runs")
    print("  Use MEDIUM/LARGE when you want best performance")
