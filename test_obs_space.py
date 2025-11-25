"""
Quick script to determine actual observation space size from TMRL
"""
import torch
import tmrl

def env_obs_to_tensor(observations):
    """Convert TMRL observation tuple to flattened tensor"""
    tensors = [torch.tensor(observation, dtype=torch.float32).view(-1) for observation in observations]
    return torch.cat(tuple(tensors), dim=-1)

# Initialize environment
print("Initializing TMRL environment...")
env = tmrl.get_environment()

# Get initial observation
obs = env.reset()[0]

print(f"\nObservation structure:")
print(f"  Type: {type(obs)}")
print(f"  Length: {len(obs)}")
for i, component in enumerate(obs):
    if hasattr(component, 'shape'):
        print(f"  Component {i}: shape {component.shape}")
    else:
        print(f"  Component {i}: {component}")

# Convert to tensor and get size
obs_tensor = env_obs_to_tensor(obs)
print(f"\nFlattened observation size: {obs_tensor.shape[0]}")

env.close()
