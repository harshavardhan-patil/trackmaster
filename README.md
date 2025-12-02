## Observation and Action Spaces

### TrackMania Observation Space

The full TrackMania environment provides:

```python
observation_space = spaces.Tuple((
    spaces.Box(low=0.0, high=1000.0, shape=(1,)),      # speed (km/h)
    spaces.Box(low=0.0, high=6.0, shape=(1,)),         # gear (0-6)
    spaces.Box(low=0.0, high=np.inf, shape=(1,)),      # rpm
    spaces.Box(low=0, high=255, shape=(H, W, 4)),      # image history (4 grayscale images)
    spaces.Box(low=-1.0, high=1.0, shape=(3,)),        # action buffer element 1
    spaces.Box(low=-1.0, high=1.0, shape=(3,)),        # action buffer element 2
    spaces.Box(low=-1.0, high=1.0, shape=(3,)),        # action buffer element 3
    spaces.Box(low=-1.0, high=1.0, shape=(3,)),        # action buffer element 4
))
```

**Components:**
1. **Telemetry**: speed, gear, rpm (3 floats)
2. **Visual**: 4 consecutive grayscale screenshots (64x64 default)
3. **Action History**: last 4 actions (real-time requirement)

### Action Space

```python
action_space = spaces.Box(
    low=np.array([-1.0, -1.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32
)
```

**Action Components:**
- `action[0]`: **Gas** (-1.0 = no gas, 1.0 = full gas)
- `action[1]`: **Brake** (-1.0 = no brake, 1.0 = full brake)
- `action[2]`: **Steering** (-1.0 = full left, 1.0 = full right)

Note: In practice, gas/brake use range [0, 1] and steering uses [-1, 1].


###
uv run trainer.py --num-updates 1000 --trajectory-path trajectory_data.pkl


Dataset Structure

Files:
- pretrain.h5 - Training dataset (120 episodes, 79,991 transitions)
- preval.h5 - Validation dataset

Top-level groups:
1. actions - Shape: (79991, 3)
- Expert actions: [gas, brake, steering]
2. observations/ - Group containing:
- speed - Shape: (79991,) - Current speed
- gear - Shape: (79991,) - Current gear
- rpm - Shape: (79991,) - Current RPM
- images - Shape: (79991, 4, 64, 64) - 4 stacked grayscale images (64x64)
- act1 - Shape: (79991, 3) - Previous action t-1
- act2 - Shape: (79991, 3) - Previous action t-2
3. metadata/ - Group containing:
- episode_lengths - Shape: (120,) - Length of each episode
- episode_rewards - Shape: (120,) - Total reward for each episode

Key Statistics:
- Total episodes: 120
- Total transitions: 79,991
- Mean episode length: ~667 steps
- Mean episode reward: ~210-211 points
- Sample episode lengths: [557, 537, 503, 552, 540, ...]

Important Note:
The dataset contains episode-level rewards only (not per-timestep rewards). This means
you'll need to:
1. Distribute the episode reward across timesteps, OR
2. Calculate discounted returns (G_t) by working backwards from episode end

The observations follow the TMRL format: (speed, gear, rpm, images, act1, act2) which
matches the network architecture in agent.py and network.py.