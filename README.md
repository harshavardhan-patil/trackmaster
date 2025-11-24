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