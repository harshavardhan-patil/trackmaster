# TMRL Concepts for TrackMaster Project

## Table of Contents
1. [Introduction](#introduction)
2. [TMRL Architecture Overview](#tmrl-architecture-overview)
3. [Core Components](#core-components)
4. [Real-Time Reinforcement Learning with rtgym](#real-time-reinforcement-learning-with-rtgym)
5. [ActorModule Interface](#actormodule-interface)
6. [TrainingAgent Interface](#trainingagent-interface)
7. [Memory Management](#memory-management)
8. [Observation and Action Spaces](#observation-and-action-spaces)
9. [RL Algorithms](#rl-algorithms)
10. [Neural Network Architectures](#neural-network-architectures)
11. [Communication and Networking](#communication-and-networking)
12. [TrackMania Environment Setup](#trackmania-environment-setup)

---

## Introduction

**TMRL (TrackMania Reinforcement Learning)** is a comprehensive distributed framework specifically designed for training Deep Reinforcement Learning agents in real-time applications. It was developed at Polytechnique Montreal and hosts the TrackMania Roborace League, a vision-based AI competition for TrackMania 2020.

### Key Features
- **Real-time training**: Policies are trained without pausing the simulation
- **Distributed architecture**: Single-server, multiple-clients design
- **Gymnasium compatibility**: Uses standardized Gymnasium API via rtgym
- **Vision-based control**: Processes raw screenshots with CNNs
- **Analog control**: Uses virtual gamepad for continuous action spaces
- **State-of-the-art algorithms**: Includes SAC, REDQ, and supports custom implementations

---

## TMRL Architecture Overview

TMRL uses a **distributed client-server architecture** similar to Ray RLlib, enabling efficient collection of training samples from multiple sources while training centrally.

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                         CENTRAL SERVER                       │
│  - Manages communication between workers and trainer         │
│  - Forwards samples from workers to trainer                  │
│  - Broadcasts updated policy weights to all workers          │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
         │ samples            │ samples            │ weights
         │ & requests         │ & requests         │ updates
         ↓                    ↓                    ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ RolloutWorker 1 │  │ RolloutWorker 2 │  │ RolloutWorker N │
│ - Runs env      │  │ - Runs env      │  │ - Runs env      │
│ - Collects data │  │ - Collects data │  │ - Collects data │
│ - Uses Actor    │  │ - Uses Actor    │  │ - Uses Actor    │
└─────────────────┘  └─────────────────┘  └─────────────────┘

                              ↓ samples

                   ┌─────────────────────┐
                   │      TRAINER        │
                   │ - Stores samples    │
                   │ - Trains policy     │
                   │ - Updates weights   │
                   └─────────────────────┘
```

### Benefits
- **Scalability**: Add workers to increase sample collection rate
- **HPC compatibility**: Train on clusters without port-forwarding
- **Flexibility**: Workers and trainer can run on different machines
- **Real-time operation**: No simulation pausing required

---

## Core Components

### 1. Server

The **Server** is the central communication hub between all entities.

**Responsibilities:**
- Listens for incoming connections from workers and trainer
- Receives compressed samples from RolloutWorkers
- Forwards samples to Trainer
- Receives updated policy weights from Trainer
- Broadcasts weights to all connected RolloutWorkers

**Instantiation:**
```python
from tmrl.networking import Server

server = Server(
    security=None,  # or "TLS" for secure communication
    password="secure_password",
    port=6666
)
```

**Key Points:**
- Should be instantiated first
- Must be accessible via network (requires port forwarding if over Internet)
- Runs continuously, waiting for connections

---

### 2. RolloutWorker

The **RolloutWorker** encapsulates the environment and policy (ActorModule) to collect training samples.

**Responsibilities:**
- Runs the Gymnasium environment (TrackMania game)
- Executes the current policy to collect samples
- Compresses samples using custom compressors
- Sends compressed samples to Server
- Receives and loads updated policy weights
- Can run test episodes periodically

**Instantiation:**
```python
from tmrl.networking import RolloutWorker
from tmrl.util import partial

worker = RolloutWorker(
    env_cls=env_cls,                    # Gymnasium environment class
    actor_module_cls=actor_module_cls,  # Policy class (ActorModule)
    sample_compressor=compressor,       # Optional compression
    device="cpu",                       # Device for inference
    server_ip="127.0.0.1",             # Server IP address
    server_port=6666,
    password="secure_password",
    max_samples_per_episode=np.inf,     # Episode length limit
    model_path="path/to/weights.tmod"   # Where to save/load weights
)

# Start collecting samples
worker.run(test_episode_interval=10)  # Test every 10 episodes
```

**Key Configuration:**
- `env_cls`: Must be wrapped in `GenericGymEnv`
- `actor_module_cls`: Your policy implementation (ActorModule)
- `sample_compressor`: Optional function to reduce bandwidth
- `device`: Use "cpu" for inference unless you have GPU worker
- `max_samples_per_episode`: Prevents infinite episodes

---

### 3. Trainer

The **Trainer** handles the training process using samples from RolloutWorkers.

**Responsibilities:**
- Connects to Server to receive samples
- Stores samples in replay memory (with decompression)
- Trains the policy using a TrainingAgent
- Periodically sends updated weights to Server
- Logs training metrics (optionally to wandb)
- Checkpoints training progress

**Instantiation:**
```python
from tmrl.networking import Trainer
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial

# Define training configuration
training_cls = partial(
    TorchTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=10,
    rounds=50,
    steps=2000,
    update_buffer_interval=100,
    update_model_interval=100,
    max_training_steps_per_env_step=1.0,
    start_training=0,
    device="cuda"  # or "cpu"
)

# Create trainer
trainer = Trainer(
    training_cls=training_cls,
    server_ip="127.0.0.1",
    server_port=6666,
    password="secure_password",
    model_path="path/to/trainer_weights.tmod",
    checkpoint_path="path/to/checkpoint.tcpt"
)

# Start training
trainer.run()
# Or with wandb logging:
# trainer.run_with_wandb(entity="wandb_entity", project="project_name", run_id="run_name")
```

**Training Parameters:**
- `epochs`: Number of checkpointing intervals
- `rounds`: Number of rounds per epoch (for logging)
- `steps`: Training steps per round
- `update_buffer_interval`: How often to check for new samples
- `update_model_interval`: How often to broadcast updated weights
- `max_training_steps_per_env_step`: Prevents training too fast
- `start_training`: Minimum samples before training begins

---

## Real-Time Reinforcement Learning with rtgym

**rtgym (Real-Time Gym)** is a framework that enables efficient real-time implementations of Delayed Markov Decision Processes (DMDPs) in real-world applications.

### Key Concepts

#### 1. Time-Step Constraints
- Time-steps are **elastically constrained** to their nominal duration
- If constraint cannot be satisfied, the time-step **times out**
- Next time-step starts from current timestamp
- This handles inherent delays in real-time systems

#### 2. Action Buffer
- Real-time systems have communication delays
- To maintain Markov property, observations include **action history**
- Default: last 4 actions are part of observation
- Critical for proper state representation in real-time

#### 3. Episode Management
In rtgym environments:
- Very last action of episode is **never sent** to environment
- After `reset()`, the **default action** is applied
- This default action becomes part of post-reset observation
- Important for correct transition reconstruction in memory

### rtgym Configuration for TrackMania

```python
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT

config = DEFAULT_CONFIG_DICT.copy()
config["interface"] = TMInterface  # Your custom interface
config["time_step_duration"] = 0.05  # 50ms per step (20 Hz)
config["start_obs_capture"] = 0.05   # When to capture observation
config["time_step_timeout_factor"] = 1.0  # Timeout threshold
config["ep_max_length"] = 1000       # Maximum episode length
config["act_buf_len"] = 4            # Action buffer length
config["reset_act_buf"] = False      # Don't reset buffer on reset()
```

### Implementing RealTimeGymInterface

```python
from rtgym import RealTimeGymInterface
import gymnasium.spaces as spaces

class TMInterface(RealTimeGymInterface):
    def get_observation_space(self):
        # Define your observation space
        pass

    def get_action_space(self):
        # Define your action space
        pass

    def get_default_action(self):
        # Default action (e.g., no input)
        return np.array([0.0, 0.0, 0.0])

    def send_control(self, control):
        # Send control to game (via virtual gamepad)
        pass

    def reset(self, seed=None, options=None):
        # Reset environment, return initial observation
        pass

    def get_obs_rew_terminated_info(self):
        # Capture current state
        pass
```

---

## ActorModule Interface

The **ActorModule** is the policy interface in TMRL. It defines how your trained agent selects actions.

### Base Requirements

```python
from tmrl.actor import ActorModule

class MyActorModule(ActorModule):
    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
        """
        super().__init__(observation_space, action_space)
        # Initialize your model

    def act(self, obs, test=False):
        """
        Compute action from observation.

        Args:
            obs: observation from environment
            test: True for deterministic/greedy policy, False for exploration

        Returns:
            numpy.array: action to take
        """
        # Your policy logic here
        pass
```

### TorchActorModule

For PyTorch models, use `TorchActorModule` which provides additional conveniences:

```python
from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn

class MyTorchActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Define your neural network
        self.network = nn.Sequential(
            # Your layers
        )

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Forward pass through network.

        Args:
            obs: torch.Tensor observation
            test: True for deterministic action
            compute_logprob: True to compute log probabilities (for training)

        Returns:
            action: torch.Tensor
            logp: torch.Tensor or None (log probability)
        """
        # Forward pass logic
        pass

    def act(self, obs, test=False):
        """Called by TMRL during rollout (automatically no_grad)"""
        with torch.no_grad():
            action, _ = self.forward(obs, test, compute_logprob=False)
            return action.cpu().numpy()
```

### Custom Serialization (JSON for Safety)

For competition/deployment, avoid pickle files (security risk). Use JSON:

```python
import json
import torch

class TorchJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)

class TorchJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct

class MyActorModule(TorchActorModule):
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.state_dict(), f, cls=TorchJSONEncoder)

    def load(self, path, device):
        with open(path, 'r') as f:
            state_dict = json.load(f, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self
```

---

## TrainingAgent Interface

The **TrainingAgent** implements your RL algorithm (e.g., PPO, SAC).

### Base Requirements

```python
from tmrl.training import TrainingAgent

class MyTrainingAgent(TrainingAgent):
    def __init__(self, observation_space, action_space, device, **kwargs):
        """
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            device: torch device (e.g., "cuda:0" or "cpu")
            **kwargs: Your custom hyperparameters
        """
        super().__init__(observation_space, action_space, device)

        # Initialize your models, optimizers, etc.
        self.actor_critic = ActorCriticModel(...)
        self.optimizer = torch.optim.Adam(...)

    def train(self, batch):
        """
        Perform one training step.

        Args:
            batch: tuple of (obs, action, reward, next_obs, terminated, truncated)
                   All elements are batched torch.Tensors on self.device

        Returns:
            dict: metrics to log (e.g., {"loss": 0.5, "value": 10.2})
        """
        obs, action, reward, next_obs, terminated, truncated = batch

        # Your training logic here
        loss = compute_loss(...)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def get_actor(self):
        """
        Return current ActorModule for broadcasting to workers.

        Returns:
            ActorModule: current policy (without gradients)
        """
        return self.actor_critic.actor.no_grad_copy()
```

### Example: SAC TrainingAgent

```python
class SACTrainingAgent(TrainingAgent):
    def __init__(self, observation_space, action_space, device,
                 gamma=0.99, polyak=0.995, alpha=0.2,
                 lr_actor=1e-3, lr_critic=1e-3):
        super().__init__(observation_space, action_space, device)

        # Actor-Critic model
        self.model = ActorCriticModel(observation_space, action_space).to(device)
        self.model_target = deepcopy(self.model).to(device)

        # Hyperparameters
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha

        # Optimizers
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=lr_actor)
        self.q_optimizer = Adam(
            itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
            lr=lr_critic
        )

    def train(self, batch):
        obs, action, reward, next_obs, terminated, _ = batch

        # Compute critic loss
        with torch.no_grad():
            next_action, log_prob = self.model.actor(next_obs)
            target_q1 = self.model_target.q1(next_obs, next_action)
            target_q2 = self.model_target.q2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            backup = reward + self.gamma * (1 - terminated) * target_q

        q1 = self.model.q1(obs, action)
        q2 = self.model.q2(obs, action)
        loss_q = ((q1 - backup)**2 + (q2 - backup)**2).mean()

        # Update critics
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Compute actor loss
        new_action, log_prob = self.model.actor(obs)
        q_pi = torch.min(self.model.q1(obs, new_action),
                         self.model.q2(obs, new_action))
        loss_pi = (self.alpha * log_prob - q_pi).mean()

        # Update actor
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Update target networks
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(),
                                 self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return {
            "loss_actor": loss_pi.item(),
            "loss_critic": loss_q.item()
        }

    def get_actor(self):
        return no_grad_copy(self.model.actor)
```

---

## Memory Management

Memory (replay buffer) management is crucial for efficient training, especially with high-dimensional observations like images.

### TorchMemory Interface

```python
from tmrl.memory import TorchMemory

class MyMemory(TorchMemory):
    def __init__(self, device, nb_steps, sample_preprocessor=None,
                 memory_size=1000000, batch_size=256, dataset_path="", **kwargs):
        super().__init__(device, nb_steps, sample_preprocessor,
                        memory_size, batch_size, dataset_path)

        # Your custom storage structure
        self.data = []

    def append_buffer(self, buffer):
        """
        Add samples from RolloutWorker to memory.

        Args:
            buffer: object with .memory attribute containing list of samples
                   Each sample: (action, new_obs, reward, terminated, truncated, info)
        """
        # Decompress and store samples
        for sample in buffer.memory:
            action, obs, reward, terminated, truncated, info = sample
            # Store in self.data
            self.data.append(...)

        # Trim if exceeds memory_size
        if len(self) > self.memory_size:
            self.data = self.data[-self.memory_size:]

    def __len__(self):
        """Return number of valid transitions available"""
        return len(self.data)

    def get_transition(self, item):
        """
        Retrieve a single transition.

        Args:
            item: int index

        Returns:
            tuple: (last_obs, action, reward, new_obs, terminated, truncated, info)
        """
        # Reconstruct full transition from compressed storage
        return self.data[item]
```

### Sample Compression

To reduce network bandwidth, compress samples before sending:

```python
def my_sample_compressor(action, obs, reward, terminated, truncated, info):
    """
    Compress sample before sending over network.

    For images: store only new image, not full history
    For actions: remove action buffer from obs (already in action)

    Returns:
        tuple: compressed (action, obs, reward, terminated, truncated, info)
    """
    # Example: remove action buffer from observation
    compressed_obs = obs[:-4]  # Assuming last 4 elements are actions

    return action, compressed_obs, reward, terminated, truncated, info
```

Then decompress in `get_transition()`:

```python
def get_transition(self, item):
    # Retrieve compressed data
    action, compressed_obs, reward, terminated, truncated, info = self.data[item]

    # Reconstruct action buffer
    actions = [self.data[i][0] for i in range(item-3, item+1)]
    full_obs = (*compressed_obs, *actions)

    # Get previous observation similarly
    prev_obs = self.get_prev_obs(item-1)

    return prev_obs, action, reward, full_obs, terminated, truncated, info
```

### CRC Debugging

TMRL provides CRC (Cyclic Redundancy Check) debugging to verify compression/decompression:

```python
memory_cls = partial(MyMemory, crc_debug=True, ...)
worker = RolloutWorker(..., crc_debug=True, ...)
```

In CRC mode:
- Worker stores full transition in `info` dict
- Memory compares decompressed transition with stored version
- Prints "CRC check passed" or shows mismatches
- **Only use for debugging** (destroys compression benefit)

---

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

### Observation Preprocessing

You can preprocess observations before they reach your ActorModule:

```python
def obs_preprocessor(obs):
    """
    Preprocess observation.

    Args:
        obs: tuple from environment

    Returns:
        preprocessed observation
    """
    speed, gear, rpm, images, act1, act2, act3, act4 = obs

    # Example: normalize speed
    speed = speed / 1000.0

    # Example: convert to float32
    images = images.astype(np.float32) / 255.0

    return speed, gear, rpm, images, act1, act2, act3, act4

worker = RolloutWorker(..., obs_preprocessor=obs_preprocessor)
```

---

## RL Algorithms

### Soft Actor-Critic (SAC)

**SAC** is the primary algorithm used in TMRL. It's an off-policy algorithm with the following features:

**Key Characteristics:**
- **Off-policy**: Can learn from old experiences (replay buffer)
- **Maximum entropy**: Encourages exploration via entropy regularization
- **Actor-Critic**: Separates policy (actor) and value estimation (critic)
- **Stable**: More stable than other methods like DDPG

**Advantages for TrackMania:**
- Sample-efficient (reuses past experiences)
- Robust to hyperparameter changes
- Good exploration-exploitation trade-off
- Works well with continuous actions

**Challenges:**
- Requires careful tuning of entropy coefficient (α)
- Can be slower to train than on-policy methods
- May need large replay buffers for best performance

**Typical Hyperparameters:**
```python
gamma = 0.995           # Discount factor (higher for racing)
polyak = 0.995          # Target network update rate
alpha = 0.01            # Entropy coefficient
lr_actor = 1e-5         # Actor learning rate
lr_critic = 5e-5        # Critic learning rate (often higher than actor)
batch_size = 256        # Batch size
memory_size = 1000000   # Replay buffer size
```

### Proximal Policy Optimization (PPO)

**PPO** is an on-policy algorithm suitable for your TrackMaster project:

**Key Characteristics:**
- **On-policy**: Uses only recent experiences
- **Clipped objective**: Prevents too large policy updates
- **Simple**: Fewer hyperparameters than SAC
- **Stable**: Good baseline for new domains

**Advantages for TrackMania:**
- Simpler to implement and tune
- Often faster wall-clock training time
- Works well with smaller batch sizes
- More predictable behavior

**Challenges:**
- Less sample-efficient (discards old data)
- Requires multiple epochs per batch
- May need careful tuning of clip ratio

**Typical Hyperparameters:**
```python
gamma = 0.99            # Discount factor
gae_lambda = 0.95       # GAE parameter
clip_ratio = 0.2        # PPO clip parameter
lr = 3e-4               # Learning rate
epochs_per_rollout = 10 # Training epochs per data collection
batch_size = 64         # Batch size
n_steps = 2048          # Steps per rollout
```

**Implementation Note:** While TMRL's `TrainingOffline` is designed for off-policy algorithms, PPO can be implemented with synchronization tricks (waiting for new samples before each update).

### Algorithm Comparison

| Feature | SAC | PPO |
|---------|-----|-----|
| Sample Efficiency | High | Medium |
| Training Time | Slower | Faster |
| Stability | Very Stable | Stable |
| Hyperparameter Sensitivity | Medium | Low |
| Memory Requirements | High | Low |
| Implementation Complexity | High | Medium |

**Recommendation:** Start with SAC if you have computational resources and can collect samples continuously. Use PPO if you want faster iteration and simpler tuning.

---

## Neural Network Architectures

### CNN-Based Architecture (Standard)

For vision-based control, CNNs are the standard choice:

```python
class VanillaCNN(nn.Module):
    def __init__(self, q_net=False):
        super().__init__()

        # Convolutional layers for image processing
        # Input: 4 grayscale images (64x64) stacked as channels
        self.conv1 = nn.Conv2d(4, 64, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2)

        # Calculate flattened size
        self.flat_features = self._calculate_flat_features()

        # MLP for combining image features with telemetry
        # Input: flattened CNN output + speed + gear + rpm + action history
        mlp_input_size = self.flat_features + 9  # 3 telemetry + 6 actions
        if q_net:
            mlp_input_size += 3  # + current action for Q-network

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, obs):
        speed, gear, rpm, images, act1, act2 = obs[:6]

        # CNN forward pass
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Concatenate with telemetry and actions
        x = torch.cat([speed, gear, rpm, x, act1, act2], dim=-1)

        # MLP forward pass
        x = self.mlp(x)
        return x
```

**Design Principles:**
- Start with larger kernels (8x8) to capture large features
- Gradually increase channels (64 → 128)
- Use stride for downsampling (more efficient than pooling)
- Combine visual features with telemetry late in network
- Keep action history separate until final layers

### Vision Transformer (ViT) Architecture (Advanced)

Vision Transformers are a promising direction for future research:

```python
class VisionTransformerActor(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=4,
                 embed_dim=256, depth=6, num_heads=8):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Telemetry embedding
        self.telemetry_embed = nn.Linear(9, embed_dim)  # speed, gear, rpm, 6 actions

        # Output head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, obs):
        speed, gear, rpm, images, act1, act2 = obs[:6]
        batch_size = images.shape[0]

        # Patch embedding
        x = self.patch_embed(images)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed

        # Embed telemetry
        telemetry = torch.cat([speed, gear, rpm, act1, act2], dim=-1)
        telem_embed = self.telemetry_embed(telemetry).unsqueeze(1)

        # Concatenate telemetry token
        x = torch.cat([telem_embed, x], dim=1)

        # Transformer
        x = self.transformer(x)

        # Use [CLS] token (telemetry token)
        x = x[:, 0]

        # Output
        x = self.head(x)
        return x
```

**ViT Advantages:**
- Better at capturing long-range spatial dependencies
- Can attend to different parts of image selectively
- Potentially better generalization to new tracks
- Pre-training possible (though not standard for RL)

**ViT Challenges:**
- Requires more data to train
- Higher computational cost
- More memory intensive
- Needs careful hyperparameter tuning

**Recommendations:**
- Start with CNN architecture (proven and efficient)
- Consider ViT if you have:
  - Large computational budget
  - Many diverse training tracks
  - Time for extensive hyperparameter search

### Hybrid CNN-RNN Architecture

For temporal dependencies:

```python
class CNNLSTMActor(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN for single frame
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, 2),  # Process single grayscale image
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        cnn_output_size = self._calculate_cnn_output()

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )

        # Output
        self.head = nn.Linear(256 + 9, 256)  # LSTM output + telemetry

    def forward(self, obs, hidden=None):
        speed, gear, rpm, images, act1, act2 = obs[:6]
        batch_size, num_frames = images.shape[0], images.shape[1]

        # Process each frame with CNN
        images = images.view(-1, 1, 64, 64)  # Flatten batch and time
        cnn_out = self.cnn(images)
        cnn_out = cnn_out.view(batch_size, num_frames, -1)

        # LSTM
        lstm_out, hidden = self.lstm(cnn_out, hidden)
        lstm_out = lstm_out[:, -1]  # Use last timestep

        # Combine with telemetry
        telemetry = torch.cat([speed, gear, rpm, act1, act2], dim=-1)
        x = torch.cat([lstm_out, telemetry], dim=-1)

        return self.head(x), hidden
```

---

## Communication and Networking

### Network Security

TMRL uses **tlspyo** (TLS Python Object) for secure communication.

#### Basic Setup (Local/Trusted Network)
```python
security = None
password = "your_secure_password"
```

#### TLS Setup (Internet/Untrusted Network)
```python
security = "TLS"
password = "your_secure_password"

# Generate TLS certificate (one-time, on server machine):
# from tlspyo import make_certificate
# make_certificate("path/to/cert.crt", "path/to/key.key")
```

**Important Security Notes:**
- Always use strong passwords
- Use TLS when training over public networks
- Never load untrusted pickle files
- Use JSON serialization for sharing models

### Network Configuration

#### Localhost (Single Machine)
```python
server_ip = "127.0.0.1"
server_port = 6666
```

#### Multiple Machines
```python
# config.json on all machines:
{
    "LOCALHOST_WORKER": false,
    "LOCALHOST_TRAINER": false,
    "PUBLIC_IP_SERVER": "192.168.1.100",  # or public IP
    "PORT": 6666
}
```

**Server Machine:**
- Must be accessible via network
- Requires port forwarding if accessed over Internet
- Can run on same machine as worker or trainer

**Worker Machine:**
- Needs server IP
- Should prioritize real-time environment performance
- Avoid heavy computation (unless dedicated GPU for inference)

**Trainer Machine:**
- Needs server IP
- Should have GPU for training
- Can be on HPC cluster

### Bandwidth Optimization

Typical sample sizes (per transition):
- **Uncompressed**: 4 images (64x64) + telemetry ≈ 16 KB
- **Compressed** (single image): 1 image + telemetry ≈ 4 KB
- **Over 1000 steps**: 4-16 MB

**Optimization Strategies:**
1. **Sample Compression**: Store only new image, not full history
2. **Action Buffer Removal**: Actions already in transition
3. **Image Compression**: Use JPEG compression (lossy but acceptable)
4. **Batching**: Send samples in batches, not individually

Example compression:
```python
def compress_sample(action, obs, reward, terminated, truncated, info):
    speed, gear, rpm, images, act1, act2, act3, act4 = obs

    # Only keep latest image (others reconstructable from history)
    new_image = images[-1]

    # Remove action buffer (reconstructable from action history)
    compressed_obs = (speed, gear, rpm, new_image)

    return action, compressed_obs, reward, terminated, truncated, info
```

---

## TrackMania Environment Setup

### Prerequisites

1. **TrackMania 2020**: Full game installed
2. **OpenPlanet**: Plugin system for TM
   - Download from [openplanet.dev](https://openplanet.dev/)
   - Install TMRL plugin
3. **TMRL**: Python package
   ```bash
   pip install tmrl
   ```
4. **vgamepad**: Virtual gamepad driver (Windows)
   - Automatically prompted during install
   - Required for analog control

### Configuration

Edit `config.json` in TmrlData folder:

```json
{
    "RUN_NAME": "trackmaster_v1",
    "WINDOW_WIDTH": 256,
    "WINDOW_HEIGHT": 128,
    "IMG_WIDTH": 64,
    "IMG_HEIGHT": 64,
    "IMG_GRAYSCALE": true,
    "IMG_HIST_LEN": 4,
    "ACT_BUF_LEN": 4,

    "LOCALHOST_WORKER": true,
    "LOCALHOST_TRAINER": false,
    "PUBLIC_IP_SERVER": "127.0.0.1",
    "PORT": 6666,
    "PASSWORD": "trackmaster_secure_pwd",
    "SECURITY": null,

    "MAX_EPOCHS": 100,
    "ROUNDS_PER_EPOCH": 10,
    "TRAINING_STEPS_PER_ROUND": 1000,
    "UPDATE_MODEL_INTERVAL": 1000,
    "UPDATE_BUFFER_INTERVAL": 100,
    "MEMORY_SIZE": 1000000,
    "BATCH_SIZE": 256,

    "ENVIRONMENT_STEPS_BEFORE_TRAINING": 2000,
    "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 1.0,

    "CUDA_TRAINING": true,
    "CUDA_INFERENCE": false,

    "WANDB_PROJECT": "trackmaster",
    "WANDB_ENTITY": "your_username",
    "WANDB_RUN_ID": "run_001",
    "WANDB_KEY": "your_wandb_api_key"
}
```

### Running the Pipeline

#### Step 1: Start Server
```bash
python trackmaster_training.py --server
```

#### Step 2: Start Trainer
```bash
python trackmaster_training.py --trainer
```

#### Step 3: Start Worker(s)
```bash
# Terminal 1
python trackmaster_training.py --worker

# Terminal 2 (optional, for parallel collection)
python trackmaster_training.py --worker
```

#### Step 4: Launch TrackMania
1. Open TrackMania 2020
2. Ensure OpenPlanet is running
3. Load a track
4. Press F3 to open OpenPlanet menu
5. Enable TMRL interface

### Track Selection

For training:
- Start with simple tracks (wide, few turns)
- Gradually increase difficulty
- Use variety to improve generalization
- TrackMania Exchange has downloadable tracks

### Monitoring Training

**Terminal Output:**
- Round statistics every N training steps
- Loss values, episode rewards
- Sample collection rate

**Wandb Dashboard:**
- Real-time loss curves
- Episode statistics
- Model checkpoints
- Hyperparameter tracking

**Saved Files:**
- `TmrlData/weights/`: Model checkpoints
- `TmrlData/checkpoints/`: Full training state
- `TmrlData/logs/`: Training logs

---

## Best Practices and Tips

### For Training
1. **Start Simple**: Use SAC with default hyperparameters first
2. **Monitor Overfitting**: Watch for divergence between train/test performance
3. **Checkpoint Frequently**: Training can take days
4. **Use Test Episodes**: Run deterministic policy periodically to gauge progress
5. **Track Diversity**: Train on multiple tracks for generalization

### For Network Architecture
1. **Start with CNN**: Proven architecture, easier to train
2. **Normalize Inputs**: Speed/1000, images/255
3. **Batch Normalization**: Can help with training stability
4. **Gradual Complexity**: Add layers/features incrementally

### For Debugging
1. **Use CRC Debug**: Verify compression/decompression
2. **Check Data Flow**: Print shapes at each stage
3. **Visualize Observations**: Ensure images are captured correctly
4. **Log Everything**: Actions, rewards, observations
5. **Start Offline**: Test components independently before full pipeline

### For Performance
1. **Separate Machines**: Worker on one, trainer on another
2. **GPU for Training**: Essential for CNN training
3. **CPU for Worker**: Inference is fast enough
4. **Sample Compression**: Critical for network efficiency
5. **Batch Size**: Balance memory usage and training stability

---

## Common Issues and Solutions

### Issue: Worker Timing Out
**Cause**: Trainer consuming too much CPU/GPU, slowing worker
**Solution**: Run worker and trainer on separate machines, or reduce trainer load

### Issue: Training Not Starting
**Cause**: Not enough samples in buffer
**Solution**: Lower `start_training` parameter or wait longer

### Issue: Loss Exploding
**Cause**: Learning rate too high or architecture issue
**Solution**: Reduce learning rate, add gradient clipping, check network architecture

### Issue: Poor Generalization
**Cause**: Overfitting to single track
**Solution**: Train on multiple diverse tracks, add data augmentation

### Issue: Slow Training
**Cause**: Sample collection bottleneck
**Solution**: Add more workers, optimize sample compression, check network bandwidth

---

## Conclusion

TMRL is a powerful framework for real-time RL, specifically tailored for TrackMania but applicable to other domains. Key takeaways:

1. **Distributed Architecture**: Scales from laptop to HPC cluster
2. **Real-Time Focus**: Handles delays and timing constraints
3. **Flexible**: Customize every component (algorithm, architecture, memory)
4. **Practical**: Proven in competition settings
5. **Extensible**: Easy to implement new algorithms and models

For your TrackMaster project:
- Use PPO or SAC (SAC recommended for sample efficiency)
- Start with CNN, consider ViT if resources allow
- Implement proper compression for image observations
- Train on diverse tracks for generalization
- Monitor training closely (use wandb)

Good luck with your TrackMaster project!

---

## Additional Resources

- [TMRL GitHub](https://github.com/trackmania-rl/tmrl)
- [TMRL Documentation](https://tmrl.readthedocs.io/)
- [rtgym Documentation](https://github.com/yannbouteiller/rtgym)
- [OpenPlanet](https://openplanet.dev/)
- [TrackMania Exchange](https://trackmania.exchange/) (for tracks)
- [Spinning Up in Deep RL](https://spinningup.openai.com/) (for RL fundamentals)
- [Proximal Policy Optimization Paper](https://arxiv.org/abs/1707.06347)
- [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
