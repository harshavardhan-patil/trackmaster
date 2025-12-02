"""
Supervised Learning Pre-training for Actor Network
Uses expert demonstrations to pre-train the policy before RL training
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import mlflow
from src.agent import Agent
from torch.optim import Adam, lr_scheduler


class ExpertDataset(Dataset):
    """Dataset for expert demonstrations"""

    def __init__(self, h5_path: str):
        print(f"Loading dataset from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            # Load observations
            self.speeds = np.array(f['observations/speed'])
            self.gears = np.array(f['observations/gear'])
            self.rpms = np.array(f['observations/rpm'])
            self.images = np.array(f['observations/images'])
            self.act1 = np.array(f['observations/act1'])
            self.act2 = np.array(f['observations/act2'])

            # Load expert actions
            self.actions = np.array(f['actions'])

            # Load metadata
            episode_lengths = np.array(f['metadata/episode_lengths'])

        print(f"  Loaded {len(self.actions)} transitions from {len(episode_lengths)} episodes")
        print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        obs = (
            self.speeds[idx],
            self.gears[idx],
            self.rpms[idx],
            self.images[idx],
            self.act1[idx],
            self.act2[idx]
        )
        action = self.actions[idx]
        return obs, action


def collate_fn(batch):
    """Batch observations properly"""
    observations, actions = zip(*batch)
    speeds, gears, rpms, images, act1s, act2s = zip(*observations)

    # Convert to batched tensors
    speeds_t = torch.tensor(np.array(speeds), dtype=torch.float32).unsqueeze(1)
    gears_t = torch.tensor(np.array(gears), dtype=torch.float32).unsqueeze(1)
    rpms_t = torch.tensor(np.array(rpms), dtype=torch.float32).unsqueeze(1)
    images_t = torch.tensor(np.array(images), dtype=torch.float32)
    act1s_t = torch.tensor(np.array(act1s), dtype=torch.float32)
    act2s_t = torch.tensor(np.array(act2s), dtype=torch.float32)
    actions_t = torch.tensor(np.array(actions), dtype=torch.float32)

    obs_tensor = (speeds_t, gears_t, rpms_t, images_t, act1s_t, act2s_t)
    return obs_tensor, actions_t


def train_epoch(agent: Agent, dataloader, optimizer, device):
    """Train for one epoch"""
    agent.train()
    total_loss = 0.0
    n_batches = 0

    for obs_batch, actions_batch in dataloader:
        obs_batch = tuple(t.to(device) for t in obs_batch)
        actions_batch = actions_batch.to(device)

        # Get mean actions, don't use mean_only which has no_grad
        features = agent.policy.backbone(obs_batch)
        means = agent.policy.action_mean(features)
        # Apply transformations: gas/brake use sigmoid, steering uses tanh
        predicted_actions = torch.stack([
            torch.sigmoid(means[:, 0]),  # Gas: 0 to 1
            torch.sigmoid(means[:, 1]),  # Brake: 0 to 1
            torch.tanh(means[:, 2])      # Steering: -1 to 1
        ], dim=1)

        loss = F.mse_loss(predicted_actions, actions_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(agent, dataloader, device):
    """Validate on validation set"""
    agent.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for obs_batch, actions_batch in dataloader:
            obs_batch = tuple(t.to(device) for t in obs_batch)
            actions_batch = actions_batch.to(device)

            predicted_actions = agent.policy.mean_only(obs_batch)
            loss = F.mse_loss(predicted_actions, actions_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Supervised pre-training for Actor")
    parser.add_argument('--train-data', type=str, default='pretrain.h5')
    parser.add_argument('--val-data', type=str, default='preval.h5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"\n{'='*60}")
    print(f"Supervised Pre-training - Actor Only")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets
    train_dataset = ExpertDataset(args.train_data)
    val_dataset = ExpertDataset(args.val_data)

    # DataLoaders with shuffling (critical to avoid sequential frames)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")

    # Initialize agent
    agent = Agent(action_space=3).to(device)
    print(f"Actor parameters: {sum(p.numel() for p in agent.policy.parameters()):,}\n")

    # Optimizer
    optimizer = Adam(agent.policy.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    mlflow.set_experiment("Trackmaster Pre-Training")
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'device': str(device),
            'train_data': args.train_data,
            'val_data': args.val_data
        })

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")

            train_loss = train_epoch(agent, train_loader, optimizer, device)
            val_loss = validate(agent, val_loader, device)
            scheduler.step()
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss
            }, step=epoch)

            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_pretrained.pt')
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved (val_loss: {best_val_loss:.6f})")
            print()

        # Log best validation loss
        mlflow.log_metric('best_val_loss', best_val_loss)

        # Save final model
        final_path = os.path.join(args.checkpoint_dir, 'final_pretrained.pt')
        torch.save(agent.state_dict(), final_path)

        # Log final model to MLflow
        mlflow.pytorch.log_model(agent, 'actor_model')

    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Final model saved: {final_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
