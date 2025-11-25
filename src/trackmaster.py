import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from argparse import ArgumentParser
import os
import torch
import logging

logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# rtgym environment class (full TrackMania Gymnasium environment)
env_cls = cfg_obj.ENV_CLS
device_worker = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using Device: {device_worker}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test mode using RolloutWorker (standalone)')
    parser.add_argument('--train', action='store_true', help='Standalone training mode (no distributed setup)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    # Legacy distributed training arguments (deprecated in favor of standalone training)
    parser.add_argument('--server', action='store_true', help='[DEPRECATED] Launch TMRL server for distributed training')
    parser.add_argument('--trainer', action='store_true', help='[DEPRECATED] Launch TMRL trainer for distributed training')
    args = parser.parse_args()

    if args.train:
        # Standalone training mode - no server/worker separation
        logger.info("Starting standalone PPO training...")
        import tmrl
        from src.standalone_trainer import PPOStandaloneTrainer
        from src.utils import obs_preprocessor

        # Get environment directly (standalone mode)
        env = tmrl.get_environment()
        logger.info(f"Environment loaded: {env.observation_space}, {env.action_space}")

        # Create trainer
        trainer = PPOStandaloneTrainer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            # PPO hyperparameters - adjust as needed
            policy_lr=1e-5,
            critic_lr=1e-5,
            gamma=0.996,
            clip_coef=0.2,
            critic_coef=0.1,
            entropy_coef=0.1,
            batch_size=256,
            num_updates=10000,
            epochs_per_update=100,
            max_episode_steps=2400,
            norm_advantages=True,
            grad_clip_val=0.1
        )

        # Wait for user to focus on TrackMania window
        logger.info("Please focus on TrackMania window...")
        import time
        time.sleep(3.0)

        # Start training
        results = trainer.train(env)

        logger.info("Training completed!")
        logger.info(f"Final reward: {results['cum_rewards'][-1]:.2f}")

    elif args.evaluate:
        # Evaluate a trained model
        logger.info("Starting evaluation...")
        import tmrl
        from src.standalone_trainer import PPOStandaloneTrainer

        env = tmrl.get_environment()
        logger.info(f"Environment loaded: {env.observation_space}, {env.action_space}")

        # Create trainer and load checkpoint
        trainer = PPOStandaloneTrainer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        checkpoint_path = input("Enter checkpoint path: ")
        trainer.load_model(checkpoint_path)

        # Wait for user to focus on TrackMania window
        logger.info("Please focus on TrackMania window...")
        import time
        time.sleep(3.0)

        # Evaluate
        avg_reward = trainer.evaluate(env, num_episodes=5)
        logger.info(f"Evaluation complete. Average reward: {avg_reward:.2f}")

    elif args.test:
        # Legacy test mode using RolloutWorker (standalone, no server)
        logger.info("Starting test mode with RolloutWorker...")
        from src.actor import TrackmasterActorModule
        from src.utils import obs_preprocessor
        from tmrl.networking import RolloutWorker

        rollout_worker = RolloutWorker(
            env_cls=env_cls,
            actor_module_cls=TrackmasterActorModule,
            device=device_worker,
            obs_preprocessor=obs_preprocessor,
            standalone=True  # Standalone mode, no server connection
        )

        rollout_worker.run(test_episode_interval=1)

    elif args.server:
        # [DEPRECATED] Distributed training: Server
        logger.warning("Distributed training (--server) is deprecated. Use --train for standalone training.")
        import time
        from tmrl.networking import Server

        server = Server()
        while True:
            time.sleep(1.0)

    elif args.trainer:
        # [DEPRECATED] Distributed training: Trainer
        logger.warning("Distributed training (--trainer) is deprecated. Use --train for standalone training.")
        logger.error("Distributed training setup is incomplete. Please use --train for standalone training.")

    else:
        parser.print_help()
