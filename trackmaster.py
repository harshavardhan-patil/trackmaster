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
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--trainer', action='store_true')
    args = parser.parse_args()

    if args.test:
        from src.actor import TrackmasterActorModule
        from src.utils import obs_preprocessor
        from tmrl.networking import RolloutWorker

        rollout_worker = RolloutWorker(
            env_cls=env_cls,
            actor_module_cls=TrackmasterActorModule,
            device=device_worker,
            obs_preprocessor=obs_preprocessor
        )

        rollout_worker.run(test_episode_interval=1) # how many occurences of initial state to terminal state (e.g: starting a race -> completing a race/crashing out)
    elif args.server:
        import time
        from tmrl.networking import Server

        server = Server()
        while True:
            time.sleep(1.0) # this is simply for us to focus on the TrackMania window
    elif args.trainer:
        from tmrl.networking import Trainer
        from src.trainer import PPOTrainingAgent
        trainer = Trainer(
            training_cls=
        )
