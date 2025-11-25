"""
TrainingOnline class for on-policy algorithms like PPO.

This module provides a custom training loop designed for online, on-policy
reinforcement learning algorithms such as Proximal Policy Optimization (PPO).
Unlike off-policy methods that use replay buffers, on-policy algorithms require
fresh trajectories collected with the current policy.

Key differences from TrainingOffline:
1. Uses episodic trajectory buffer instead of replay memory
2. Trains for multiple epochs on the same batch of trajectories
3. Clears buffer after training to maintain on-policy constraint
4. Synchronizes policy updates with data collection
"""

# standard library imports
import time
from dataclasses import dataclass
import logging

# third-party imports
import torch
from pandas import DataFrame

# local imports
from tmrl.util import pandas_dict


__docformat__ = "google"


@dataclass(eq=0)
class TrainingOnline:
    """
    Training wrapper for on-policy algorithms like PPO.

    This class is designed for online RL algorithms that require:
    - Fresh trajectories from the current policy
    - Multiple training epochs over the same data
    - Synchronous policy updates

    Args:
        env_cls (type): class of a dummy environment, used only to retrieve observation
            and action spaces if needed. Alternatively, this can be a tuple of the form
            (observation_space, action_space).
        memory_cls (type): class of the trajectory buffer (should support episodic storage)
        training_agent_cls (type): class of the training agent (must implement PPO-style training)
        epochs (int): total number of epochs, we save the agent every epoch
        rounds (int): number of rounds per epoch, we generate statistics every round
        update_epochs (int): number of training epochs per batch of trajectories (PPO typically uses 3-10)
        num_trajectories (int): number of complete trajectories to collect before training
        update_model_interval (int): number of training steps between model broadcasts
        update_buffer_interval (int): number of trajectories to collect before starting training
        sleep_between_buffer_retrieval_attempts (float): sleep time when waiting for new trajectories
        profiling (bool): if True, run_epoch will be profiled
        agent_scheduler (callable): if not None, must be of the form f(Agent, epoch)
        device (str): device on which the model will live
        clear_buffer_after_training (bool): if True, clear buffer after each training round (maintains on-policy)
        min_trajectories_before_training (int): minimum number of trajectories before starting training
    """
    env_cls: type = None
    memory_cls: type = None
    training_agent_cls: type = None
    epochs: int = 10
    rounds: int = 50
    update_epochs: int = 4  # PPO-specific: number of epochs to train on each batch
    num_trajectories: int = 10  # number of trajectories to collect before training
    update_model_interval: int = 1  # update after each training round for on-policy
    update_buffer_interval: int = 1  # check for new trajectories frequently
    sleep_between_buffer_retrieval_attempts: float = 0.5
    profiling: bool = False
    agent_scheduler: callable = None
    device: str = None
    clear_buffer_after_training: bool = True  # important for on-policy algorithms
    min_trajectories_before_training: int = 5

    total_updates = 0
    total_trajectories = 0

    def __post_init__(self):
        device = self.device
        self.epoch = 0
        self.memory = self.memory_cls(device=device)

        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space

        self.agent = self.training_agent_cls(
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )

        self.total_trajectories = len(self.memory)
        logging.info(f" Initial total_trajectories: {self.total_trajectories}")

    def update_buffer(self, interface):
        """
        Retrieve new trajectories from the interface and append to memory.

        Args:
            interface: The communication interface to retrieve buffers from workers
        """
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)

        # Count number of complete trajectories (episodes)
        # This depends on your buffer implementation
        new_trajectories = len(buffer)  # Assuming buffer tracks episode count
        self.total_trajectories += new_trajectories
        logging.debug(f" Retrieved {new_trajectories} new trajectories. Total: {self.total_trajectories}")

    def wait_for_trajectories(self, interface, min_trajectories):
        """
        Wait until we have enough trajectories to start training.

        Args:
            interface: The communication interface
            min_trajectories (int): Minimum number of trajectories needed
        """
        current_trajectories = len(self.memory)

        if current_trajectories < min_trajectories:
            logging.info(f" Waiting for trajectories ({current_trajectories}/{min_trajectories})")

            while current_trajectories < min_trajectories:
                self.update_buffer(interface)
                current_trajectories = len(self.memory)

                if current_trajectories < min_trajectories:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)

            logging.info(f" Sufficient trajectories collected ({current_trajectories})")

    def run_epoch(self, interface):
        """
        Run one training epoch with multiple rounds.

        Each round:
        1. Collect fresh trajectories from workers
        2. Train for multiple epochs on this batch
        3. Clear buffer to maintain on-policy constraint
        4. Broadcast updated policy to workers

        Args:
            interface: The communication interface for worker coordination

        Returns:
            list: Statistics for this epoch
        """
        stats = []

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(
                f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') +
                f" round {rnd}/{self.rounds} ".ljust(50, '=')
            )
            logging.debug(f"(Training): current buffer size: {len(self.memory)} trajectories")

            stats_training = []
            t0 = time.time()

            # Wait for sufficient trajectories
            self.wait_for_trajectories(interface, self.min_trajectories_before_training)

            # Optionally collect more trajectories
            for _ in range(self.num_trajectories - len(self.memory)):
                self.update_buffer(interface)
                time.sleep(0.1)  # Small delay between requests

            t1 = time.time()

            if self.profiling:
                from pyinstrument import Profiler
                pro = Profiler()
                pro.start()

            t2 = time.time()

            # PPO-style training: multiple epochs over the same batch of trajectories
            logging.info(f" Training for {self.update_epochs} epochs on {len(self.memory)} trajectories")

            for epoch_idx in range(self.update_epochs):
                logging.debug(f"  Training epoch {epoch_idx + 1}/{self.update_epochs}")

                # Iterate over batches from the trajectory buffer
                epoch_stats = []
                for batch_idx, batch in enumerate(self.memory):
                    t_train_start = time.time()

                    # Train on this batch
                    stats_training_dict = self.agent.train(batch)

                    t_train_end = time.time()

                    # Add metadata
                    stats_training_dict["training_step_duration"] = t_train_end - t_train_start
                    stats_training_dict["epoch_idx"] = epoch_idx
                    stats_training_dict["batch_idx"] = batch_idx
                    epoch_stats.append(stats_training_dict)

                    self.total_updates += 1

                # Average stats for this epoch
                if epoch_stats:
                    stats_training += pandas_dict(**DataFrame(epoch_stats).mean(skipna=True)),

            # Add episode statistics from memory
            if stats_training:
                stats_training[-1]["return_test"] = self.memory.stat_test_return
                stats_training[-1]["return_train"] = self.memory.stat_train_return
                stats_training[-1]["episode_length_test"] = self.memory.stat_test_steps
                stats_training[-1]["episode_length_train"] = self.memory.stat_train_steps

            t3 = time.time()

            # Broadcast updated policy to workers (important for on-policy!)
            interface.broadcast_model(self.agent.get_actor())
            logging.debug(f" Policy broadcasted to workers after round {rnd}")

            # Clear buffer to maintain on-policy constraint
            if self.clear_buffer_after_training:
                self.memory.clear()
                logging.debug(f" Buffer cleared (on-policy constraint)")

            round_time = t3 - t0
            collection_time = t1 - t0
            train_time = t3 - t2

            logging.debug(
                f"round_time: {round_time:.2f}s, "
                f"collection_time: {collection_time:.2f}s, "
                f"train_time: {train_time:.2f}s"
            )

            stats += pandas_dict(
                memory_len=len(self.memory),
                round_time=round_time,
                collection_time=collection_time,
                train_time=train_time,
                **DataFrame(stats_training).mean(skipna=True)
            ),

            logging.info(stats[-1].add_prefix("  ").to_string() + '\n')

            if self.profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        self.epoch += 1
        return stats


class TorchTrainingOnline(TrainingOnline):
    """
    TrainingOnline for trainers based on PyTorch.

    This class implements automatic device selection with PyTorch.
    """
    def __init__(self,
                 env_cls: type = None,
                 memory_cls: type = None,
                 training_agent_cls: type = None,
                 epochs: int = 10,
                 rounds: int = 50,
                 update_epochs: int = 4,
                 num_trajectories: int = 10,
                 update_model_interval: int = 1,
                 update_buffer_interval: int = 1,
                 sleep_between_buffer_retrieval_attempts: float = 0.5,
                 profiling: bool = False,
                 agent_scheduler: callable = None,
                 device: str = None,
                 clear_buffer_after_training: bool = True,
                 min_trajectories_before_training: int = 5):
        """
        Same arguments as `TrainingOnline`, but when `device` is `None` it is
        selected automatically for torch.

        Args:
            env_cls (type): class of a dummy environment
            memory_cls (type): class of the trajectory buffer
            training_agent_cls (type): class of the training agent (PPO)
            epochs (int): total number of epochs
            rounds (int): number of rounds per epoch
            update_epochs (int): number of PPO epochs per batch
            num_trajectories (int): trajectories to collect before training
            update_model_interval (int): model broadcast frequency
            update_buffer_interval (int): buffer retrieval frequency
            sleep_between_buffer_retrieval_attempts (float): sleep time when waiting
            profiling (bool): enable profiling
            agent_scheduler (callable): optional learning rate scheduler
            device (str): device for training (None for automatic)
            clear_buffer_after_training (bool): clear buffer after training
            min_trajectories_before_training (int): minimum trajectories needed
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            env_cls,
            memory_cls,
            training_agent_cls,
            epochs,
            rounds,
            update_epochs,
            num_trajectories,
            update_model_interval,
            update_buffer_interval,
            sleep_between_buffer_retrieval_attempts,
            profiling,
            agent_scheduler,
            device,
            clear_buffer_after_training,
            min_trajectories_before_training
        )
