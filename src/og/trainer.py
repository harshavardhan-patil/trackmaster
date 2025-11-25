from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from src.actor import TrackmasterActorCritic
import torch

class PPOTrainingAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))
    def __init__(self, observation_space=None, action_space=None, device=None,
                 model_cls=TrackmasterActorCritic):
        super().__init__(observation_space, action_space, device)

        self.model=model_cls(observation_space, action_space).to(self.device)
    
    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        obs_init, action, reward, obs_final, is_terminal, _ = batch

        with torch.no_grad():
            A_t = self.model.actor.forward(obs_init)
            Q_t = self.model.critic.forward(obs_init + A_t)
            
