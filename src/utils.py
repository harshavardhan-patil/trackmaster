import torch
import numpy as np
import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct

def obs_preprocessor(obs):
    """
    Preprocess observation.

    Args:
        obs: tuple from environment

    Returns:
        preprocessed observation
    """
    speed, gear, rpm, images, act1, act2 = obs
    speed = speed / 1000.0
    images = images.astype(np.float32) / 255.0

    return speed, gear, rpm, images, act1, act2