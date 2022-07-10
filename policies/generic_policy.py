from torch import nn
from functools import partial

class GenericPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def evaluate_observation(self, observation):
        raise NotImplementedError

    def evaluate_action(self, observation, action):
        raise NotImplementedError

    def get_action(self, observation):
        raise NotImplementedError

    @staticmethod
    # similar to stable baselines 3
    # inits parameters with orthogonal matrices
    def init_ortho_weights(module_gains):
        def ortho_weights(module, gain = 1):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
        for module, gain in module_gains.items():
            module.apply(partial(ortho_weights, gain=gain))
