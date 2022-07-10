import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from policies.generic_policy import GenericPolicy

class FeedForwardAgent(GenericPolicy):
    """
    simple feed forward model for cart-pole
    outputs actions as 2 dimensional vector for probability of moving left or right
    """
    def __init__(self, input_size, ortho_init=True, device='cuda'):
        super().__init__()
        self.device = device
        self.action_embed = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.value_embed = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(64, 2)
        self.value_net = nn.Linear(64, 1)

        if ortho_init:
            module_gains = {
                self.action_embed: np.sqrt(2),
                self.value_embed: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
            }
            self.init_ortho_weights(module_gains)

    def forward(self, x):
        return self.action_embed(x), self.value_embed(x)

    def evaluate_observation(self, observation):
        action_embed, value_embed = self.forward(observation)
        action_logit = self.action_net(action_embed)
        action_prob = Categorical(torch.nn.functional.softmax(action_logit, dim=1))
        action = action_prob.sample()
        log_prob = action_prob.log_prob(action)
        state_value = self.value_net(value_embed).flatten()

        return action.to("cpu").numpy(), log_prob, state_value

    def evaluate_action(self, observation, action):
        act_embed, val_embed = self.forward(observation)
        action_logit = self.action_net(act_embed)
        action_prob = Categorical(torch.nn.functional.softmax(action_logit, dim=1))
        entropy = action_prob.entropy()
        log_prob = action_prob.log_prob(action)
        state_value = self.value_net(val_embed).flatten()

        return log_prob, state_value, entropy

    def get_action(self, observation):
        observation = torch.from_numpy(observation).to(self.device).float()
        act_embed, _ = self.forward(observation.unsqueeze(0))
        action_logit = self.action_net(act_embed)
        action_mode = torch.max(torch.nn.functional.softmax(action_logit, dim=1), dim=1)
        action = action_mode.indices

        return action.to("cpu").numpy()[0]
