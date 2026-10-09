import torch
from torch import nn


class Transition(nn.Module):
    def __init__(self, state_dim=4, action_dim=0, width=64, residual=True):
        super().__init__()
        self.action_dim, self.residual = action_dim, residual
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, state_dim),
        )

    def forward(self, state, action=None):
        inputs = torch.cat([state, action], -1) if self.action_dim else state
        delta = self.net(inputs)
        return state + delta if self.residual else delta


def rollout(model, state, steps, actions=None):
    predictions = []
    for t in range(steps):
        state = model(state, None if actions is None else actions[:, t])
        predictions.append(state)
    return torch.stack(predictions, dim=1)
