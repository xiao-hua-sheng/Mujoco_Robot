import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """策略网络（Actor）"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: list):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.Tanh(),
            nn.Linear(hidden_dim[2], output_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, input_dim: int, hidden_dim: list):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.Tanh(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.Tanh(),
            nn.Linear(hidden_dim[2], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent(nn.Module):
    """PPO智能体"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: dict):
        super().__init__()
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim["policy_hdim"])
        self.value = ValueNetwork(state_dim, hidden_dim["v_hdim"])

    def get_action(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        state_tensor = torch.FloatTensor(state)
        mean, std = self.policy(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach(), log_prob.detach()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.policy(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value(states)
        return log_probs, values, entropy

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)