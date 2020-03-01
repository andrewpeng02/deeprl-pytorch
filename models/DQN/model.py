import torch
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_stream = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU()
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)

        x = self.shared_stream(state)
        advantages = self.advantage_stream(x)
        value = self.value_stream(x)
        return value + advantages - torch.mean(advantages)
