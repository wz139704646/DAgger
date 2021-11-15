import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod


class Actor:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def act(self, obs):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def __call__(self, obs):
        return self.act(obs)


class CNNActor(Actor):
    """The actor using CNN as feature net and MLP as last mapping"""
    def __init__(
        self, c, h, w, action_shape, lr=1e-3,
        hidden_sizes=[512], device='cpu', **kwargs):
        super(CNNActor, self).__init__(**kwargs)

        self.device = device
        self.output_dim = int(np.prod(action_shape))
        # feature net
        self.feature_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.feature_dim = np.prod(self.feature_net(torch.zeros(1, c, h, w)).shape[1:])

        # action mapping
        hidden_sizes = [self.feature_dim] + list(hidden_sizes)
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.feature_to_hidden = nn.Sequential(*layers)
        self.hidden_to_action = nn.Linear(hidden_sizes[-1], self.output_dim)

        # optimizer
        self.optim = torch.optim.Adam(
            list(self.feature_net.parameters()) \
                + list(self.feature_to_hidden.parameters()) \
                + list(self.hidden_to_action.parameters()),
            lr=lr
        )

    def act(self, obs):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        f = self.feature_net(obs)
        h = self.feature_to_hidden(f)
        logits = self.hidden_to_action(h)

        return logits

    def save(self, filepath):
        checkpoint = {
            "feature_net": self.feature_net.state_dict(),
            "feature_to_hidden": self.feature_to_hidden.state_dict(),
            "hidden_to_action": self.hidden_to_action.state_dict(),
            "optim": self.optim.state_dict()
        }

        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)

        self.feature_net.load_state_dict(checkpoint['feature_net'])
        self.feature_to_hidden.load_state_dict(checkpoint['feature_to_hidden'])
        self.hidden_to_action.load_state_dict(checkpoint['hidden_to_action'])
        self.optim.load_state_dict(checkpoint['optim'])

    def update(self, obs, actions):
        """update with expert data
        :param obs: n x obs_shape, observations data
        :param actions: 1-D discrete (len n), actions data
        """
        imitation_logits = self(obs)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        loss = F.cross_entropy(imitation_logits, actions)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "loss": loss.item()
        }

    def train(self, mode=True):
        self.feature_net.train(mode)
        self.feature_to_hidden.train(mode)
        self.hidden_to_action.train(mode)

