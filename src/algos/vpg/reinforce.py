import numpy as np
import torch as T


class Policy(T.nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(Policy, self).__init__()
        # define network architecture
        self.l1 = T.nn.Linear(input_dims, 128)
        self.l2 = T.nn.Linear(128, 128)
        self.l3 = T.nn.Linear(128, n_actions)

        # define optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        
        # forward pass
    def forward(self, state):
        x = T.nn.functional.tanh(self.l1(state))
        x = T.nn.functional.tanh(self.l2(x))
        x = self.l3(x)
        return x