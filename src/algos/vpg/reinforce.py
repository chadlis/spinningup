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


class Agent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []
        self.policy = Policy(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        observation = T.Tensor(observation)
        state = T.autograd.Variable(observation)
        probabilities = T.nn.functional.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        
        return action.item()


    def store_rewards(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G)
        
        #loss = - G[0] * np.sum(self.action_memory)
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        steps_nb = len(self.reward_memory)
        self.action_memory = []
        self.reward_memory = []

        return loss.item(), steps_nb
