import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter

import random
import collections

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
T.manual_seed(RANDOM_SEED)


class DQN(T.nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DQN, self).__init__()
        # define network architecture
        self.l1 = T.nn.Linear(input_dims, 256)
        self.l2 = T.nn.Linear(256, 256)
        self.l3 = T.nn.Linear(256, n_actions)

        # define optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        # define loss
        self.loss_fn = T.nn.MSELoss()
        
        # send to GPU if availabe
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)       
    
    # forward pass
    def forward(self, state):
        x = T.tanh(self.l1(state))
        x = T.tanh(self.l2(x))
        x = self.l3(x)
        return x


class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma, replay_capacity, batch_size, ddqn=False, prioritised_replay=False, multistep=False, replace_target=1000, epsilon=1, epsilon_min=0.1, epsilon_decay_steps=1000, exp_param=''):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon - epsilon_min)/epsilon_decay_steps
        
        self.gamma = gamma
        self.lr = lr

        self.replay_capacity = replay_capacity
        self.batch_size = batch_size

        self.state_memory = np.zeros((self.replay_capacity, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.replay_capacity, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.replay_capacity, dtype=np.int32)
        self.reward_memory = np.zeros(self.replay_capacity, dtype=np.float32)
        self.terminal_memory = np.zeros(self.replay_capacity, dtype=np.bool)
        self.memory_count = 0

        self.qfct_eval = DQN(self.lr, input_dims, n_actions)
        self.qfct_next = DQN(self.lr, input_dims, n_actions) #DDQN

        self.ddqn=ddqn
        self.prioritised_replay=prioritised_replay
        self.multistep=multistep

        self.replace_target = replace_target

        self.learning_iter = 0
        self.writer = SummaryWriter(comment='_'+exp_param)

        self.evaluation_states=T.Tensor(list([])).to(self.qfct_eval.device)


    def store_transitions(self, state, action, reward, state_, done):
        idx = self.memory_count % self.replay_capacity

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.new_state_memory[idx] = state_
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.memory_count += 1
    
    def choose_action(self, state, greedy=False):
        if greedy or np.random.random() > self.epsilon:
            state = T.Tensor(state).to(self.qfct_eval.device)
            actions_values = self.qfct_eval.forward(state)
            action = T.argmax(actions_values).item()
            self.epsilon = min(self.epsilon - self.epsilon_decay_rate , self.epsilon_min)
        else:
            action = np.random.choice(self.action_space)
        return action


    def sample_transitions(self):
        memory_size = min(self.replay_capacity, self.memory_count)
        batch = np.random.choice(memory_size, self.batch_size, replace=False)
        state_batch = T.tensor(self.state_memory[batch]).to(self.qfct_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.qfct_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.qfct_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.qfct_eval.device)
        action_batch = self.action_memory[batch]
        return state_batch, new_state_batch, reward_batch, terminal_batch, action_batch
    
    def replace_target_network(self):
        if self.learning_iter % self.replace_target == 0:
            self.qfct_next.load_state_dict(self.qfct_eval.state_dict())

    def learn(self):
        if self.memory_count < self.batch_size:
            return 0

        self.qfct_eval.optimizer.zero_grad()

        self.replace_target_network()

        state_batch, new_state_batch, reward_batch, terminal_batch, action_batch = self.sample_transitions()

        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        q_eval = self.qfct_eval.forward(state_batch)
        q_pred = self.qfct_eval.forward(state_batch)[batch_indices, action_batch]

        if self.ddqn:
            q_next = self.qfct_next.forward(new_state_batch)
            q_next[terminal_batch] = 0
            max_actions = T.argmax(q_eval, dim=1)
            q_target = reward_batch + self.gamma * q_next[batch_indices, max_actions]

        else:
            q_next = self.qfct_next.forward(new_state_batch)
            q_next[terminal_batch] = 0
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.qfct_eval.loss_fn(q_target, q_pred).to(self.qfct_eval.device)
        
        self.writer.add_scalar("Loss/train", loss, self.learning_iter)
        self.writer.add_scalar("q_value/train", T.mean(q_pred), self.learning_iter)
        self.writer.add_scalar("q_value/test", self.evaluate(), self.learning_iter)
        
        loss.backward()
        self.qfct_eval.optimizer.step()
        
        self.learning_iter += 1
        return loss.item()
    
    def add_to_tb(self, metric_name, metric_value, n_episode):
        self.writer.add_scalar(metric_name, metric_value, n_episode)
        print(metric_name, metric_value, n_episode)


    def flush_tb(self):
        self.writer.flush()

    def close_tb(self):
        self.writer.close()

    def store_evaluation_state(self, states):
        self.evaluation_states =  T.Tensor(list(states)).to(self.qfct_eval.device)

    def evaluate(self):
        return T.mean(T.max(self.qfct_eval.forward(self.evaluation_states), dim=1)[0])