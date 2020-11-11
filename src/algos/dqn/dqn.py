import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter

import random
import collections

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
T.manual_seed(RANDOM_SEED)


class DQN(T.nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DQN, self).__init__()
        # define network architecture
        self.l1 = T.nn.Linear(input_dims, 128)
        self.l2 = T.nn.Linear(128, 128)
        self.l3 = T.nn.Linear(128, n_actions)

        # define optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        # define loss
        self.loss_fn = T.nn.MSELoss(reduction='mean')
        
    # forward pass
    def forward(self, state):
        x = T.tanh(self.l1(state))
        x = T.tanh(self.l2(x))
        x = self.l3(x)
        return x


class Agent():
    def __init__(self, input_dims, n_actions, epsilon=1, epsilon_min=0.1, epsilon_decay_steps=1000, replay_capacity=1000, batch_size=100, lr=0.001, gamma=0.99):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon - epsilon_min)/epsilon_decay_steps
        
        self.gamma = gamma
        self.lr = lr

        self.replay_capacity = replay_capacity
        self.batch_size = batch_size

        self.transition_memory = collections.deque(maxlen=replay_capacity)
        self.qfunction = DQN(self.lr, input_dims, n_actions)
        #print(self.qfunction.layer[0].weight)
        #exit()
        self.n_actions = n_actions
        self.epoch_nb = 0
        self.writer = SummaryWriter(comment='_lr='+str(lr)+'_gamma='+str(gamma)+'_replay='+str(replay_capacity)+'_batch='+str(batch_size))

        self.evaluation_state_memory=[]
        self.evaluation_states=T.Tensor(list([]))

    def choose_action(self, observation):
        state = T.Tensor(observation)
        probabilities = T.nn.functional.softmax(self.qfunction.forward(state), dim=0)

        if random.uniform(0,100) < self.epsilon:
            action = T.tensor(random.randint(0, self.n_actions-1))
        else:
            action = T.argmax(probabilities)
        
        #decay epislon linearly from epsilon -> epsilon_min
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.epsilon_min)

        return action.item()

    def store_transitions(self, transition):
        self.transition_memory.append(transition)

    def store_evaluation_state(self, states):
        self.evaluation_states =  T.Tensor(list(states))

    def evaluate(self):
        return T.mean(T.max(self.qfunction.forward(self.evaluation_states), dim=1)[0])

    def learn(self):
        batch_size = min(self.batch_size, len(self.transition_memory))
        random_batch_idx = random.sample(range(len(self.transition_memory)), batch_size)
        transitions_array = np.array(self.transition_memory)[random_batch_idx]


        s_i = T.Tensor(list(transitions_array.T[0]))

        a_i = list(transitions_array.T[1])
        a_i_np = np.array(a_i)
        a_i_ohe = np.zeros((a_i_np.size, max(a_i_np.max(),1)+1))
        a_i_ohe[np.arange(a_i_np.size), a_i_np] = 1
        a_i = T.Tensor([a_i_ohe])
 
        r_i = T.Tensor([list(transitions_array.T[2])])

        s_ip1 = T.Tensor(list(transitions_array.T[3]))

        s_ip1_not_terminal = T.Tensor([list(transitions_array.T[4])])
        
        q_i = self.qfunction.forward(s_i)
        q_i_a_i = T.sum(T.mul(q_i, a_i), 2)
        
        q_ip1 = self.qfunction.forward(s_ip1)
        q_ip1_max,_ = T.max(q_ip1, dim=1)

        y_i = r_i + self.gamma*T.mul(q_ip1_max, s_ip1_not_terminal)
        
        loss = self.qfunction.loss_fn(q_i_a_i, y_i)


        self.epoch_nb += 1
        self.writer.add_scalar("Loss/train", loss, self.epoch_nb)
        self.writer.add_scalar("q_value/test", self.evaluate(), self.epoch_nb)

        loss.backward()

        self.qfunction.optimizer.step()
        self.qfunction.optimizer.zero_grad()
        

        return loss.item()
    
    def add_to_tb(self, metric_value, metric_name):
        self.writer.add_scalar(metric_name, metric_value, self.epoch_nb)


    def flush_tb(self):
        self.writer.flush()

    def close_tb(self):
        self.writer.close()
