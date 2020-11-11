import gym
import random
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def modify_reward(reward, done, env_name):
    if env_name == 'FrozenLake-v0': 
        # (FrozenLake-v0) modify the reward to encourage short paths and discourage falls
        if reward == 0:
            reward_mod = -0.01
        if done:
            if reward < 1:
                reward_mod = -1
            else:
                reward_mod = 10
    else:
        reward_mod = reward
    
    return reward_mod
    

def make(env_name):
    if env_name == 'FrozenLake-v0':
        env = gym.make(env_name, is_slippery=False)
    else:
        env = gym.make(env_name)
    
    return env


def get_dims(env, env_name): 
    if env_name =='CartPole-v1':
        input_dims = 4
    else:
        input_dims = env.observation_space.n

    n_actions = env.action_space.n

    return input_dims, n_actions



def process_obs(observation, input_dims, env_name):
    if env_name == 'FrozenLake-v0':
        # (FrozenLake-v0) transform the observation from 1D to 16D OHE 
        observation_mod = np.eye(input_dims)[observation]
    else:
        observation_mod = observation

    return observation_mod


def collect_evaluation_states(env_name, nb_states=1000):
    env = make(env_name)
    env.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    # get the dimensions of the states and actions spaces
    input_dims, _ = get_dims(env, env_name)

    # iterate over the episodes to collect evaluation states with a random policy
    eval_states = []
    n_iter = 0
    while n_iter<nb_states*10:
        done = False
        # reset the environment to begin the episode
        observation = env.reset()
        while ((not done) and n_iter<nb_states*10):
            # preprocess observations if needed
            observation = process_obs(observation, input_dims, env_name)

            # use a random policy to choose an action
            action = env.action_space.sample()

            # make a step in the environment and get the new observation and the reward
            observation_, _, done, _ = env.step(action)

            # store evaluation state
            eval_states.append(observation)

            # make the new observation as the current one
            observation = observation_
 
            n_iter += 1

    random_states_idx = random.sample(range(nb_states*10), nb_states)
    eval_states = np.array(eval_states)[random_states_idx]
    return eval_states

if __name__ == '__main__':
    pass
    # env_name = 'CartPole-v1'
    # eval_states = collect_evaluation_states(env_name, nb_states=10)
    # print('nb: ', len(eval_states))
    # print(eval_states)
