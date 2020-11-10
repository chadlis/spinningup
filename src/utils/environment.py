import gym


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



def process_obs(observation, env_name):
    if env_name == 'FrozenLake-v0':
        # (FrozenLake-v0) transform the observation from 1D to 16D OHE 
        observation_mod = np.eye(input_dims)[observation]
    else:
        observation_mod = observation

    return observation_mod
