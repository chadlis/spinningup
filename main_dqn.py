import gym
import numpy as np
from src.algos.dqn.dqn import Agent
from datetime import datetime   


import numpy as np
import argparse
import logging

from src.utils import plotting
from src.utils import environment

parser = argparse.ArgumentParser(description="DQN algorithm")
parser.add_argument('--gymenv', type=str, default="CartPole-v1", help='Gym environment name')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

parser.add_argument('--epsilon', type=float, default=1, help='exploration vs exploitation epsilon-greedy factor (default: 1%)')
parser.add_argument('--epsilonmin', type=float, default=0.1, help='exploration vs exploitation epsilon-greedy min (default: 0.1%)')
parser.add_argument('--epsilondecaysteps', type=float, default=1, help='exploration vs exploitation epsilon-greedy decay steps (default: 1000)')

parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.001)')

parser.add_argument('--replaycapacity', type=int, default=1000, help='Replay buffer capacity (default: 1000)')
parser.add_argument('--batchsize', type=int, default=100, help='batch size (default: 100)')


parser.add_argument('--games', type=int, default=250, help='number of episodes/games (default: 1000)')
parser.add_argument('--render', action='store_true',  help='render the environment')
parser.add_argument('--runs', type=int, default=2, help='number of runs (default: 10)')
parser.add_argument('--avgnb', type=int, default=100, help='number of episodes to include in the running average (default: 100)')
parser.add_argument('--filenameaddtext', type=str, default='', help='additional text to add to the filename')
args = parser.parse_args()


alg_name = 'dqn'
    
if __name__ == '__main__':

    # algorithm and parameters
    env_name = args.gymenv
    seed = args.seed
    epsilon = args.epsilon
    epsilon_min = args.epsilonmin
    epsilon_decay_steps = args.epsilondecaysteps

    gamma = args.gamma
    lr = args.lr

    replay_capacity = args.replaycapacity
    batch_size = args.batchsize

    n_games = args.games
    flag_render = args.render
    n_runs = args.runs

    filename_add_text = args.filenameaddtext
    
    
    # monitoring
    avg_episodes_nb = args.avgnb
    
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d%H%M%S")

    fname = alg_name + '_' + env_name + '_' + 'lr' + str(lr) + '_' + 'gamma' + str(gamma) + '_' + str(n_games) + 'games' \
         + str('' if filename_add_text=='' else '_'+ filename_add_text)
         
    figure_file_score =  'reports/figures/' + fname + '_' + 'score' + '_' + timestampStr  + '.html'
    figure_file_loss = 'reports/figures/' + fname + '_' + 'loss' + '_' + timestampStr +  '.html'
    log_file = 'reports/logs/' + fname + '_' + timestampStr +  '.log'

    score_plotter = plotting.Plotter(running_average=True, running_avg_episodes_nb=100, title="Running avg score", x_title="episode", y_title="Running avg score", filepath=figure_file_score)
    loss_plotter = plotting.Plotter(running_average=True, running_avg_episodes_nb=100, title="Running avg loss", x_title="episode", y_title="Running avg loss", filepath=figure_file_loss)

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file,
                    filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-4s: %(levelname)-4s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)
    
    logger = logging.getLogger(__name__)

    # iterate over 10 runs
    for run_nb in range(n_runs):
        logger.info('RUN ' + str(run_nb))

        # initialize environment
        env = environment.make(env_name)

        # improve reproducibility
        env.seed(seed)
        #np.random.seed(seed)
        
        # get the dimensions of the states and actions spaces
        input_dims, n_actions = environment.get_dims(env, env_name)

        # initialize the agent with the choosen parameters
        agent = Agent(input_dims=input_dims, n_actions=n_actions, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay_steps=epsilon_decay_steps, replay_capacity=replay_capacity, batch_size=batch_size, gamma=gamma, lr=lr)
        
        # scores and losses memory array for monitoring
        scores = []
        losses = []

        # iterate over the episodes
        for i in range(n_games):
            done = False
            # reset the environment to begin the episode
            observation = env.reset()
            score = 0
            modified_score = 0
            steps_nb = 0

            while not done:
                steps_nb += 1
                if flag_render:
                    env.render()
                
                # preprocess observations if needed
                observation = environment.process_obs(observation, env_name)

                # use the agent's current policy to choose an action
                action = agent.choose_action(observation)

                # make a step in the environment and get the new observation and the reward
                observation_, reward, done, info = env.step(action)
                score += reward

                modified_score += environment.modify_reward(reward, done, env_name)
                
                # store transition in the agent's transition memory
                transition = (observation, action, reward, observation_, int(not done))
                agent.store_transitions(transition)

                # make the new observation as the current one
                observation = observation_
            
                # learn and modify the agent policy, and get the loss term and the number of steps of the episode
                loss_item = agent.learn()

            # save loss and score and compute and print running average for monitoring
            losses.append(loss_item)
            scores.append(score)
            agent.add_to_tb(score, 'score')
            avg_score = np.mean(scores[-avg_episodes_nb:])
            avg_loss = np.mean(losses[-avg_episodes_nb:])

            # debugging 
            info_to_log = 'episode ', i, 'score %.2f' % score, ' modified score %.2f' % modified_score, 'average score %.2f' % avg_score,\
                'average loss %.2f' % avg_loss, 'steps number', steps_nb
            logger.debug(info_to_log)
            steps_nb = 0
        # add score and loss running avg curves to plotter
        score_plotter.add_curve(scores)
        loss_plotter.add_curve(losses)

        agent.flush_tb()
    
    agent.close_tb()
    score_plotter.plot()
    loss_plotter.plot()
        
