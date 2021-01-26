import os

import gym

# noinspection PyUnresolvedReferences
import gym_aqua
from impl.dqn import DQN
from plotting.dqn_train_plotter import ProgressPlotter
from utils import AquaStateNormalizer

if __name__ == '__main__':
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    policy_savepath = 'dqn_with_obs/new_policy'  # ! change this path
    
    os.makedirs(policy_savepath, exist_ok=True)
    
    # train network
    env = gym.make("AquaEnv-v0", obstacles=True)
    policy = DQN(env, policy_savepath, normalizer=AquaStateNormalizer(env))
    policy.train(episodes=15000, verbose=True, render=False)
    
    # plot results
    plotter = ProgressPlotter()
    plotter.load(policy_savepath, [policy_savepath], fetch=True)
    plotter.plot_training_rewards(save=True, window=500)
    plotter.plot_training_success(save=True, window=500)
