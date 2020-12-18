import json
import os

import gym
import gym_aqua

from policies.dqn_policy import DQNPolicy


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go

    # load the testing setup
    test_path = "train_dqn_4"
    with open(os.path.join(test_path, "setup.json"), 'r') as json_file:
        setup = json.load(json_file)
    
    # run the test
    with gym.make(setup['env_name']) as env:
        policy = DQNPolicy(env, setup['policy_savepath'])
        policy.test()
