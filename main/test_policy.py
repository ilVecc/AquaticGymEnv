import json
import os

import gym
import gym_aqua

from policies.dqn_policy import DQNPolicy
from policies.random_policy import RandomPolicy


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go

    test_path = "test_2"

    with open(os.path.join(test_path, "setup.json"), 'r') as json_file:
        setup = json.load(json_file)
    
    with gym.make(setup['env_name']) as env:
        
        policy = DQNPolicy(env, setup['policy_savepath'])
        # policy = RandomPolicy()
        
        policy.test()
