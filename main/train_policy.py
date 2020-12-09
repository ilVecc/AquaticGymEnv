import json
import os

import gym
import gym_aqua

from policies.dqn_policy import DQNPolicy
from policies.plotting import PerformanceUtils

if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    setup = {
        'env_name': 'AquaSmallEnv-v0',
        'policy_savepath': 'test_2'
    }

    # save setup
    if not os.path.exists(setup['policy_savepath']):
        os.mkdir(setup['policy_savepath'])
    with open(os.path.join(setup['policy_savepath'], "setup.json"), "w") as setup_file:
        json.dump(setup, setup_file, indent='\t')
    
    # train network
    with gym.make(setup['env_name']) as env:
        
        policy = DQNPolicy(env, setup['policy_savepath'])
        
        improve = True
        if policy.is_trained():
            while improve not in ['y', 'n']:
                improve = input("Policy already trained. Improve? [y]/n: ")
            improve = improve == 'y'
        
        if improve:
            neural_policy, episodes_score = policy.train()
            PerformanceUtils.show_recent(episodes_score, setup['policy_savepath'])
            PerformanceUtils.show_overall(episodes_score, setup['policy_savepath'])
    
    print("Training completed")
