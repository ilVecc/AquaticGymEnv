import os

from policies.dqn_policy import DQNAquaPolicy
from dqn_plotter import plot_results
from tests.test_setup import *
from utils import assure_exists

if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    assure_exists(policy_savepath)
    
    # train network
    policy = DQNAquaPolicy(policy_savepath, load_network=policy_load, params=env_params)
    policy.train(debug=policy_debug, render=env_render)

    policy_id, run_id = tuple(os.path.split(policy_savepath))
    
    plot_results(policy_id, run_selector=run_id, colors=["red"], labels=[policy_savepath])
