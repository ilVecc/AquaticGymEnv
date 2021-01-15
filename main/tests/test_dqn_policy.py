from policies.dqn_policy import DQNAquaPolicy
from test_setup import policy_savepath, env_params

if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    # run the test
    policy = DQNAquaPolicy(policy_savepath, load_network=True, params=env_params)
    policy.test()
