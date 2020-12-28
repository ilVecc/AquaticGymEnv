from policies.dqn_policy import DQNAquaPolicy


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go

    # load the testing setup
    policy_savepath = "train_dqn_TEST"
    
    # run the test
    policy = DQNAquaPolicy(policy_savepath, load_network=False,
                           env_size='small', with_obstacles=False, with_waves=False)
    policy.test()
