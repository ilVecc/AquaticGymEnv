import gym

from policies.random_policy import RandomAquaPolicy


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go

    policy = RandomAquaPolicy(strategy=2, with_obstacles=False, with_waves=True)
    policy.test()
