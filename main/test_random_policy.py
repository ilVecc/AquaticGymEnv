import gym
import gym_aqua

from policies.random_policy import RandomPolicy


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go

    with gym.make('AquaSmallContinuousEnv-v0') as env:
        policy = RandomPolicy(env)
        policy.test()
