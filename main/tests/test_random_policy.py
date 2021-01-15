from policies.random_policy import RandomAquaPolicy


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go

    params = {
        'obstacles': False,
        'waves': True,
        'random_goal': False,
        'random_boat': True
    }

    policy = RandomAquaPolicy(strategy=2, params=params)
    policy.test()
