from policies.optimal_policy import OptimalAquaPolicy

if __name__ == '__main__':
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    policy = OptimalAquaPolicy(with_waves=True)
    policy.test()
