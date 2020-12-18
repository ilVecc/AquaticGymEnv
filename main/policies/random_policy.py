import numpy as np

from policies.basic_policy import Policy


class RandomPolicy(Policy):
    
    def __init__(self, env):
        super().__init__(env)

    def is_trained(self):
        return True
    
    def train(self, debug, render):
        pass
    
    def get_action(self, state):
        # best_action = env.action_space.sample()
        # best_action = np.array([-0.3, 0.3])  # + np.array(np.random.randint(-5, 6, 2) / 200)
        best_action = np.array([0.3, 0.1])
        return best_action
