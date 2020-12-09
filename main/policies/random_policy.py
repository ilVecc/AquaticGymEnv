import numpy as np

from policies.basic_policy import Policy


class RandomPolicy(Policy):
    
    def __init__(self):
        super().__init__()

    def is_trained(self):
        return True
    
    def train(self):
        pass
    
    def get_action(self, state):
        # best_action = env.action_space.sample()
        best_action = np.array([0.28, 0.3])  # + np.array(np.random.randint(-5, 6, 2) / 200)
        return best_action
