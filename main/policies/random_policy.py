import numpy as np

from policies.basic_policy import AquaPolicy


class RandomAquaPolicy(AquaPolicy):
    
    def __init__(self, strategy, with_obstacles, with_waves):
        
        if strategy == 0:
            self.strategy = self._sequence_0
        elif strategy == 1:
            self.strategy = self._sequence_1
        elif strategy == 2:
            self.strategy = self._sequence_2
        elif strategy == 3:
            self.strategy = self._sequence_3
        
        is_discrete = strategy == 2
        super().__init__(env_size='small', is_discrete=is_discrete,
                         with_obstacles=with_obstacles, with_waves=with_waves)
        self.i = 0
    
    def train(self, debug, render):
        pass
    
    def _sequence_0(self):
        return self.env.action_space.sample()

    def _sequence_1(self):
        return np.array([-0.3, 0.3])  # + np.array(np.random.randint(-5, 6, 2) / 200)
    
    def _sequence_2(self):
        if self.i < 100:
            # best_action = np.array([0.3, 0.3 - 0.3*self.i/100])
            best_action = 0
        elif self.i < 200:
            # best_action = np.array([0.3 - 0.3*(self.i-100)/100, 0.0])
            best_action = 1
        elif self.i < 300:
            # best_action = np.array([0.0, 0.0]) + 0.3*(self.i-200)/100
            best_action = 3
        elif self.i < 500:
            # best_action = np.array([0.3, 0.3]) - 0.6*(self.i-300)/200
            best_action = 2
        else:
            best_action = 2
        return best_action
    
    def _sequence_3(self):
        return np.array([0, 0])

    def get_action(self, state):
        best_action = self.strategy()
        self.i += 1
        return best_action
