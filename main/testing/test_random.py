import gym

from impl.utils import aqua_term_string
from testing import Policy


class RandomAquaPolicy(Policy):
    
    def __init__(self, strategy, env_action_space):
        
        self.actions = env_action_space
        
        if strategy == 0:
            self.strategy = self._sequence_0
        elif strategy == 1:
            self.strategy = self._sequence_1
        self.time = 0
    
    def _sequence_0(self):
        return self.actions.sample()
    
    def _sequence_1(self):
        if self.time < 100:
            best_action = 0
        elif self.time < 200:
            best_action = 1
        elif self.time < 300:
            best_action = 3
        elif self.time < 500:
            best_action = 2
        else:
            best_action = 2
        return best_action
    
    def get_action(self, state):
        best_action = self.strategy()
        self.time += 1
        return best_action


if __name__ == '__main__':
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    env = gym.make("AquaEnv-v0", obstacles=False)
    policy = RandomAquaPolicy(strategy=2, env_action_space=env.action_space)
    tot_rew, info, tot_steps = policy.test(env)
    print(aqua_term_string(info, tot_rew, tot_steps))
