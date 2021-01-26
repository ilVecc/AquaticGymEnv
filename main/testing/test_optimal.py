import gym
import numpy as np

from impl.utils import aqua_term_string
from testing import Policy


class OptimalAquaPolicy(Policy):
    
    @staticmethod
    def _normalize_angle(angle):
        return (angle + 2 * np.pi) % (2 * np.pi)
    
    ANGLE_THRESHOLD = 8 / 180 * np.pi
    
    def get_action(self, state):
        boat, goal = state[:2], state[3:]
        boat_angle_wrt_zero = self._normalize_angle(state[2] + np.pi / 2)
        goal_angle_wrt_boat = self._normalize_angle(np.arctan2(*(goal - boat)[::-1]))
        angle_diff = goal_angle_wrt_boat - boat_angle_wrt_zero
        if self.ANGLE_THRESHOLD < abs(angle_diff):
            if angle_diff > 0:
                best_action = 0
            else:
                best_action = 1
        else:
            best_action = 2
        return best_action


if __name__ == '__main__':
    # the optimal policy is meant to be used only on discrete, non-obstacles setups
    
    env = gym.make("AquaEnv-v0", obstacles=False)
    policy = OptimalAquaPolicy()
    tot_rew, info, tot_steps = policy.test(env)
    print(aqua_term_string(info, tot_rew, tot_steps))
