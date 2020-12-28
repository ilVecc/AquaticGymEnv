import numpy as np

from policies.basic_policy import AquaPolicy


class OptimalAquaPolicy(AquaPolicy):
    
    def __init__(self, with_waves):
        super().__init__(env_size='small', is_discrete=True,
                         with_obstacles=False, with_waves=with_waves)
        self.i = 0
        self.goal = self.env.goal_state
    
    def train(self, debug, render):
        pass
    
    @staticmethod
    def _normalize_angle(angle):
        return (angle + 2*np.pi) % (2*np.pi)

    @staticmethod
    def _deg2rad(angle_deg):
        return angle_deg / 180 * np.pi
    
    def get_action(self, state):
        boat_angle_wrt_zero = self._normalize_angle(state[2] + np.pi / 2)
        goal_angle_wrt_boat = self._normalize_angle(np.arctan2(*(self.goal - state[0:2])[::-1]))
        angle_diff = goal_angle_wrt_boat - boat_angle_wrt_zero
        if self._deg2rad(8) < abs(angle_diff):
            if angle_diff > 0:
                best_action = 1
            else:
                best_action = 2
        else:
            best_action = 3
        self.i += 1
        return best_action