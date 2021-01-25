import math
from typing import Union

import gym
import numpy as np
from gym import spaces


class AquaEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False
    
    def __init__(self, obstacles: Union[bool, list] = False, waves=True, random_boat=True, random_goal=True):
        
        self.has_waves = int(waves)
        self.random_boat = random_boat
        self.random_goal = random_goal
        
        # WORLD CONSTANTS
        self.world_size = 100
        self.motor_min_thrust = +0.2
        self.motor_max_thrust = +0.5
        self.wave_min_speed = -0.05 * self.has_waves
        self.wave_max_speed = +0.05 * self.has_waves
        self.wave_speed_variance = 0.001 * self.has_waves
        
        # ENV SPACES
        if self.continuous:
            # vL, vR
            self.action_space = spaces.Box(self.motor_min_thrust, self.motor_max_thrust,
                                           shape=[2], dtype=np.float64)
        else:
            self.actions = [
                # reverse
                # (self.motor_min_thrust, self.motor_min_thrust),
                # rotate left
                (self.motor_min_thrust, self.motor_max_thrust),
                # rotate right
                (self.motor_max_thrust, self.motor_min_thrust),
                # full throttle
                (self.motor_max_thrust, self.motor_max_thrust)
            ]
            self.action_space = spaces.Discrete(len(self.actions))
        
        # x, y, angle
        self.observation_space = spaces.Box(
            np.array([0, 0, -np.pi, 0, 0]),
            np.array([self.world_size, self.world_size, np.pi, self.world_size, self.world_size]),
            dtype=np.float64)
        
        # vx, vy
        self.wave_space = spaces.Box(self.wave_min_speed, self.wave_max_speed,
                                     shape=[2], dtype=np.float64)
        
        # OBSTACLES AND GOAL SETUP
        if isinstance(obstacles, list):
            self.obstacles = obstacles
        elif obstacles:
            self.obstacles = [
                # x    y  type  radius/(width,height)
                (np.array([15, 75]), "c", 5),
                (np.array([20, 35]), "c", 10),
                (np.array([65, 85]), "r", (5, 5)),
                (np.array([85, 20]), "c", 10),
                (np.array([85, 75]), "c", 5)
            ]
        else:
            self.obstacles = []
        
        # internal state
        self._goal_state = None
        self._goal_radius = 2.5
        self._boat_prev_state = None
        self._boat_state = None
        self._boat_radius = 2.5
        self._axle_length = self._boat_radius  # absolute distance between the two motors
        self._icc = np.zeros(2, dtype=np.float64)
        self._thrust_left = None
        self._thrust_right = None
        self._thrust_total = None
        self._thrust_epsilon = 1e-8
        self.time = None
        self._wave_speed = None
        
        # reward
        self.reward_goal = lambda: +10
        self.reward_max_steps = lambda: -10
        self.reward_collision = lambda: -10
        self.reward_step = \
            lambda: (self._distance_boat_prev_from_goal() - self._distance_boat_from_goal()) * 0.7
        self.time_limit = 1000
        
        # rendering
        self.tau = 1  # time instant for numerical computation
        self.view_scale = 5.0  # window scaling (world is 100x100 pixels)
        self._viewer = None
        self._viewer_objs_holder = {}
        self._vectors_length = 8 * self.view_scale
    
    def reset(self):
        # set goal
        if self.random_goal:
            self._goal_state = self.observation_space.sample()[3:]
            while self._goal_collided_border() or self._goal_collided_obstacle():
                self._goal_state = self.observation_space.sample()[3:]
        else:
            self._goal_state = np.array([25, 80], dtype=np.float64)
        
        # set state
        if self.random_boat:
            self._boat_state = self.observation_space.sample()[:3]
            while self._boat_reached_goal() \
                    or self._boat_collided_border() \
                    or self._boat_collided_obstacle():
                self._boat_state = self.observation_space.sample()[:3]
        else:
            self._boat_state = np.array([85, 45, 0], dtype=np.float64)
        self._boat_prev_state = None
        
        # set velocities
        self._thrust_left = 0.0
        self._thrust_right = 0.0
        self._thrust_total = 0.0
        self._wave_speed = self.wave_space.sample()
        self.time = 0
        return np.concatenate((self._boat_state, self._goal_state))
    
    def normalize_angle(self, value):
        range_start, range_end = self.observation_space.low[2], self.observation_space.high[2]
        width = range_end - range_start
        offset_value = value - range_start
        # basically angle <- angle - 2pi * floor((angle + pi) / 2pi)
        return (offset_value - (math.floor(offset_value / width) * width)) + range_start
    
    def step(self, action):
        # ensure ndarray is float
        if isinstance(action, np.ndarray):
            action = action.astype(np.float64)
        
        self._boat_prev_state = self._boat_state.copy()
        self.time += 1
        
        # get thrust values of the Left and Right motor
        if self.continuous:
            if not self.action_space.contains(action):
                print("input {0!r} provided is out of bounds and has been normalized".format(
                    action, type(action)))
                action = np.clip(action,
                                 self.motor_min_thrust,
                                 self.motor_max_thrust).astype(np.float64)
            self._thrust_left, self._thrust_right = action[0], action[1]
        else:
            # noinspection PyTypeChecker
            self._thrust_left, self._thrust_right = self.actions[action]
        
        # we just want a rotation motion, so linear is handled using an epsilon sentinel:
        # enforce the minimum epsilon to avoid the quasi-linear situation and treat it as angular;
        # keep the same sign thou, since it's important for the rotation calculation
        thrust_diff = self._thrust_right - self._thrust_left
        thrust_diff = math.copysign(max(abs(thrust_diff), self._thrust_epsilon), thrust_diff)
        
        ####
        # MOVE BOAT
        ####
        # signed distance between (x,y) and ICC [pix]
        r = self._axle_length / 2 * (self._thrust_right + self._thrust_left) / thrust_diff
        # angular speed [rad/s]
        w = thrust_diff / self._axle_length
        # total tangential speed of the boat
        self._thrust_total = w * r
        
        # Instantaneous Center of Curvature (point around which (x,y) rotates)
        angle = np.pi / 2 + self._boat_state[2]
        self._icc = self._boat_state[0:2] + r * np.array([-np.sin(angle), np.cos(angle)])
        # differential drive direct kinematics
        c, s = np.cos(w * self.tau), np.sin(w * self.tau)
        rot = np.array([[c, -s], [s, c]])
        
        # compute movement and add waves contribution
        self._boat_state[0:2] = rot.dot(self._boat_state[0:2] - self._icc) + self._icc \
                                + self._wave_speed * self.tau
        # compute angle change and normalize it in [-pi/2, pi/2] w.r.t. north
        self._boat_state[2] = self.normalize_angle(self._boat_state[2] + w * self.tau)
        
        ####
        # UPDATE WAVES
        ####
        wave_speed_diff = np.random.uniform(-self.wave_speed_variance, self.wave_speed_variance, 2)
        self._wave_speed = np.clip(self._wave_speed + wave_speed_diff,
                                   self.wave_min_speed,
                                   self.wave_max_speed)
        
        # REWARD AND GOAL
        info = {
            'Termination.collided': False,
            'Termination.time': False,
            'Termination.success': False,
        }
        done = True
        if self._boat_collided_obstacle() or self._boat_collided_border():
            info['Termination.collided'] = True
            reward = self.reward_collision()
        elif self.time > self.time_limit:
            info['Termination.time'] = True
            reward = self.reward_max_steps()
        elif self._boat_reached_goal():
            info['Termination.success'] = True
            reward = self.reward_goal()
        else:
            reward = self.reward_step()
            done = False
        
        return np.concatenate((self._boat_state, self._goal_state)), reward, done, info
    
    def render(self, mode='human'):
        
        boat_color = (.0, .6, .4)
        direction_color = (.4, .0, .1)
        thrust_forward_color = (.8, .1, .0)
        thrust_reverse_color = (.0, .3, .8)
        icc_color = (.4, .0, .1)
        obstacle_color = (.15, .15, .15)
        goal_color = (.0, .0, .8)
        wave_color = (.0, .5, .65)
        
        # scaling of the scene
        world_size = int(self.world_size * self.view_scale)
        goal = self._goal_state * self.view_scale
        goal_radius = self._goal_radius * self.view_scale
        boat_pos = self._boat_state[0:2] * self.view_scale
        boat_rad = self._boat_radius * self.view_scale
        axle_width = self._axle_length * self.view_scale
        thrust_left = self._thrust_left * self.view_scale
        thrust_right = self._thrust_right * self.view_scale
        icc_pos = self._icc * self.view_scale
        wave_speed = self._wave_speed * self.view_scale
        vectors_width = boat_rad / 2
        wave_width = self.view_scale
        
        if self._viewer is None:
            from gym.envs.classic_control import rendering
            # origin is in bottom left corner (thanks God)
            self._viewer = rendering.Viewer(world_size, world_size)
            
            #
            # STATIC OBJECTS
            #
            
            # DRAW OBSTACLES
            for obs_pos, obs_type, obs_dims in self.obstacles:
                obs_pos = obs_pos.astype(np.float64) * self.view_scale
                if obs_type == 'c':
                    view_obs = rendering.make_circle(obs_dims * self.view_scale)
                elif obs_type == 'r':
                    obs_dims = (obs_dims[0] * self.view_scale, obs_dims[1] * self.view_scale)
                    l, r, t, b = -obs_dims[0] / 2, obs_dims[0] / 2, obs_dims[1] / 2, -obs_dims[
                        1] / 2
                    view_obs = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                else:
                    raise Exception("Obstacle type unknown: {}".format(obs_type))
                view_obs.set_color(*obstacle_color)
                view_obs.add_attr(rendering.Transform(translation=(obs_pos[0], obs_pos[1])))
                self._viewer.add_geom(view_obs)
            
            # DRAW GOAL
            view_goal = rendering.make_circle(goal_radius)
            view_goal.set_color(*goal_color)
            self._viewer_objs_holder["goal_trans"] = rendering.Transform()
            view_goal.add_attr(self._viewer_objs_holder["goal_trans"])
            self._viewer.add_geom(view_goal)
            
            #
            # DYNAMIC OBJECTS
            #
            
            # DRAW BOAT
            view_boat = rendering.make_circle(boat_rad)
            view_boat.set_color(*boat_color)
            self._viewer_objs_holder["boat_trans"] = rendering.Transform()  # create handler
            view_boat.add_attr(self._viewer_objs_holder["boat_trans"])  # add handler to object
            self._viewer.add_geom(view_boat)  # add in viewer
            
            # DRAW THRUST LEFT
            l, r = -vectors_width / 2 - axle_width / 2, vectors_width / 2 - axle_width / 2
            t, b = self._vectors_length, 0
            view_thrust_left = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self._viewer_objs_holder["thrust_left_trans"] = rendering.Transform()
            view_thrust_left.add_attr(self._viewer_objs_holder["thrust_left_trans"])
            view_thrust_left.add_attr(self._viewer_objs_holder["boat_trans"])
            self._viewer.add_geom(view_thrust_left)
            self._viewer_objs_holder["view_thrust_left"] = view_thrust_left
            
            # DRAW THRUST RIGHT
            l, r = -vectors_width / 2 + axle_width / 2, vectors_width / 2 + axle_width / 2
            t, b = self._vectors_length, 0
            view_thrust_right = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self._viewer_objs_holder["thrust_right_trans"] = rendering.Transform()
            view_thrust_right.add_attr(self._viewer_objs_holder["thrust_right_trans"])
            view_thrust_right.add_attr(self._viewer_objs_holder["boat_trans"])
            self._viewer.add_geom(view_thrust_right)
            self._viewer_objs_holder["view_thrust_right"] = view_thrust_right
            
            # DRAW DIRECTION
            w, t, b = vectors_width / 2, boat_rad, 0
            view_direction = rendering.FilledPolygon([(-w, b), (-w, t), (w, t), (w, b)])
            view_direction.set_color(*direction_color)
            self._viewer_objs_holder["direction_trans"] = rendering.Transform()
            view_direction.add_attr(self._viewer_objs_holder["direction_trans"])
            view_direction.add_attr(self._viewer_objs_holder["boat_trans"])
            self._viewer.add_geom(view_direction)
            
            # DRAW ICC
            view_icc = rendering.make_circle(boat_rad / 4)
            view_icc.set_color(*icc_color)
            self._viewer_objs_holder["icc_trans"] = rendering.Transform()
            view_icc.add_attr(self._viewer_objs_holder["icc_trans"])
            self._viewer.add_geom(view_icc)
            
            if self.has_waves:
                # DRAW WAVE SPEED
                # arrow body
                w, t, b = wave_width / 2, 0, -self._vectors_length
                view_wave_arrow = rendering.FilledPolygon([(-w, b), (-w, t), (w, t), (w, b)])
                view_wave_arrow.set_color(*wave_color)
                self._viewer_objs_holder["wave_trans_scale"] = rendering.Transform()
                self._viewer_objs_holder["wave_trans"] = rendering.Transform(translation=(-b / 2, -b / 2))
                view_wave_arrow.add_attr(self._viewer_objs_holder["wave_trans_scale"])
                view_wave_arrow.add_attr(self._viewer_objs_holder["wave_trans"])
                self._viewer.add_geom(view_wave_arrow)
                # arrow tip
                w, t, off = wave_width * 1.5, wave_width * 1.5, t
                view_wave_tip = rendering.FilledPolygon([(-w, off), (0, t + off), (w, off)])
                view_wave_tip.set_color(*wave_color)
                view_wave_tip.add_attr(self._viewer_objs_holder["wave_trans"])
                self._viewer.add_geom(view_wave_tip)
        
        #
        # EDIT OBJECTS
        #
        self._viewer_objs_holder["goal_trans"].set_translation(*goal)
        
        # boat movement
        self._viewer_objs_holder["boat_trans"].set_translation(*boat_pos)
        self._viewer_objs_holder["boat_trans"].set_rotation(self._boat_state[2])
        
        # thrust left magnitude
        self._viewer_objs_holder["thrust_left_trans"].set_scale(1, thrust_left)
        color = thrust_forward_color if thrust_left > 0 else thrust_reverse_color
        self._viewer_objs_holder["view_thrust_left"].set_color(*color)
        
        # thrust right magnitude
        self._viewer_objs_holder["thrust_right_trans"].set_scale(1, thrust_right)
        color = thrust_forward_color if thrust_right > 0 else thrust_reverse_color
        self._viewer_objs_holder["view_thrust_right"].set_color(*color)
        
        # ICC movement
        self._viewer_objs_holder["icc_trans"].set_translation(*icc_pos)
        
        # wave perturbation
        if self.has_waves:
            wave_arrow_angle = np.arctan2(*wave_speed[::-1]) - np.pi / 2
            self._viewer_objs_holder["wave_trans_scale"].set_scale(1, np.linalg.norm(wave_speed))
            self._viewer_objs_holder["wave_trans"].set_rotation(wave_arrow_angle)
        
        return self._viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
            self._viewer_objs_holder.clear()
    
    @staticmethod
    def _distance_circle_circle(circ1_pos, circ1_rad, circ2_pos, circ2_rad):
        centers_dist = np.linalg.norm(circ1_pos - circ2_pos)
        total_radius_length = circ1_rad + circ2_rad
        return centers_dist - total_radius_length
    
    @staticmethod
    def _distance_rectangle_circle(rect_pos, rect_dims, circ_pos, circ_rad):
        rect_l = rect_pos[0] - rect_dims[0] / 2
        rect_r = rect_pos[0] + rect_dims[0] / 2
        rect_b = rect_pos[1] - rect_dims[1] / 2
        rect_t = rect_pos[1] + rect_dims[1] / 2
        # find nearest point of rectangle border from center of circle
        min_values = np.array([rect_l, rect_b])
        max_values = np.array([rect_r, rect_t])
        dist = circ_pos - np.clip(circ_pos, min_values, max_values)
        # return actual distance between circle border and rectangle border
        return np.linalg.norm(dist) - circ_rad
    
    def _distance_state_from_goal(self, state):
        return self._distance_circle_circle(
            self._goal_state, self._goal_radius,
            state[0:2], self._boat_radius)
    
    # BOAT DISTANCES
    def _distance_boat_prev_from_goal(self):
        return self._distance_state_from_goal(self._boat_prev_state)
    
    def _distance_boat_from_goal(self):
        return self._distance_state_from_goal(self._boat_state)
    
    def _distance_boat_from_nearest_border(self):
        bl = np.min((self._boat_state[0:2] - self._boat_radius) - self.observation_space.low[0:2])
        tr = np.min(self.observation_space.high[0:2] - (self._boat_state[0:2] + self._boat_radius))
        return min(bl, tr)
    
    def _distance_boat_from_nearest_obstacle(self):
        min_dist = +math.inf
        for obs_pos, obs_type, obs_dims in self.obstacles:
            if obs_type == 'c':
                distance_func = self._distance_circle_circle
            else:
                distance_func = self._distance_rectangle_circle
            dist = distance_func(obs_pos, obs_dims, self._boat_state[0:2], self._boat_radius)
            min_dist = min(dist, min_dist)
        return min_dist
    
    # BOAT CONDITIONS
    def _boat_reached_goal(self):
        return self._distance_boat_from_goal() <= 0
    
    def _boat_collided_border(self):
        # optimized version of _distance_boat_from_nearest_border
        return np.any(self._boat_state[0:2] - self._boat_radius < self.observation_space.low[0:2]) \
               or np.any(self._boat_state[0:2] + self._boat_radius > self.observation_space.high[0:2])
    
    def _boat_collided_obstacle(self):
        # optimized version of _distance_boat_from_nearest_obstacle
        for obs_pos, obs_type, obs_dims in self.obstacles:
            if obs_type == 'c':
                distance_func = self._distance_circle_circle
            else:
                distance_func = self._distance_rectangle_circle
            dist = distance_func(obs_pos, obs_dims, self._boat_state[0:2], self._boat_radius)
            if dist <= 0:
                return True
        return False
    
    # GOAL CONDITIONS
    def _goal_collided_obstacle(self):
        for obs_pos, obs_type, obs_dims in self.obstacles:
            if obs_type == 'c':
                distance_func = self._distance_circle_circle
            else:
                distance_func = self._distance_rectangle_circle
            dist = distance_func(obs_pos, obs_dims, self._goal_state, self._goal_radius)
            if dist <= 0:
                return True
        return False
    
    def _goal_collided_border(self):
        return np.any(self._goal_state - self._goal_radius < self.observation_space.low[0:2]) \
               or np.any(self._goal_state + self._goal_radius > self.observation_space.high[0:2])


class AquaContinuousEnv(AquaEnv):
    continuous = True
