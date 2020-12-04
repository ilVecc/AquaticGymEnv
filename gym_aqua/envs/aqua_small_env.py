import math

import gym
import numpy as np
from gym import spaces


class AquaSmallEnv(gym.Env):
    """
    World size: [0, 100] x [0, 100] pix^2
    Motor speed:   [  -0.3,   +0.3] pix/s
    Waves speed:   [ -0.05,  +0.05] pix/s
    Waves speed variance:     0.001 pix/s
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, with_obstacles):
        
        # world constants
        self.world_size = 100
        self.motor_min_thrust = -0.3
        self.motor_max_thrust = +0.3
        self.wave_min_speed = -0.05
        self.wave_max_speed = +0.05
        self.wave_speed_variance = 0.001
        
        # env spaces
        self.action_space = spaces.Box(self.motor_min_thrust,
                                       self.motor_max_thrust,
                                       shape=[2], dtype=np.float64)  # vL, vR
        self.observation_space = spaces.Box(np.array([0, 0, -np.pi]),
                                            np.array([self.world_size, self.world_size, np.pi]),
                                            dtype=np.float64)  # x, y, angle
        self.wave_space = spaces.Box(self.wave_min_speed,
                                     self.wave_max_speed,
                                     shape=[2], dtype=np.float64)  # vx, vy
        
        # obstacles and goal
        if with_obstacles:
            self.obstacles = [
                # x    y  type  radius/(width,heigth)
                (15, 75, "c", 5),
                (20, 35, "c", 10),
                (35, 55, "r", (5, 5)),
                (45, 20, "r", (10, 10)),
                (55, 70, "c", 10),
                (65, 35, "c", 5),
                (65, 85, "r", (5, 5)),
                (80, 55, "c", 10),
                (85, 20, "c", 10),
                (85, 75, "c", 5)
            ]
        else:
            self.obstacles = []
        self.goal_state = np.array([15, 65], dtype=np.float64)
        self.goal_radius = 2.5
        
        # world dynamics
        self.wave_speed = np.zeros(2, dtype=np.float64)
        
        # internal state
        self.boat_radius = 2.5
        self.position = np.array([55, 30], dtype=np.float64)
        self.angle = None
        self.thrust_left = None
        self.thrust_right = None
        self.thrust_total = None
        self.time = None
        
        # reward
        self.reward_goal = lambda: +100
        self.reward_max_steps = lambda: -100
        self.reward_collision = lambda: -100
        self.reward_step = lambda: -0.1
        self.time_limit = 1000
        
        # rendering
        self.scale = 5.0
        self.tau = 1  # time instant for numerical computation
        self.vectors_length_multiplier = 40
        self.axle_length = self.boat_radius  # absolute distance between the two motors
        self.ICC = np.zeros(2, dtype=np.float64)
        self.viewer = None
    
    def reset(self):
        self.position = np.array([55, 30], dtype=np.float64)
        self.angle = 0.0
        self.thrust_left = 0.0
        self.thrust_right = 0.0
        self.thrust_total = 0.0
        self.wave_speed = self.wave_space.sample()
        self.time = 0
        state = np.array([self.position[0], self.position[1], self.angle])
        return state
    
    @staticmethod
    def normalize_angle(value, range_start, range_end):
        width = range_end - range_start
        offset_value = value - range_start
        return (offset_value - (math.floor(offset_value / width) * width)) + range_start
    
    def step(self, action: np.ndarray):
        action = np.array(action, dtype=np.float64)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        # MOVE BOAT
        # linear speed values of the Left and Right motor
        self.thrust_left = action[0]
        self.thrust_right = action[1]
        
        # direction
        direction = np.array([np.cos(np.pi / 2 + self.angle), np.sin(np.pi / 2 + self.angle)])
        
        if self.thrust_left != self.thrust_right:
            # ROTATION MOTION
            # signed distance between (x,y) and ICC
            R = self.axle_length / 2 * (self.thrust_right + self.thrust_left) / \
                (self.thrust_right - self.thrust_left)
            # angular speed
            w = (self.thrust_right - self.thrust_left) / self.axle_length
            # total tangential speed of the boat
            self.thrust_total = w * R
            
            # Instantaneous Center of Curvature
            # (point around which (x,y) rotates)
            self.ICC = self.position + R * np.array(
                [-np.sin(np.pi / 2 + self.angle), np.cos(np.pi / 2 + self.angle)])
            
            # differential drive direct kinematics
            c, s = np.cos(w * self.tau), np.sin(w * self.tau)
            rot = np.array([[c, -s], [s, c]])
            self.position = rot.dot(self.position - self.ICC) + self.ICC
            self.angle += w * self.tau
            # basically angle <- angle - 2pi * floor((angle + pi) / 2pi)
            self.angle = self.normalize_angle(self.angle,
                                              self.observation_space.low[2],
                                              self.observation_space.high[2])
        else:
            # LINEAR MOTION
            # self.angle remains the same
            self.thrust_total = self.thrust_right + self.thrust_left
            self.position += self.thrust_total * direction * self.tau
            self.ICC = self.position
        
        # add waves contribution
        self.position += self.wave_speed * self.tau
        
        # UPDATE WAVES
        diff = np.random.uniform(-self.wave_speed_variance, self.wave_speed_variance, 2)
        new_wave_speed = self.wave_speed + diff
        new_wave_speed[new_wave_speed > self.wave_max_speed] = self.wave_max_speed
        new_wave_speed[new_wave_speed < self.wave_min_speed] = self.wave_min_speed
        self.wave_speed = new_wave_speed
        
        # REWARD AND GOAL
        info = {
            'Termination.collided': False,
            'Termination.stuck': False,
            'Termination.time': False,
        }
        done = False
        if self._has_collided_obstacle():
            info['Termination.collided'] = True
            reward = self.reward_collision()
            done = True
        elif self._has_collided_border():
            info['Termination.collided'] = True
            reward = self.reward_collision()
            done = True
        elif self._has_reached_goal():
            reward = self.reward_goal()
            done = True
        elif self.time > self.time_limit:
            info['Termination.time'] = True
            reward = self.reward_max_steps()
            done = True
        else:
            reward = self.reward_step()
        
        self.time += 1
        
        return np.array([self.position[0], self.position[1], self.angle]), reward, done, info
    
    def render(self, mode='human'):
        
        boat_color = (.0, .6, .4)
        direction_color = (.4, .0, .1)
        thrust_straight_color = (.8, .0, .0)
        thrust_reverse_color = (.0, .3, .8)
        icc_color = (.4, .0, .1)
        obstacle_color = (.15, .15, .15)
        goal_color = (.0, .0, .8)
        wave_color = (.0, .5, .65)
        
        # scaling of the scene
        world_size = int(self.world_size * self.scale)
        position = self.position * self.scale
        boat_radius = self.boat_radius * self.scale
        axle_width = self.axle_length * self.scale
        thrust_left = self.thrust_left * self.scale
        thrust_right = self.thrust_right * self.scale
        icc = self.ICC * self.scale
        wave_speed = self.wave_speed * self.scale
        
        # standard sizes in the scene
        vec_width = boat_radius / 2
        thrust_left_length = self.vectors_length_multiplier * abs(thrust_left)
        thrust_right_length = self.vectors_length_multiplier * abs(thrust_right)
        wave_width = self.scale
        wave_length = self.vectors_length_multiplier * np.linalg.norm(wave_speed)
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            # origin is in bottom left corner (thank God)
            screen_width = world_size
            screen_height = world_size
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            # DRAW BOAT
            view_boat = rendering.make_circle(boat_radius)
            view_boat.set_color(*boat_color)
            # append transformation handler
            self.boat_trans = rendering.Transform()
            view_boat.add_attr(self.boat_trans)
            # add in viewer
            self.viewer.add_geom(view_boat)
            
            # DRAW THRUST LEFT
            l, r, t, b = -vec_width / 2, vec_width / 2, thrust_left_length / 2, -thrust_left_length / 2
            view_thrust_left = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            view_thrust_left.set_color(*thrust_straight_color)
            self.thrust_left_trans = rendering.Transform()
            view_thrust_left.add_attr(self.thrust_left_trans)
            view_thrust_left.add_attr(self.boat_trans)
            self.viewer.add_geom(view_thrust_left)
            
            # DRAW THRUST RIGHT
            l, r, t, b = -vec_width / 2, vec_width / 2, thrust_right_length / 2, -thrust_right_length / 2
            view_thrust_right = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            view_thrust_right.set_color(*thrust_straight_color)
            self.thrust_right_trans = rendering.Transform()
            view_thrust_right.add_attr(self.thrust_right_trans)
            view_thrust_right.add_attr(self.boat_trans)
            self.viewer.add_geom(view_thrust_right)
            
            # DRAW DIRECTION
            l, r, t, b = -vec_width / 2, vec_width / 2, boat_radius / 2, -boat_radius / 2
            view_direction = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            view_direction.set_color(*direction_color)
            self.direction_trans = rendering.Transform()
            view_direction.add_attr(self.direction_trans)
            view_direction.add_attr(self.boat_trans)
            self.viewer.add_geom(view_direction)
            
            # DRAW ICC
            view_icc = rendering.make_circle(boat_radius / 4)
            view_icc.set_color(*icc_color)
            self.icc_trans = rendering.Transform()
            view_icc.add_attr(self.icc_trans)
            self.viewer.add_geom(view_icc)
            
            # DRAW OBSTACLES
            for obs_x, obs_y, obs_type, obs_dim in self.obstacles:
                obs_x *= self.scale
                obs_y *= self.scale
                if obs_type == 'c':
                    view_obs = rendering.make_circle(obs_dim * self.scale)
                # elif obs_type == 'r':
                else:
                    obs_dim = (obs_dim[0] * self.scale, obs_dim[1] * self.scale)
                    l, r, t, b = -obs_dim[0] / 2, obs_dim[0] / 2, obs_dim[1] / 2, -obs_dim[1] / 2
                    view_obs = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                view_obs.set_color(*obstacle_color)
                view_obs.add_attr(rendering.Transform(translation=(obs_x, obs_y)))
                self.viewer.add_geom(view_obs)
            
            # DRAW GOAL
            goal = self.goal_state * self.scale
            goal_radius = self.goal_radius * self.scale
            view_goal = rendering.make_circle(goal_radius)
            view_goal.set_color(*goal_color)
            view_goal.add_attr(rendering.Transform(translation=(goal[0], goal[1])))
            self.viewer.add_geom(view_goal)
            
            # DRAW WAVE SPEED
            # arrow body
            l, r, t, b = -wave_width / 2, wave_width / 2, wave_length / 2, -wave_length / 2
            view_wave_arrow = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            view_wave_arrow.set_color(*wave_color)
            self.wave_arrow_trans = rendering.Transform()
            view_wave_arrow.add_attr(self.wave_arrow_trans)
            self.viewer.add_geom(view_wave_arrow)
            
            # arrow tip
            l, r, t = -wave_width * 1.5, wave_width * 1.5, wave_width * 1.5
            view_wave_tip = rendering.FilledPolygon([(l, 0), (0, t), (r, 0)])
            view_wave_tip.set_color(*wave_color)
            view_wave_tip.add_attr(self.wave_arrow_trans)
            self.wave_tip_trans = rendering.Transform()
            view_wave_tip.add_attr(self.wave_tip_trans)
            self.viewer.add_geom(view_wave_tip)
            
            # EXPORT VARIABLE SIZE OBJECTS
            self._view_thrust_left = view_thrust_left
            self._view_thrust_right = view_thrust_right
            self._view_wave_arrow = view_wave_arrow
        
        # env not reset
        if self.position is None:
            return None
        
        #
        # EDIT OBJECTS
        #
        # boat movement
        self.boat_trans.set_translation(position[0], position[1])
        
        # thrust left magnitude and movement
        l, r, t, b = -vec_width / 2 - axle_width / 2, vec_width / 2 - axle_width / 2, \
                     thrust_left_length / 2, -thrust_left_length / 2
        self._view_thrust_left.v = [(l, b), (l, t), (r, t), (r, b)]
        
        if thrust_left > 0:
            self._view_thrust_left.set_color(*thrust_straight_color)
        else:
            self._view_thrust_left.set_color(*thrust_reverse_color)
            thrust_left_length *= -1
        
        thrust_left_offset = (
            thrust_left_length / 2 * np.cos(np.pi / 2 + self.angle),
            thrust_left_length / 2 * np.sin(np.pi / 2 + self.angle),
        )
        self.thrust_left_trans.set_translation(*thrust_left_offset)
        self.thrust_left_trans.set_rotation(self.angle)
        
        # thrust right magnitude and movement
        l, r, t, b = -vec_width / 2 + axle_width / 2, vec_width / 2 + axle_width / 2, \
                     thrust_right_length / 2, -thrust_right_length / 2
        self._view_thrust_right.v = [(l, b), (l, t), (r, t), (r, b)]
        
        if thrust_right > 0:
            self._view_thrust_right.set_color(*thrust_straight_color)
        else:
            self._view_thrust_right.set_color(*thrust_reverse_color)
            thrust_right_length *= -1
        
        thrust_right_offset = (
            thrust_right_length / 2 * np.cos(np.pi / 2 + self.angle),
            thrust_right_length / 2 * np.sin(np.pi / 2 + self.angle),
        )
        self.thrust_right_trans.set_translation(*thrust_right_offset)
        self.thrust_right_trans.set_rotation(self.angle)
        
        # direction movement
        direction_offset = (
            boat_radius / 2 * np.cos(np.pi / 2 + self.angle),
            boat_radius / 2 * np.sin(np.pi / 2 + self.angle),
        )
        self.direction_trans.set_translation(*direction_offset)
        self.direction_trans.set_rotation(self.angle)
        
        # ICC movement
        self.icc_trans.set_translation(icc[0], icc[1])
        
        # wave perturbation
        l, r, t, b = -wave_width / 2, wave_width / 2, wave_length / 2, -wave_length / 2
        self._view_wave_arrow.v = [(l, b), (l, t), (r, t), (r, b)]
        
        wave_arrow_displacement = (
            wave_width + wave_length / 2 * 1.25,
            wave_width + wave_length / 2 * 1.25
        )
        wave_arrow_angle = np.arctan2(wave_speed[1], wave_speed[0])
        self.wave_arrow_trans.set_translation(*wave_arrow_displacement)
        self.wave_arrow_trans.set_rotation(wave_arrow_angle - np.pi / 2)
        
        wave_tip_displacement = (
            wave_length / 2 * np.cos(wave_arrow_angle),
            wave_length / 2 * np.sin(wave_arrow_angle)
        )
        self.wave_tip_trans.set_translation(*wave_tip_displacement)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    @staticmethod
    def _distance_circles(obs_pos, obs_radius, boat_pos, boat_radius):
        centers_dist = np.linalg.norm(obs_pos - boat_pos)
        total_radius_length = obs_radius + boat_radius
        return centers_dist - total_radius_length
    
    @staticmethod
    def _distance_rectangle(obs_pos, obs_dimensions, boat_pos, boat_radius):
        obs_l = obs_pos[0] - obs_dimensions[0] / 2
        obs_r = obs_pos[0] + obs_dimensions[0] / 2
        obs_b = obs_pos[1] - obs_dimensions[1] / 2
        obs_t = obs_pos[1] + obs_dimensions[1] / 2
        # find nearest point of obstacle border from center of circle
        min_values = np.array([obs_l, obs_b])
        max_values = np.array([obs_r, obs_t])
        dist = boat_pos - np.clip(boat_pos, min_values, max_values)
        # return actual distance between boat border and obstacle border
        return np.linalg.norm(dist) - boat_radius
    
    def _distance_from_goal(self):
        return self._distance_circles(self.goal_state, self.goal_radius,
                                      self.position, self.boat_radius)
    
    def _has_collided_obstacle(self):
        result = False
        for obs_x, obs_y, obs_type, obs_dim in self.obstacles:
            obs_pos = np.array([obs_x, obs_y])
            if obs_type == 'c':
                dist = self._distance_circles(obs_pos, obs_dim, self.position, self.boat_radius)
                result |= dist <= 0
            else:
                dist = self._distance_rectangle(obs_pos, obs_dim, self.position, self.boat_radius)
                result |= dist <= 0
        return result
    
    def _has_reached_goal(self):
        return self._distance_from_goal() <= 0
    
    def _has_collided_border(self):
        return np.any(self.position - self.boat_radius < self.observation_space.low[0:2]) \
               or np.any(self.position - self.boat_radius > self.observation_space.high[0:2])
