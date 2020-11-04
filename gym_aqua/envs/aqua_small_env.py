import gym
import numpy as np
from gym import spaces


class AquaSmallEnv(gym.Env):
    """
    World size: 1000 x 1000 pix^2
    Motor speed: [-10, +10] pix/s
    Waves speed: [ -1,  +1] pix/s
    Waves speed variance: 0.01 pix/s
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, scenario=0):
        
        # # get screen resolution
        # screen = pyglet.window \
        #     .get_platform() \
        #     .get_default_display() \
        #     .get_default_screen()
        
        self.scale = 0.6
        
        # world constants
        self.world_size = 1000
        self.motor_min_thrust = -2
        self.motor_max_thrust = +2
        self.wave_min_speed = -1
        self.wave_max_speed = +1
        self.wave_speed_variance = 0.01
        self.reward_step = -0.1
        self.reward_collision = -100
        self.reward_goal = +100
        
        # env spaces
        # vL, vR
        self.action_space = \
            spaces.Box(self.motor_min_thrust,
                       self.motor_max_thrust,
                       shape=[2], dtype=np.float32)
        # x, y, angle
        self.observation_space = \
            spaces.Box(np.array([0, 0, -np.pi]),
                       np.array([self.world_size, self.world_size, np.pi]),
                       dtype=np.float32)
        # vx, vy
        self.wave_space = \
            spaces.Box(self.wave_min_speed,
                       self.wave_max_speed,
                       shape=[2], dtype=np.float32)
        
        if scenario == 0:
            self.obstacles = [
                # x    y  type  radius/(width,heigth)
                # (150, 750, "c", 50),
                # (200, 350, "c", 100),
                # (350, 550, "r", (50, 50)),
                # (450, 200, "r", (100, 100)),
                # (550, 700, "c", 100),
                # (650, 350, "c", 50),
                # (650, 850, "r", (50, 50)),
                # (800, 550, "c", 100),
                # (850, 200, "c", 100),
                # (850, 750, "c", 50)
            ]
            self.goal = np.array([150, 650, 25], dtype=np.float32)
        else:
            # TODO add more scenarios
            raise Exception('No scenario available at that index!')
        
        # world dynamics
        self.wave_speed = np.zeros(2, dtype=np.float32)
        
        # internal state
        self.position = None
        self.angle = None
        self.thrust_left = None
        self.thrust_right = None
        self.thrust_total = None
        
        # rendering
        self.tau = 0.2  # time instant for numerical computation
        self.boat_radius = 25
        self.l = self.boat_radius  # absolute distance between the two motors
        self.ICC = np.zeros(2, dtype=np.float32)
        self.viewer = None
    
    def reset(self):
        self.position = np.array([550, 300], dtype=np.float32)
        self.angle = 0
        self.thrust_left = 0
        self.thrust_right = 0
        self.thrust_total = 0
        self.wave_speed = self.wave_space.sample()
        return np.array([self.position[0], self.position[1], self.angle], dtype=np.int32)
    
    def step(self, action: np.ndarray):
        action = np.array(action, dtype=np.float32)
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
            R = self.l / 2 * (self.thrust_right + self.thrust_left) / \
                (self.thrust_right - self.thrust_left)
            # angular speed
            w = (self.thrust_right - self.thrust_left) / self.l
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
        else:
            # LINEAR MOTION
            # self.angle remains the same
            self.thrust_total = self.thrust_right + self.thrust_left
            self.position += self.thrust_total * direction * self.tau
        
        # add waves contribution
        self.position += self.wave_speed * self.tau
        
        # UPDATE WAVES
        diff = np.random.uniform(-self.wave_speed_variance, self.wave_speed_variance, 2)
        new_wave_speed = self.wave_speed + diff
        new_wave_speed[new_wave_speed > self.wave_max_speed] = self.wave_max_speed
        new_wave_speed[new_wave_speed < self.wave_min_speed] = self.wave_min_speed
        self.wave_speed = new_wave_speed
        
        # REWARD AND GOAL
        reward = self.reward_step
        done = False
        if self._has_collided_obstacle() or self._has_collided_border():
            reward = self.reward_collision
            done = True
        elif self._has_reached_goal():
            reward = self.reward_goal
            done = True
        
        return np.array([self.position[0], self.position[1], self.angle], dtype=np.int32), \
               reward, done, {}
    
    # TODO make this prettier
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
        axle_width = self.l * self.scale
        vL = self.thrust_left * self.scale
        vR = self.thrust_right * self.scale
        ICC = self.ICC * self.scale
        wave_speed = self.wave_speed * self.scale
        
        # standard sizes in the scene
        vec_width = boat_radius / 2
        thrust_left_length = 40 * abs(vL)
        thrust_right_length = 40 * abs(vR)
        wave_width = 10 * self.scale
        wave_length = 40 * np.linalg.norm(wave_speed)
        
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
            goal = self.goal * self.scale
            view_goal = rendering.make_circle(goal[2])
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
        
        if vL > 0:
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
        
        if vR > 0:
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
        self.icc_trans.set_translation(ICC[0], ICC[1])
        
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
    
    def _has_collided_obstacle(self):
        result = False
        for obs_x, obs_y, obs_type, obs_dim in self.obstacles:
            if obs_type == 'c':
                pos_dist = np.linalg.norm(np.array([obs_x, obs_y]) - self.position[0:2])
                size_dist = obs_dim + self.boat_radius
                actual_dist = pos_dist - size_dist
                result |= actual_dist <= 0
            else:
                obs_l = obs_x - obs_dim[0] / 2
                obs_r = obs_x + obs_dim[0] / 2
                obs_b = obs_y - obs_dim[1] / 2
                obs_t = obs_y + obs_dim[1] / 2
                # find nearest point to circle
                dist_x = self.position[0] - np.clip(self.position[0], obs_l, obs_r)
                dist_y = self.position[1] - np.clip(self.position[1], obs_b, obs_t)
                dist = np.linalg.norm([dist_x, dist_y])
                result |= dist <= self.boat_radius
        return result
    
    def _has_reached_goal(self):
        pos_dist = np.linalg.norm(self.goal[0:2] - self.position[0:2])
        size_dist = self.goal[2] + self.boat_radius
        actual_dist = pos_dist - size_dist
        return actual_dist <= 0
    
    def _has_collided_border(self):
        return self.position[0] - self.boat_radius < 0 \
               or self.position[0] + self.boat_radius > self.world_size \
               or self.position[1] - self.boat_radius < 0 \
               or self.position[1] + self.boat_radius > self.world_size
