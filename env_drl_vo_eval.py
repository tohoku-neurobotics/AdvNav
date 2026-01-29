import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import numpy as np
import os
from datetime import datetime

from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.ackermann import AckermannRobot
from utils.state import ObservableState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
from collections import deque

class CrowdSim:
    def __init__(self, args, action_range, action_choices=None, digit_env=None):
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        # self.robot_policy = args.policy
        self.robot_model = args.robot_model
        self.robot_test_model = args.robot_test_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        self.human_num_max = args.human_num_max
        self.static_obstacle_num_max = args.static_obstacle_num_max
        self.use_angular = args.use_angular
        self.random_radius = args.random_radius
        
        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32) # used for visualization

        # laser state
        self.scan_current = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        self.scan_current_layer = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        
        self.global_time = 0.0
        self.global_step = 0
        self.time_limit = 50
        self.time_step = 0.2
        self.y_range = 7.0
        self.x_range = 7.0
        self.v_min = 0.1
        self.max_episode_step = int(self.time_limit / self.time_step)
        self.randomize_attributes = False
        self.success_reward = 0.5
        self.collision_penalty = -0.6
        self.collision_layer_penalty = -0.1
        self.emotion_penalty = -0.1
        self.emotion_penalty_factor = 0.08
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = 0.4
        self.goal_distance_factor = 0.3
        # self.digit_reward_factor = 0.2 # with torque penalty
        self.digit_reward_factor = 1.0 # without torque penalty
        self.digit_crazy_penalty = -0.5  
        self.angular_penalty = -0.03

        # here, more lines can be added to simulate obstacles
        self.lines = np.zeros((4, 2, 2), dtype=np.float32)
        margin = [10.0, 10.0]
        self.lines[0, :, :] = np.array([[-margin[0], -margin[1]],
                                        [-margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[1, :, :] = np.array([[-margin[0],  margin[1]],
                                        [margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[2, :, :] = np.array([[margin[0],  margin[1]],
                                        [margin[0], -margin[1]]], dtype=np.float32) 
        self.lines[3, :, :] = np.array([[margin[0], -margin[1]],
                                        [-margin[0], -margin[1]]], dtype=np.float32) 
        self.circle_radius = 5.0 # human distribution margin
        self.static_obstacle_area_x = 3.0 # static obstacle distribution area
        self.static_obstacle_area_y = 1.5 
        self.static_obstacles = None

        self.human_num = None
        self.static_obstacle_num = None

        self.obstacle_layer_len = 0.2
        # self.layer_len = [0.2, 0.5]
        # self.human_emotions = ['positive', 'negative']
        self.layer_len = [0.2, 0.2, 0.2]
        self.human_emotions = ['happy', 'neutral', 'angry']
        self.humans = None
        self.human_v_pref = 0.6
        self.rectangles = None
        self.action_range = action_range
        self.action_choices = action_choices
        if self.robot_test_model == 'ackermann':
            self.robot = AckermannRobot(radius=0.45)
        else:
            self.robot = Robot(radius=0.3)
        self.robot.time_step = self.time_step
        self.robot.v_pref = action_range[1, 0]
        self.action_last = np.zeros(2)
        self.acceleration = [1.0, 1.0]
        self.robot_visible_threshold = 0.5

        # dwa parameters
        self.acc_linear_max = self.acceleration[0]
        self.acc_angular_max = self.acceleration[1]
        self.dwa_resolution_linear_v = 0.02
        self.dwa_resolution_angular_v = 0.02
        self.dwa_look_forward_steps = 5
        self.dwa_dist_goal_cost = 0.4
        self.delta_linear_v_max = self.acc_linear_max * self.time_step
        self.delta_angular_v_max = self.acc_angular_max * self.time_step
        self.delta_linear_v = 0.05
        self.delta_angular_v = 0.05
        
        # observation image
        self.frame_stack = 10  # Fixed to match expected dimensions in construct_obs
        self.lidar_frames = deque([], maxlen=self.frame_stack)
        self.image_size = args.image_size
        self.range_resolution = (2 * self.laser_max_range) / self.image_size
        self.lidar_resolution = int(self.n_laser / self.image_size)

        # LIPM
        self.w = np.sqrt(9.81/1.02)
        self.cosh_wt = np.cosh(self.w * self.time_step)
        self.sinh_wt = np.sinh(self.w * self.time_step)
        
        # mujoco digit model
        self.digit_env = digit_env
        self.mujoco_visualize = False
        if digit_env is not None:
            self.repeat_action_num = int(self.time_step / digit_env.cfg.control.control_dt)
        
        # lidar to image
        self.frames = deque([], maxlen=self.frame_stack)
        self.image_size = args.image_size
        self.single_frame = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.r_resolution = self.laser_max_range / self.image_size
        self.theta_resolution = self.n_laser / self.image_size

        # visualization on 2D plane
        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # log lidar, robot, and humans
        self.log_env = {}

    def generate_random_static_obstacle(self):
        self.static_obstacle_num = int(np.random.randint(self.static_obstacle_num_max, size=1)[0] + 1)
        self.static_obstacles = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
        while True:
            positions_x = np.random.uniform(-self.static_obstacle_area_x, self.static_obstacle_area_x, 
                                            (self.static_obstacle_num, 1))
            positions_y = np.random.uniform(-self.static_obstacle_area_y, self.static_obstacle_area_y, 
                                            (self.static_obstacle_num, 1))
            # radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            # keep radius constant
            if self.random_radius:
                radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            else:
                radiuses = 0.3 * np.ones((self.static_obstacle_num, 1), dtype=np.float32)
            collision = False
            for i in range(self.static_obstacle_num):
                temp = False
                for j in range(i + 1, self.static_obstacle_num):
                    # allow 0.1m overlap
                    if hypot(positions_x[i] - positions_x[j], positions_y[i] - positions_y[j]) <= radiuses[i] + radiuses[j] - 0.1:
                        collision = True
                        temp = True
                        break
                if temp:
                    break
            if not collision:
                self.static_obstacles = np.hstack((positions_x, positions_y, radiuses))
                break
        
    def generate_random_human_position(self):
        max_restarts = 100            # maximum number of full restarts
        relax_factor = 1.0           # 1.0 means no relaxation; <1.0 relaxes the collision constraints
        if self.human_num_max >= 15:
            relax_factor = 0.2
        for restart in range(max_restarts):
            self.human_num = self.human_num_max
            self.humans = [None] * self.human_num
            failed = False

            # Try to generate each human with the current relax_factor.
            for i in range(self.human_num):
                human = self.generate_circle_crossing_human(relax_factor)
                if human is None:
                    failed = True
                    break  # stop trying to add more humans
                self.humans[i] = human

            if failed:
                # print("Restarting human generation with relaxed constraints: relax_factor =", relax_factor)
                relax_factor *= 0.9  # reduce the required separation by 10%
                continue  # restart the whole generation process

            # If all humans are successfully generated, set their policies.
            for i in range(len(self.humans)):
                human_policy = policy_factory[self.human_policy_name]()
                human_policy.time_step = self.time_step
                human_policy.max_speed = self.humans[i].v_pref
                human_policy.radius = self.humans[i].radius
                human_policy.max_robot_speed = self.robot.v_pref
                self.humans[i].set_policy(human_policy)

            # Generation succeeded; exit the restart loop.
            return

        # If we get here, even after several restarts we could not generate valid positions.
        raise Exception("Failed to generate valid human positions after multiple restarts.")

    def generate_circle_crossing_human(self, relax_factor=1.0):
        """
        Tries to generate a single human with non-colliding position.
        The effective minimum distance is relaxed by the relax_factor (should be <= 1.0).
        Returns a Human object if successful, or None if no valid candidate is found.
        """
        if self.static_obstacles is None:
            raise NotImplementedError(self.static_obstacles)
            
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = self.human_v_pref

        max_attempts = 100
        attempt = 0
        found = False

        while attempt < max_attempts:
            attempt += 1
            angle = np.random.random() * np.pi * 2
            
            # add some noise to simulate all the possible cases the robot could encounter
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise

            collide = False
            # Check collisions with the robot and any humans already placed.
            for agent in [self.robot] + self.humans:
                if agent is None:
                    continue
                # The required minimum distance is the sum of the radii plus the (possibly relaxed) discomfort distance.
                min_dist = human.radius + agent.radius + self.discomfort_dist * relax_factor
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break

            if not collide:
                # Check collisions with static obstacles.
                for static_obs in range(self.static_obstacle_num):
                    min_dist = human.radius + self.static_obstacles[static_obs, 2] + self.discomfort_dist * relax_factor
                    if norm((px - self.static_obstacles[static_obs, 0],
                            py - self.static_obstacles[static_obs, 1])) < min_dist:
                        collide = True
                        break

            if not collide:
                # Candidate accepted.
                human_theta = atan2(-py, -px)
                human.set(px, py, -px, -py, 0, 0, human_theta)
                emotion_idx = np.random.randint(0, len(self.human_emotions))
                human.emotion = self.human_emotions[emotion_idx]
                found = True
                break

        if not found:
            # After max_attempts, if no valid candidate is found, return None to signal failure.
            return None

        return human


        
    def get_lidar(self, layer):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        # robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        num_line = self.lines.shape[0]
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        InitializeEnv(num_line, num_circle_human + num_circle_obstacle, self.n_laser, self.laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle_human):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            emotion_idx = self.human_emotions.index(self.humans[i].emotion)
            set_circles(3 * i + 2, self.humans[i].radius + self.layer_len[emotion_idx] * layer)
        for i in range (num_circle_obstacle):
            set_circles(3 * (i + num_circle_human)    , self.static_obstacles[i, 0])
            set_circles(3 * (i + num_circle_human) + 1, self.static_obstacles[i, 1])
            set_circles(3 * (i + num_circle_human) + 2, self.static_obstacles[i, 2] + self.obstacle_layer_len * layer)
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        if layer == 0:
            self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32)
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            if layer == 0:
                ### used for visualization
                self.scan_intersection[i, 0, 0] = self.robot.px
                self.scan_intersection[i, 0, 1] = self.robot.py
                self.scan_intersection[i, 1, 0] = get_scan_line(4 * i + 2)
                self.scan_intersection[i, 1, 1] = get_scan_line(4 * i + 3)
                ### used for visualization
        if layer == 0:
            self.scan_current = np.clip(scan, self.laser_min_range, self.laser_max_range)
        elif layer == 1:
            self.scan_current_layer = np.clip(scan, self.laser_min_range, self.laser_max_range)
        ReleaseEnv()
        

    def get_frame(self):
        self.single_frame = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.get_lidar(0)
        for i in range(self.n_laser):
            if self.scan_current[i] == self.laser_max_range:
                continue
            j = int(i / self.theta_resolution)
            if j >= self.image_size:
                j = self.image_size - 1
            k = int(self.scan_current[i] / self.r_resolution)
            if k >= self.image_size:
                k = self.image_size - 1
            self.single_frame[0, j, k] = 255

        self.get_lidar(1)
        for i in range(self.n_laser):
            if self.scan_current_layer[i] == self.laser_max_range:
                continue
            j = int(i / self.theta_resolution)
            if j >= self.image_size:
                j = self.image_size - 1
            k = int(self.scan_current_layer[i] / self.r_resolution)
            if k >= self.image_size:
                k = self.image_size - 1
            if self.single_frame[0, j, k] != 255:
                self.single_frame[0, j, k] = 127

    def is_collision(self, layer):
        for i in range(self.human_num):
            dis = hypot(self.robot.px - self.humans[i].px, self.robot.py - self.humans[i].py)
            emotion_idx = self.human_emotions.index(self.humans[i].emotion)
            if dis < self.robot.radius + self.humans[i].radius + layer * self.layer_len[emotion_idx]:
                return True
        for i in range(self.static_obstacle_num):
            dis = hypot(self.robot.px - self.static_obstacles[i, 0], self.robot.py - self.static_obstacles[i, 1])
            if dis < self.robot.radius + self.static_obstacles[i, 2] + layer * self.obstacle_layer_len:
                return True
        return False

    def construct_obs(self):
        # px, py, vx, vy
        self.humans_state_in_robot_frame = np.zeros((self.human_num+self.static_obstacle_num, 4), dtype=np.float32)
        ped_pos_map = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
        c_theta = cos(self.robot.theta)
        s_theta = sin(self.robot.theta)
        for i in range (self.human_num+self.static_obstacle_num):
            if i < self.human_num:
                px_world = self.humans[i].px
                py_world = self.humans[i].py
                vx_world = self.humans[i].vx
                vy_world = self.humans[i].vy
            else:
                px_world = self.static_obstacles[i - self.human_num, 0]
                py_world = self.static_obstacles[i - self.human_num, 1]
                vx_world = 0.0
                vy_world = 0.0
            dx = px_world - self.robot.px
            dy = py_world - self.robot.py
            px_local = dy * s_theta + dx * c_theta
            py_local = dy * c_theta - dx * s_theta
            vx_local = vy_world * s_theta + vx_world * c_theta
            vy_local = vy_world * c_theta - vx_world * s_theta
            self.humans_state_in_robot_frame[i, :] = np.array([px_local, py_local, vx_local, vy_local], dtype=np.float32)
            if fabs(px_local) < self.laser_max_range and fabs(py_local) < self.laser_max_range:
                r = int((self.laser_max_range - py_local) / self.range_resolution)
                c = int((px_local + self.laser_max_range) / self.range_resolution)
                if c < self.image_size and r < self.image_size and c >= 0 and r >= 0:
                    ped_pos_map[0, r, c] = vx_local
                    ped_pos_map[1, r, c] = vy_local

        lidar_map = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        lidar_avg = np.zeros((2 * self.frame_stack, self.image_size), dtype=np.float32)
        lidar_frames = np.concatenate(list(self.lidar_frames), axis=0)
        lidar_frames = np.reshape(lidar_frames, (self.frame_stack, -1))
        for idx, lidar in enumerate(lidar_frames):
            for j in range(self.image_size):
                lidar_avg[2 * idx, j] = np.min(lidar[j*self.lidar_resolution:(j+1)*self.lidar_resolution])
                lidar_avg[2 * idx + 1, j] = np.mean(lidar[j*self.lidar_resolution:(j+1)*self.lidar_resolution])
        # Repeat lidar_avg five times along the first axis
        lidar_avg_map = np.tile(lidar_avg, (5, 1))
        lidar_map[0, :, :] = lidar_avg_map
        obs = np.concatenate((ped_pos_map, lidar_map), axis=0)
        return obs
    
        
    def step(self, action, eval=False, save_data=False):
        human_actions = np.zeros((self.human_num, 2), dtype=np.float32)
        for i in range(self.human_num):
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != self.humans[i]]
            for k in range(self.static_obstacle_num):
                ob.append(ObservableState(
                        self.static_obstacles[k, 0], 
                        self.static_obstacles[k, 1], 
                        0.0, 0.0, self.static_obstacles[k, 2])
                        )
            if self.robot_visible_threshold > hypot(self.robot.px - self.humans[i].px, self.robot.py - self.humans[i].py):
                robot_ob = ObservableState(
                        self.robot.px, 
                        self.robot.py, 
                        self.robot.vx, 
                        self.robot.vy, 
                        self.robot.radius)
                action_temp = self.humans[i].act(ob, robot_state=robot_ob)
                human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)
            else:
                action_temp = self.humans[i].act(ob)
                human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)

        # update robot states
        action_copy = np.array([action[0], action[1]])
        if self.digit_env is None:
            robot_theta = self.robot.theta + action[1] * self.time_step
            if robot_theta > np.pi:
                robot_theta -= (2.0 * np.pi)
            elif robot_theta < -np.pi:
                robot_theta += (2.0 * np.pi)
            if self.robot_test_model == 'differential' or self.robot_model == 'differential':
                robot_x = self.robot.px + action[0] * self.time_step * cos(robot_theta)
                robot_y = self.robot.py + action[0] * self.time_step * sin(robot_theta)
                
            elif self.robot_test_model == 'lip' or self.robot_model == 'lip':
                pf_x = (self.action_last[0] * self.cosh_wt - action[0]) / (self.w * self.sinh_wt)
                x_n =  pf_x - pf_x * self.cosh_wt + self.action_last[0] * self.sinh_wt / self.w
                robot_x = self.robot.px + x_n * cos(robot_theta)
                robot_y = self.robot.py + x_n * sin(robot_theta)
            elif self.robot_test_model == 'ackermann' or self.robot_model == 'ackermann':
                robot_x, robot_y, robot_theta, actual_velocity, actual_steering, actual_theta_dot = self.robot.compute_pose(action)

        else:
            vel_command_to_digit = {
                'x_vel': action[0],
                'y_vel': 0.0,
                'yaw_vel': action[1]
            }
            self.digit_env.set_vel_command(vel_command_to_digit)
            
            for _ in range(self.repeat_action_num):
                st_time = time()
                self.digit_env.step(np.zeros(12))
                if self.mujoco_visualize:
                    end_time = time()
                    if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                        sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
            robot_x = self.digit_env.root_xy_pos[0]
            robot_y = self.digit_env.root_xy_pos[1]
            robot_theta = self.digit_env.root_rpy[2]
        action_copy[0] = hypot(robot_y - self.robot.py, robot_x - self.robot.px) / self.time_step
            
        # update states
        if self.robot_test_model == 'ackermann':
            self.robot.update_states(robot_x, robot_y, robot_theta, action_copy, forward_velocity=actual_velocity,
                steering_angle=actual_steering, 
                theta_dot=actual_theta_dot)
        else:
            self.robot.update_states(robot_x, robot_y, robot_theta, action_copy, differential=True)
       
        for i in range(self.human_num):
            self.humans[i].update_states(human_actions[i])

        # get new laser scan and grid map
        self.get_frame() 
        
        self.frames.append(self.single_frame)
        assert len(self.frames) == self.frame_stack
        lidar_image = np.concatenate(list(self.frames), axis=0)
        
        self.lidar_frames.append(self.scan_current / self.laser_max_range)
        assert len(self.lidar_frames) == self.frame_stack
        obs_image = self.construct_obs()
        
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        if eval:
            reaching_goal = goal_dist < (self.robot.radius - 0.1)
        else:
            reaching_goal = goal_dist < (self.robot.radius - 0.2)

        # collision detection between the robot and humans
        collision = self.is_collision(0)
        collision_layer = self.is_collision(1)
            
        dis_goal_reward = self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        # dis_goal_reward = 0.0
        self.goal_distance_last = goal_dist
        
        # angular_reward = fabs(action[1] - self.action_last[1]) * self.angular_penalty
        if self.use_angular:
            angular_reward = fabs(action[1]) * self.angular_penalty if fabs(action[1]) > 1.0 else 0.0
        else:
            angular_reward = 0.0
        
        self.action_last = action
        reward = collision_layer * self.collision_layer_penalty + dis_goal_reward + angular_reward
        if collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif collision_layer:
            done = False
            info = Danger(0.1)
        else:
            done = False
            info = Nothing()
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)

        # get the observation
        robot_goal_emotion_state = np.array([r / self.square_width, t / np.pi, 
                                             self.action_last[0] / self.action_range[1, 0],
                                             self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
        
        if save_data:
            self.global_step += 1
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, action[0], action[1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 4), dtype=np.float32)
            for i in range (self.human_num):
                emotion_idx = self.human_emotions.index(self.humans[i].emotion)
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius, emotion_idx], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers

        # Compute closest distance to any pedestrian (humans only)
        closest_human_distance = inf
        for i in range(self.human_num):
            dis_center = hypot(self.robot.px - self.humans[i].px, self.robot.py - self.humans[i].py)
            if dis_center < closest_human_distance:
                closest_human_distance = dis_center
        if self.human_num == 0:
            closest_human_distance = inf

        return obs_image, robot_goal_emotion_state, reward, done, info, closest_human_distance
    
    def save_video(self, steps, episodes):
        filename = 'eval_' + str(steps) + '_' + str(episodes)
        if self.digit_env is None:
            raise NotImplementedError(self.digit_env)
        self.digit_env.save_video(filename)
    
    def reset(self, seed=-1, save_data=False):
        self.global_time = 0.0
        self.global_step = 0
        self.action_last = np.zeros(2)
        self.static_obstacles = None
        self.log_env = {}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)
        
        if self.digit_env is not None:    
            # for initializing
            self.digit_env.reset(robot=np.array([self.robot.px, self.robot.py], dtype=np.float32))
            sleep(self.digit_env.cfg.control.control_dt)
            # initialize the locomotion for 2 seconds to let the robot step in place
            initial_time = np.random.uniform(2.0, 2.0 + self.time_step + self.digit_env.cfg.control.control_dt)
            for i in range(int(initial_time / self.digit_env.cfg.control.control_dt)):
                st_time = time()
                self.digit_env.step(np.zeros(12))
                if self.mujoco_visualize:
                    end_time = time()
                    if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                        sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
            
            robot_x = self.digit_env.root_xy_pos[0]
            robot_y = self.digit_env.root_xy_pos[1]
            robot_theta = self.digit_env.root_rpy[2] 
            # update states
            if self.robot_test_model == 'ackermann':
                self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), forward_velocity=0.0, steering_angle=0.0, theta_dot=0.0)
            else:
                self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), differential=True)
        
        self.goal_distance_last = self.robot.get_goal_distance()

        # 3,5 save
        # np.random.seed(5)
        if seed >= 0:
            np.random.seed(seed)
        self.generate_random_static_obstacle()
        self.generate_random_human_position()

        self.get_frame() 
        
        for _ in range(self.frame_stack):
            self.frames.append(self.single_frame)
        assert len(self.frames) == self.frame_stack
        lidar_image = np.concatenate(list(self.frames), axis=0)

        for _ in range(self.frame_stack):
            self.lidar_frames.append(self.scan_current / self.laser_max_range)
        assert len(self.lidar_frames) == self.frame_stack
        obs_image = self.construct_obs()

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)
        
        robot_goal_emotion_state = np.array([r / self.square_width, t / np.pi, 
                                             self.action_last[0] / self.action_range[1, 0],
                                             self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
       
        if save_data:
            self.log_env['ypr'] = -100.0 * np.ones((self.max_episode_step + 1, 3), dtype=np.float32)
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 4), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)
            self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 4), dtype=np.float32)
            for i in range(self.human_num):
                emotion_idx = self.human_emotions.index(self.humans[i].emotion)
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius, emotion_idx], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers
       
        return obs_image, robot_goal_emotion_state

    def get_density_in_radius(self, radius):
        """Calculate the number of obstacles (humans and static) within a given radius around the robot."""
        obstacle_count = 0
        robot_pos = (self.robot.px, self.robot.py)

        # Count humans within radius
        if self.humans is not None:
            for human in self.humans:
                human_pos = (human.px, human.py)
                distance = hypot(robot_pos[0] - human_pos[0], robot_pos[1] - human_pos[1])
                if distance <= radius:
                    obstacle_count += 1
        
        # Count static obstacles within radius
        if self.static_obstacles is not None:
            for i in range(self.static_obstacle_num):
                obstacle_pos = (self.static_obstacles[i, 0], self.static_obstacles[i, 1])
                distance = hypot(robot_pos[0] - obstacle_pos[0], robot_pos[1] - obstacle_pos[1])
                if distance <= radius:
                    obstacle_count += 1
                    
        return obstacle_count

    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-5.0, 5.0)
            self.ax.set_ylim(-5.0, 5.0)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            for i in range(self.static_obstacle_num):
                self.ax.add_artist(plt.Circle((self.static_obstacles[i, 0], self.static_obstacles[i, 1]), 
                                              self.static_obstacles[i, 2],
                                              fill=True, color='c'))
            plt.text(-4.5, -4.5, str(round(self.global_time, 2)), fontsize=20)
            x, y, theta = self.robot.px, self.robot.py, self.robot.theta
            dx = cos(theta)
            dy = sin(theta)
            self.ax.arrow(x, y, dx, dy,
                width=0.01,
                length_includes_head=True, 
                head_width=0.15,
                head_length=1,
                fc='r',
                ec='r')
            ii = 0
            lines = []
            while ii < self.n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.draw()
            plt.pause(0.001)
            plt.cla()

    def export_initial_state(self, filepath):
        """Export current initial state to NPZ file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Environment metadata
            environment_type = 'eval'
            timestamp = datetime.now().isoformat()
            
            # Robot state [px, py, gx, gy, vx, vy, theta, radius, v_pref]
            robot_state = np.array([
                self.robot.px, self.robot.py, self.robot.gx, self.robot.gy,
                self.robot.vx, self.robot.vy, self.robot.theta,
                self.robot.radius, self.robot.v_pref
            ], dtype=np.float32)
            
            # Humans state [n_humans x 10] -> [px, py, gx, gy, vx, vy, theta, radius, v_pref, emotion_id]
            humans_data = []
            for human in self.humans:
                emotion_id = self.human_emotions.index(human.emotion)
                human_data = [
                    human.px, human.py, human.gx, human.gy,
                    human.vx, human.vy, human.theta,
                    human.radius, human.v_pref, emotion_id
                ]
                humans_data.append(human_data)
            humans_state = np.array(humans_data, dtype=np.float32)
            
            # Static obstacles [n_obstacles x 3] -> [px, py, radius]
            static_obstacles_state = self.static_obstacles.copy()
            
            # Environment parameters
            env_params = np.array([
                self.circle_radius, self.human_num_max, self.static_obstacle_num_max,
                self.human_num, self.static_obstacle_num
            ], dtype=np.float32)
            
            # Save to NPZ file
            np.savez_compressed(
                filepath,
                environment_type=np.array([environment_type], dtype='U20'),
                timestamp=np.array([timestamp], dtype='U30'),
                robot=robot_state,
                humans=humans_state,
                static_obstacles=static_obstacles_state,
                env_params=env_params
            )
            
            print(f"Eval environment state exported to: {filepath}")
            
        except Exception as e:
            print(f"Error exporting eval environment state: {e}")
            raise

    def import_initial_state(self, filepath):
        """Import and validate initial state from NPZ file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"State file not found: {filepath}")
            
            # Load state data
            state_data = np.load(filepath)
            
            # Validate state data
            self._validate_imported_state(state_data)
            
            return state_data
            
        except Exception as e:
            print(f"Error importing eval environment state: {e}")
            raise

    def reset_from_imported_state(self, state_data_or_filepath, save_data=False):
        """Reset environment using imported state data"""
        try:
            # Handle both state data dict and filepath
            if isinstance(state_data_or_filepath, str):
                state_data = self.import_initial_state(state_data_or_filepath)
            else:
                state_data = state_data_or_filepath
            
            # Reset basic environment state
            self.global_time = 0.0
            self.global_step = 0
            self.action_last = np.zeros(2)
            self.log_env = {}
            
            # Set robot state from imported data
            robot_data = state_data['robot']
            self.robot.set(
                px=float(robot_data[0]), py=float(robot_data[1]),
                gx=float(robot_data[2]), gy=float(robot_data[3]),
                vx=float(robot_data[4]), vy=float(robot_data[5]),
                theta=float(robot_data[6]), v_pref=float(robot_data[8])
            )
            self.robot.radius = float(robot_data[7])
            
            # Set static obstacles from imported data
            static_obstacles_data = state_data['static_obstacles']
            self.static_obstacle_num = len(static_obstacles_data)
            self.static_obstacles = static_obstacles_data.copy()
            
            # Set humans from imported data
            humans_data = state_data['humans']
            self.human_num = len(humans_data)
            self.humans = [None] * self.human_num
            
            for i, human_data in enumerate(humans_data):
                human = Human()
                human.time_step = self.time_step
                human.set(
                    px=float(human_data[0]), py=float(human_data[1]),
                    gx=float(human_data[2]), gy=float(human_data[3]),
                    vx=float(human_data[4]), vy=float(human_data[5]),
                    theta=float(human_data[6]), radius=float(human_data[7]),
                    v_pref=float(human_data[8])
                )
                emotion_id = int(human_data[9])
                human.emotion = self.human_emotions[emotion_id]
                
                # Set policy for human
                human_policy = policy_factory[self.human_policy_name]()
                human_policy.time_step = self.time_step
                human_policy.max_speed = human.v_pref
                human_policy.radius = human.radius
                human_policy.max_robot_speed = self.robot.v_pref
                human.set_policy(human_policy)
                
                self.humans[i] = human
            
            self.goal_distance_last = self.robot.get_goal_distance()
            
            # Get frame and prepare observation
            self.get_frame()
            
            for _ in range(self.frame_stack):
                self.frames.append(self.single_frame)
            assert len(self.frames) == self.frame_stack
            
            for _ in range(self.frame_stack):
                self.lidar_frames.append(self.scan_current / self.laser_max_range)
            assert len(self.lidar_frames) == self.frame_stack
            obs_image = self.construct_obs()
            
            # Calculate robot goal emotion state
            dx = self.robot.gx - self.robot.px
            dy = self.robot.gy - self.robot.py
            theta = self.robot.theta
            y_rel = dy * cos(theta) - dx * sin(theta)
            x_rel = dy * sin(theta) + dx * cos(theta)
            r = hypot(x_rel, y_rel)
            t = atan2(y_rel, x_rel)
            
            robot_goal_emotion_state = np.array([
                r / self.square_width, t / np.pi,
                self.action_last[0] / self.action_range[1, 0],
                self.action_last[1] / self.action_range[1, 1]
            ], dtype=np.float32)

            if save_data:
                self.log_env['ypr'] = -100.0 * np.ones((self.max_episode_step + 1, 3), dtype=np.float32)
                self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
                self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
                self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 4), dtype=np.float32)
                self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)
                self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

                self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
                self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
                humans_info = np.zeros((self.human_num, 4), dtype=np.float32)
                for i in range(self.human_num):
                    emotion_idx = self.human_emotions.index(self.humans[i].emotion)
                    humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius, emotion_idx], dtype=np.float32)
                self.log_env['humans'][self.global_step] = humans_info
                static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
                for i in range (self.static_obstacle_num):
                    static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                        self.static_obstacles[i, 1],
                                                        self.static_obstacles[i, 2]])
                self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
                lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
                for i in range(self.n_laser):
                    laser = self.scan_intersection[i]
                    lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
                self.log_env['laser'][self.global_step] = lasers
            
            print(f"Eval environment reset from imported state successfully")
            return obs_image, robot_goal_emotion_state
            
        except Exception as e:
            print(f"Error resetting eval environment from imported state: {e}")
            raise

    def _validate_imported_state(self, state_data):
        """Validate that imported state is safe and valid"""
        try:
            # Check required keys
            required_keys = ['environment_type', 'robot', 'humans', 'static_obstacles', 'env_params']
            for key in required_keys:
                if key not in state_data:
                    raise ValueError(f"Missing required key in state data: {key}")
            
            # Check environment type
            env_type = str(state_data['environment_type'][0])
            if env_type != 'eval':
                raise ValueError(f"Environment type mismatch. Expected 'eval', got '{env_type}'")
            
            # Validate robot state
            robot_data = state_data['robot']
            if robot_data.shape != (9,):
                raise ValueError(f"Invalid robot state shape. Expected (9,), got {robot_data.shape}")
            
            # Check robot position bounds
            robot_px, robot_py = robot_data[0], robot_data[1]
            if abs(robot_px) > self.square_width or abs(robot_py) > self.square_width:
                raise ValueError(f"Robot position out of bounds: ({robot_px}, {robot_py})")
            
            # Validate robot attributes
            robot_radius, robot_v_pref = robot_data[7], robot_data[8]
            if not (0.1 <= robot_radius <= 1.0):
                raise ValueError(f"Invalid robot radius: {robot_radius}")
            if not (0.1 <= robot_v_pref <= 3.0):
                raise ValueError(f"Invalid robot v_pref: {robot_v_pref}")
            
            # Validate humans state
            humans_data = state_data['humans']
            if len(humans_data.shape) != 2 or humans_data.shape[1] != 10:
                raise ValueError(f"Invalid humans state shape. Expected (n, 10), got {humans_data.shape}")
            
            # Check each human
            for i, human_data in enumerate(humans_data):
                # Position bounds
                human_px, human_py = human_data[0], human_data[1]
                if abs(human_px) > self.square_width or abs(human_py) > self.square_width:
                    raise ValueError(f"Human {i} position out of bounds: ({human_px}, {human_py})")
                
                # Attributes bounds
                human_radius, human_v_pref = human_data[7], human_data[8]
                if not (0.1 <= human_radius <= 1.0):
                    raise ValueError(f"Invalid human {i} radius: {human_radius}")
                if not (0.1 <= human_v_pref <= 3.0):
                    raise ValueError(f"Invalid human {i} v_pref: {human_v_pref}")
                
                # Emotion ID bounds
                emotion_id = int(human_data[9])
                if not (0 <= emotion_id < len(self.human_emotions)):
                    raise ValueError(f"Invalid emotion ID for human {i}: {emotion_id}")
            
            # Validate static obstacles
            obstacles_data = state_data['static_obstacles']
            if len(obstacles_data.shape) != 2 or obstacles_data.shape[1] != 3:
                raise ValueError(f"Invalid static obstacles shape. Expected (n, 3), got {obstacles_data.shape}")
            
            # Check each obstacle
            for i, obstacle_data in enumerate(obstacles_data):
                obs_px, obs_py, obs_radius = obstacle_data
                if abs(obs_px) > self.square_width or abs(obs_py) > self.square_width:
                    raise ValueError(f"Obstacle {i} position out of bounds: ({obs_px}, {obs_py})")
                if not (0.1 <= obs_radius <= 2.0):
                    raise ValueError(f"Invalid obstacle {i} radius: {obs_radius}")
            
            # Check for initial collisions between entities
            all_entities = []
            
            # Add robot
            all_entities.append({
                'type': 'robot',
                'px': robot_data[0], 'py': robot_data[1], 'radius': robot_data[7]
            })
            
            # Add humans
            for i, human_data in enumerate(humans_data):
                all_entities.append({
                    'type': f'human_{i}',
                    'px': human_data[0], 'py': human_data[1], 'radius': human_data[7]
                })
            
            # Add static obstacles
            for i, obstacle_data in enumerate(obstacles_data):
                all_entities.append({
                    'type': f'obstacle_{i}',
                    'px': obstacle_data[0], 'py': obstacle_data[1], 'radius': obstacle_data[2]
                })
            
            # Check for collisions
            for i in range(len(all_entities)):
                for j in range(i + 1, len(all_entities)):
                    entity1, entity2 = all_entities[i], all_entities[j]
                    distance = hypot(entity1['px'] - entity2['px'], entity1['py'] - entity2['py'])
                    min_distance = entity1['radius'] + entity2['radius'] + 0.1  # Small safety margin
                    
                    if distance < min_distance:
                        raise ValueError(f"Initial collision detected between {entity1['type']} and {entity2['type']}")
            
            print(f"Eval state validation passed successfully")
            
        except Exception as e:
            print(f"Eval state validation failed: {e}")
            raise


