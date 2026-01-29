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
from utils.state import ObservableState, JointState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
from collections import deque

class CrowdSim:
    def __init__(self, args, action_range, chaser_action_range, action_choices=None, chaser_action_choices=None, digit_env=None):
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.robot_policy = args.policy
        self.robot_model = args.robot_model
        self.robot_test_model = args.robot_test_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        self.chaser_model = args.chaser_model
        self.chaser_test_model = args.chaser_test_model
        self.chaser_goal_state_dim = args.robot_goal_state_dim
        self.human_num_max = args.human_num_max
        self.static_obstacle_num_max = args.static_obstacle_num_max
        self.use_angular = args.use_angular
        self.random_radius = args.random_radius
        self.training_object = args.training_object
        print('training_object: ' + self.training_object)
        self.random_chaser_reset = args.random_chaser_reset
        self.random_robot_goal = args.random_robot_goal
        
        # Robot policy mode configuration
        self.robot_policy_mode = getattr(args, 'robot_policy_mode', 'drl')
        self.action_choices = action_choices
        
        # Validate robot policy mode
        if self.robot_policy_mode not in ['drl', 'dwa', 'orca', 'rgl', 'drl-vo']:
            raise ValueError(f"Invalid robot_policy_mode: {self.robot_policy_mode}. Must be one of ['drl', 'dwa', 'orca', 'rgl', 'drl-vo']")
        
        # Check that action_choices is provided for DWA
        if self.robot_policy_mode == 'dwa' and action_choices is None:
            raise ValueError("action_choices must be provided when robot_policy_mode is 'dwa'")
        
        # Initialize ORCA policy for ORCA and RGL modes
        if self.robot_policy_mode in ['orca', 'rgl']:
            self.robot_orca_policy = policy_factory['orca']()
            self.robot_orca_policy.time_step = None  # Will be set later
            self.robot_orca_policy.safety_space = 0.2
            self.robot_orca_policy.neighbor_dist = 10
            self.robot_orca_policy.max_neighbors = 30
            self.robot_orca_policy.time_horizon = 5
            self.robot_orca_policy.time_horizon_obst = 5
        
        # DWA specific parameters
        self.dwa_horizon = 5
        self.dwa_obstacle_cost_weight = 1.0
        self.dwa_goal_cost_weight = 0.2
        self.dwa_safety_margin = 0.8
        
        # last-time distance from the robot to the goal
        self.goal_distance_last = None
        self.chaser_distance_last = None

        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32) # used for visualization
        
        # Per-chaser scan intersections (will be initialized after chaser_num is set)
        # Placeholder - will be properly initialized below
        
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
        self.vo_penalty = -0.1
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = 0.4
        self.goal_distance_factor = 0.3
        # self.digit_reward_factor = 0.2 # with torque penalty
        self.digit_reward_factor = 1.0 # without torque penalty
        self.digit_crazy_penalty = -0.5  
        self.angular_penalty = -0.03
        self.waiting_penalty = -0.3
        self.vo_threshold = 3.0
        # here, more lines can be added to simulate obstacles
        self.lines = np.zeros((4, 2, 2), dtype=np.float32)
        # Use x_range and y_range to align wall boundaries with collision boundaries
        margin = [self.x_range, self.y_range]
        self.lines[0, :, :] = np.array([[-margin[0], -margin[1]],
                                        [-margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[1, :, :] = np.array([[-margin[0],  margin[1]],
                                        [margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[2, :, :] = np.array([[margin[0],  margin[1]],
                                        [margin[0], -margin[1]]], dtype=np.float32) 
        self.lines[3, :, :] = np.array([[margin[0], -margin[1]],
                                        [-margin[0], -margin[1]]], dtype=np.float32) 
        self.circle_radius = 4.0 # human distribution margin
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
        self.human_v_pref = 1.0
        self.rectangles = None
        self.action_range = action_range
        self.chaser_action_range = chaser_action_range
        
        # Initialize robot
        if self.robot_model == 'ackermann' or self.robot_test_model == 'ackermann':
            self.robot = AckermannRobot(radius=0.45)
        else:
            self.robot = Robot(radius=0.3)
        self.robot.time_step = self.time_step
        self.robot.v_pref = action_range[1, 0]
        self.action_last = np.zeros(2)
        self.acceleration = [1.0, 1.0]
        self.robot_visible_threshold = 1.0
        
        # Initialize multiple chasers
        self.chaser_num = getattr(args, 'chaser_num', 1)
        self.chasers = []
        for i in range(self.chaser_num):
            if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                chaser = AckermannRobot(radius=0.45)
            else:
                chaser = Robot(radius=0.3)
            chaser.time_step = self.time_step
            chaser.v_pref = chaser_action_range[1, 0]
            self.chasers.append(chaser)
        
        # Per-chaser arrays
        self.chaser_action_lasts = [np.zeros(2) for _ in range(self.chaser_num)]
        self.chaser_acceleration = [1.0, 1.0]
        self.chaser_visible_threshold = 1.0
        self.chaser_active = [True] * self.chaser_num  # Track which chasers are still active (not removed after collision)
        self.chaser_distance_lasts = [None] * self.chaser_num  # Track last distance to goal for each chaser
        
        # Per-chaser scan arrays
        self.chaser_scan_intersections = [np.zeros((self.n_laser, 2, 2), dtype=np.float32) for _ in range(self.chaser_num)]
        self.chaser_scan_currents = [self.laser_max_range * np.ones(self.n_laser, dtype=np.float32) for _ in range(self.chaser_num)]
        self.chaser_scan_current_layers = [self.laser_max_range * np.ones(self.n_laser, dtype=np.float32) for _ in range(self.chaser_num)]
        
        # Maintain backward compatibility for single chaser access
        self.chaser = self.chasers[0] if self.chaser_num > 0 else None
        self.chaser_action_last = self.chaser_action_lasts[0] if self.chaser_num > 0 else np.zeros(2)
        self.chaser_time_step = self.time_step
        self.chaser_scan_intersection = self.chaser_scan_intersections[0] if self.chaser_num > 0 else np.zeros((self.n_laser, 2, 2), dtype=np.float32)
        self.chaser_scan_current = self.chaser_scan_currents[0] if self.chaser_num > 0 else self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        self.chaser_scan_current_layer = self.chaser_scan_current_layers[0] if self.chaser_num > 0 else self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)

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
        self.frame_stack = 10
        self.lidar_frames = deque([], maxlen=self.frame_stack)
        
        # Per-chaser lidar frames 
        self.chaser_lidar_frames_list = [deque([], maxlen=self.frame_stack) for _ in range(self.chaser_num)]
        # Backward compatibility
        self.chaser_lidar_frames = self.chaser_lidar_frames_list[0] if self.chaser_num > 0 else deque([], maxlen=self.frame_stack)
        
        self.image_size = args.image_size
        self.range_resolution = (2*self.laser_max_range) / self.image_size
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
        
        # Per-chaser frames
        self.chaser_frames_list = [deque([], maxlen=self.frame_stack) for _ in range(self.chaser_num)]
        self.chaser_single_frames = [np.zeros((1, self.image_size, self.image_size), dtype=np.uint8) for _ in range(self.chaser_num)]
        # Backward compatibility
        self.chaser_frames = self.chaser_frames_list[0] if self.chaser_num > 0 else deque([], maxlen=self.frame_stack)
        self.chaser_single_frame = self.chaser_single_frames[0] if self.chaser_num > 0 else np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        
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
            if self.random_radius:
                radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            else:
                radiuses = 0.3 * np.ones((self.static_obstacle_num, 1), dtype=np.float32)
            collision = False
            for i in range(self.static_obstacle_num):
                temp = False
                # Check collision with other static obstacles
                for j in range(i + 1, self.static_obstacle_num):
                    if hypot(positions_x[i] - positions_x[j], positions_y[i] - positions_y[j]) <= radiuses[i] + radiuses[j] - 0.1:
                        collision = True
                        temp = True
                        break
                if temp:
                    break
                # Check minimum distance to robot's initial position
                robot_init_x, robot_init_y = -self.circle_radius, 0.0
                robot_radius = self.robot.radius if hasattr(self, 'robot') else 0.3
                if hypot(positions_x[i] - robot_init_x, positions_y[i] - robot_init_y) < (robot_radius + radiuses[i] + self.discomfort_dist):
                    collision = True
                    break
                # Check minimum distance to chaser's initial position if not random
                if not self.random_chaser_reset:
                    chaser_init_x, chaser_init_y = 0.0, -self.circle_radius
                    chaser_radius = self.chaser.radius if hasattr(self, 'chaser') else 0.3
                    if hypot(positions_x[i] - chaser_init_x, positions_y[i] - chaser_init_y) < (chaser_radius + radiuses[i] + self.discomfort_dist):
                        collision = True
                        break
            if not collision:
                self.static_obstacles = np.hstack((positions_x, positions_y, radiuses))
                break
        
    def generate_random_human_position(self):
        self.human_num = int(np.random.randint(self.human_num_max, size=1)[0] + 1)
        self.humans = [None] * self.human_num
        for i in range(self.human_num):
            self.humans[i] = self.generate_circle_crossing_human()

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            human_policy.max_speed = self.humans[i].v_pref
            human_policy.radius = self.humans[i].radius
            human_policy.max_robot_speed = self.robot.v_pref
            self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        if self.static_obstacles is None:
            raise NotImplementedError(self.static_obstacles)
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = self.human_v_pref
        while True:
            angle = np.random.random() * np.pi * 2
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in self.humans:
                if agent is None:
                    continue
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            for static_obs in range(self.static_obstacle_num):
                min_dist = human.radius + self.static_obstacles[static_obs, 2] + self.discomfort_dist
                if norm((px - self.static_obstacles[static_obs, 0], 
                         py - self.static_obstacles[static_obs, 1])) < min_dist:
                    collide = True
                    break
            # Check collision with robot (always at -self.circle_radius, 0.0 at reset)
            robot_x = -self.circle_radius
            robot_y = 0.0
            min_dist = human.radius + self.robot.radius + self.discomfort_dist
            if norm((px - robot_x, py - robot_y)) < min_dist:
                collide = True
            # Check collision with chaser if not random (at 0.0, self.circle_radius)
            if not self.random_chaser_reset:
                chaser_x = 0.0
                chaser_y = -self.circle_radius
                chaser_radius = self.chaser.radius if hasattr(self, 'chaser') else 0.3
                min_dist = human.radius + chaser_radius + self.discomfort_dist
                if norm((px - chaser_x, py - chaser_y)) < min_dist:
                    collide = True
            if not collide:
                human_theta = atan2(-py - py, -px - px)
                human.set(px, py, -px, -py, 0, 0, human_theta)
                emotion_idx = np.random.randint(0, len(self.human_emotions))  
                human.emotion = self.human_emotions[emotion_idx]
                break
        return human
    
    def get_chaser_lidar(self, layer, chaser_idx=0):
        """
        Get lidar scan for a specific chaser.
        
        Args:
            layer (int): Layer index (0 or 1)
            chaser_idx (int): Index of the chaser requesting lidar scan
        
        Note: For SAC chasers - includes robot and other chasers so they can see and avoid/chase
              For ORCA chasers - uses get_full_state() instead (which doesn't include robot)
        """
        scan = np.zeros(self.n_laser, dtype=np.float32)
        chaser = self.chasers[chaser_idx]
        chaser_pose = np.array([chaser.px, chaser.py, chaser.theta], dtype=np.float32)
        num_line = self.lines.shape[0]
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        # Include robot + other active chasers (excluding self and inactive)
        num_active_other_chasers = sum(1 for j in range(self.chaser_num) if j != chaser_idx and self.chaser_active[j])
        num_circles_total = num_circle_human + num_circle_obstacle + 1 + num_active_other_chasers
        InitializeEnv(num_line, num_circles_total, self.n_laser, self.laser_angle_resolute)
        
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
        
        # Add robot as observable circle (for SAC chasers to see and chase)
        circle_idx = num_circle_human + num_circle_obstacle
        set_circles(3 * circle_idx    , self.robot.px)
        set_circles(3 * circle_idx + 1, self.robot.py)
        set_circles(3 * circle_idx + 2, self.robot.radius)
        
        # Add other active chasers as observable circles (excluding self and inactive)
        for j, other_chaser in enumerate(self.chasers):
            if j != chaser_idx and self.chaser_active[j]:
                circle_idx += 1
                set_circles(3 * circle_idx    , other_chaser.px)
                set_circles(3 * circle_idx + 1, other_chaser.py)
                set_circles(3 * circle_idx + 2, other_chaser.radius)
        
        set_robot_pose(chaser_pose[0], chaser_pose[1], chaser_pose[2])
        cal_laser()
        
        if layer == 0:
            self.chaser_scan_intersections[chaser_idx] = np.zeros((self.n_laser, 2, 2), dtype=np.float32)
        
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            if layer == 0:
                ### used for visualization
                self.chaser_scan_intersections[chaser_idx][i, 0, 0] = chaser.px
                self.chaser_scan_intersections[chaser_idx][i, 0, 1] = chaser.py
                self.chaser_scan_intersections[chaser_idx][i, 1, 0] = get_scan_line(4 * i + 2)
                self.chaser_scan_intersections[chaser_idx][i, 1, 1] = get_scan_line(4 * i + 3)
        
        if layer == 0:
            self.chaser_scan_currents[chaser_idx] = np.clip(scan, self.laser_min_range, self.laser_max_range)
        elif layer == 1:
            self.chaser_scan_current_layers[chaser_idx] = np.clip(scan, self.laser_min_range, self.laser_max_range)
        
        # Update backward compatibility references
        if chaser_idx == 0:
            self.chaser_scan_intersection = self.chaser_scan_intersections[0]
            if layer == 0:
                self.chaser_scan_current = self.chaser_scan_currents[0]
            elif layer == 1:
                self.chaser_scan_current_layer = self.chaser_scan_current_layers[0]
        
        ReleaseEnv()

        
    def get_lidar(self, layer):
        """
        Get lidar scan for the robot. Includes all active chasers as observable circles.
        """
        scan = np.zeros(self.n_laser, dtype=np.float32)
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        num_line = self.lines.shape[0]
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        # Include all active chasers as observable circles
        num_active_chasers = sum(1 for j in range(self.chaser_num) if self.chaser_active[j])
        InitializeEnv(num_line, num_circle_human + num_circle_obstacle + num_active_chasers, self.n_laser, self.laser_angle_resolute)
        
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
        
        # Add all active chasers as observable circles
        circle_idx = num_circle_human + num_circle_obstacle
        for j, chaser in enumerate(self.chasers):
            if self.chaser_active[j]:
                set_circles(3 * circle_idx    , chaser.px)
                set_circles(3 * circle_idx + 1, chaser.py)
                set_circles(3 * circle_idx + 2, chaser.radius)
                circle_idx += 1
        
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
        
        if layer == 0:
            self.scan_current = np.clip(scan, self.laser_min_range, self.laser_max_range)
        elif layer == 1:
            self.scan_current_layer = np.clip(scan, self.laser_min_range, self.laser_max_range)
        
        ReleaseEnv()

    def get_chaser_frame(self, chaser_idx=0):
        """
        Get frame representation from lidar scan for a specific chaser.
        
        Args:
            chaser_idx (int): Index of the chaser requesting frame
        """
        self.chaser_single_frames[chaser_idx] = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.get_chaser_lidar(0, chaser_idx)
        
        for i in range(self.n_laser):
            if self.chaser_scan_currents[chaser_idx][i] == self.laser_max_range:
                continue
            j = int(i / self.theta_resolution)
            if j >= self.image_size:
                j = self.image_size - 1
            k = int(self.chaser_scan_currents[chaser_idx][i] / self.r_resolution)
            if k >= self.image_size:
                k = self.image_size - 1
            self.chaser_single_frames[chaser_idx][0, j, k] = 255

        self.get_chaser_lidar(1, chaser_idx)
        for i in range(self.n_laser):
            if self.chaser_scan_current_layers[chaser_idx][i] == self.laser_max_range:
                continue
            j = int(i / self.theta_resolution)
            if j >= self.image_size:
                j = self.image_size - 1
            k = int(self.chaser_scan_current_layers[chaser_idx][i] / self.r_resolution)
            if k >= self.image_size:
                k = self.image_size - 1
            if self.chaser_single_frames[chaser_idx][0, j, k] != 255:
                self.chaser_single_frames[chaser_idx][0, j, k] = 127
        
        # Update backward compatibility reference
        if chaser_idx == 0:
            self.chaser_single_frame = self.chaser_single_frames[0]
        

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
        """
        Check if robot collides with humans, static obstacles, any chaser, or boundary.
        """
        # robot collision with humans
        for i in range(self.human_num):
            dis = hypot(self.robot.px - self.humans[i].px, self.robot.py - self.humans[i].py)
            emotion_idx = self.human_emotions.index(self.humans[i].emotion)
            if dis < self.robot.radius + self.humans[i].radius + layer * self.layer_len[emotion_idx]:
                return True
        
        # robot collision with static obstacles
        for i in range(self.static_obstacle_num):
            dis = hypot(self.robot.px - self.static_obstacles[i, 0], self.robot.py - self.static_obstacles[i, 1])
            if dis < self.robot.radius + self.static_obstacles[i, 2] + layer * self.obstacle_layer_len:
                return True
        
        # robot collision with any active chaser
        for j, chaser in enumerate(self.chasers):
            if self.chaser_active[j] and hypot(chaser.px - self.robot.px, chaser.py - self.robot.py) < chaser.radius + self.robot.radius:
                return True
        
        # robot collision with boundary
        if self.robot.px < -self.x_range + self.robot.radius or self.robot.px > self.x_range - self.robot.radius or \
            self.robot.py < -self.y_range + self.robot.radius or self.robot.py > self.y_range - self.robot.radius:
            return True
        
        return False
    
    def is_chaser_collision(self, layer, chaser_idx=0):
        """
        Check if a specific chaser collides with humans, static obstacles, other chasers, or boundary.
        Note: Does NOT check collision with robot (chasers need to chase robot, not avoid it).
        
        Args:
            layer (int): Layer index for collision checking
            chaser_idx (int): Index of the chaser to check
        """
        chaser = self.chasers[chaser_idx]
        
        # chaser collision with humans
        for i in range(self.human_num):
            dis = hypot(chaser.px - self.humans[i].px, chaser.py - self.humans[i].py)
            emotion_idx = self.human_emotions.index(self.humans[i].emotion)
            if dis < chaser.radius + self.humans[i].radius + layer * self.layer_len[emotion_idx]:
                return True
        
        # chaser collision with static obstacles
        for i in range(self.static_obstacle_num):
            dis = hypot(chaser.px - self.static_obstacles[i, 0], chaser.py - self.static_obstacles[i, 1])
            if dis < chaser.radius + self.static_obstacles[i, 2] + layer * self.obstacle_layer_len:
                return True
        
        # chaser collision with other active chasers (excluding self and inactive)
        for j, other_chaser in enumerate(self.chasers):
            if j != chaser_idx and self.chaser_active[j]:
                dis = hypot(chaser.px - other_chaser.px, chaser.py - other_chaser.py)
                if dis < chaser.radius + other_chaser.radius:
                    return True
        
        # chaser collision with boundary
        if chaser.px < -self.x_range + chaser.radius or chaser.px > self.x_range - chaser.radius or \
            chaser.py < -self.y_range + chaser.radius or chaser.py > self.y_range - chaser.radius:
            return True
        
        return False
    
    def get_chaser_chaser_collisions(self):
        """
        Returns a list of (i, j) tuples where active chaser i and active chaser j have collided.
        Only returns each collision pair once (i < j).
        Only checks collisions between active chasers.
        """
        collisions = []
        for i in range(self.chaser_num):
            if not self.chaser_active[i]:
                continue
            for j in range(i+1, self.chaser_num):
                if not self.chaser_active[j]:
                    continue
                dis = hypot(self.chasers[i].px - self.chasers[j].px, 
                           self.chasers[i].py - self.chasers[j].py)
                if dis < self.chasers[i].radius + self.chasers[j].radius:
                    collisions.append((i, j))
        return collisions

    def velocity_obstacle(self):
        for human in self.humans:
            relative_position = np.array([human.px - self.robot.px, human.py - self.robot.py])
            relative_velocity = np.array([human.vx - self.robot.vx, human.vy - self.robot.vy])
            distance = norm(relative_position)
            combined_radius = self.robot.radius + human.radius
            relative_speed = norm(relative_velocity)
            if relative_speed == 0:
                continue
            time_to_collision = distance / relative_speed
            if time_to_collision < self.vo_threshold:
                angle_to_human = atan2(relative_position[1], relative_position[0])
                angle_of_velocity = atan2(relative_velocity[1], relative_velocity[0])
                sigma = atan2(distance, combined_radius)
                if fabs(angle_to_human - angle_of_velocity) < sigma:
                    return True
        return False
    def velocity_obstacle(self):
        for human in self.humans:
            relative_position = np.array([human.px - self.robot.px, human.py - self.robot.py])
            relative_velocity = np.array([human.vx - self.robot.vx, human.vy - self.robot.vy])
            distance = norm(relative_position)
            combined_radius = self.robot.radius + human.radius
            relative_speed = norm(relative_velocity)
            if relative_speed == 0:
                continue
            time_to_collision = distance / relative_speed
            if time_to_collision < self.vo_threshold:
                angle_to_human = atan2(relative_position[1], relative_position[0])
                angle_of_velocity = atan2(relative_velocity[1], relative_velocity[0])
                sigma = atan2(distance, combined_radius)
                if fabs(angle_to_human - angle_of_velocity) < sigma:
                    return True
        return False
    
    def chaser_velocity_obstacle(self):
        for human in self.humans:
            relative_position = np.array([human.px - self.chaser.px, human.py - self.chaser.py])
            relative_velocity = np.array([human.vx - self.chaser.vx, human.vy - self.chaser.vy])
            distance = norm(relative_position)
            combined_radius = self.chaser.radius + human.radius
            relative_speed = norm(relative_velocity)
            if relative_speed == 0:
                continue
            time_to_collision = distance / relative_speed
            if time_to_collision < self.vo_threshold:
                angle_to_human = atan2(relative_position[1], relative_position[0])
                angle_of_velocity = atan2(relative_velocity[1], relative_velocity[0])
                sigma = atan2(distance, combined_radius)
                if fabs(angle_to_human - angle_of_velocity) < sigma:
                    return True
        return False
    
    def construct_chaser_obs(self, chaser_idx=0):
        """
        Construct observation for a specific chaser (for SAC chasers).
        Includes humans, static obstacles, robot, and other chasers.
        
        Args:
            chaser_idx (int): Index of the chaser requesting observation
        
        Note: For SAC chasers - includes robot so they can see and chase it
        """
        chaser = self.chasers[chaser_idx]
        # px, py, vx, vy - include robot + other active chasers (excluding self)
        num_active_other_chasers = sum(1 for j in range(self.chaser_num) if j != chaser_idx and self.chaser_active[j])
        num_entities = self.human_num + self.static_obstacle_num + 1 + num_active_other_chasers
        self.humans_state_in_robot_frame = np.zeros((num_entities, 4), dtype=np.float32)
        ped_pos_map = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
        c_theta = cos(chaser.theta)
        s_theta = sin(chaser.theta)
        
        # Add humans and static obstacles
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
            dx = px_world - chaser.px
            dy = py_world - chaser.py
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
        
        # Add robot as observable (for SAC chasers to see and chase)
        idx = self.human_num + self.static_obstacle_num
        dx = self.robot.px - chaser.px
        dy = self.robot.py - chaser.py
        px_local = dy * s_theta + dx * c_theta
        py_local = dy * c_theta - dx * s_theta
        vx_local = self.robot.vy * s_theta + self.robot.vx * c_theta
        vy_local = self.robot.vy * c_theta - self.robot.vx * s_theta
        self.humans_state_in_robot_frame[idx, :] = np.array([px_local, py_local, vx_local, vy_local], dtype=np.float32)
        if fabs(px_local) < self.laser_max_range and fabs(py_local) < self.laser_max_range:
            r = int((self.laser_max_range - py_local) / self.range_resolution)
            c = int((px_local + self.laser_max_range) / self.range_resolution)
            if c < self.image_size and r < self.image_size and c >= 0 and r >= 0:
                ped_pos_map[0, r, c] = vx_local
                ped_pos_map[1, r, c] = vy_local
        
        # Add other active chasers as observables (excluding self and inactive)
        for j, other_chaser in enumerate(self.chasers):
            if j != chaser_idx and self.chaser_active[j]:
                idx += 1
                dx = other_chaser.px - chaser.px
                dy = other_chaser.py - chaser.py
                px_local = dy * s_theta + dx * c_theta
                py_local = dy * c_theta - dx * s_theta
                vx_local = other_chaser.vy * s_theta + other_chaser.vx * c_theta
                vy_local = other_chaser.vy * c_theta - other_chaser.vx * s_theta
                self.humans_state_in_robot_frame[idx, :] = np.array([px_local, py_local, vx_local, vy_local], dtype=np.float32)
                if fabs(px_local) < self.laser_max_range and fabs(py_local) < self.laser_max_range:
                    r = int((self.laser_max_range - py_local) / self.range_resolution)
                    c = int((px_local + self.laser_max_range) / self.range_resolution)
                    if c < self.image_size and r < self.image_size and c >= 0 and r >= 0:
                        ped_pos_map[0, r, c] = vx_local
                        ped_pos_map[1, r, c] = vy_local

        lidar_map = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        lidar_avg = np.zeros((2 * self.frame_stack, self.image_size), dtype=np.float32)
        # Use per-chaser lidar frames
        lidar_frames = np.concatenate(list(self.chaser_lidar_frames_list[chaser_idx]), axis=0)
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

    def construct_obs(self):
        """
        Construct observation for the robot. Includes all active chasers as moving obstacles.
        """
        # px, py, vx, vy - include all active chasers
        num_active_chasers = sum(1 for j in range(self.chaser_num) if self.chaser_active[j])
        self.humans_state_in_robot_frame = np.zeros((self.human_num + self.static_obstacle_num + num_active_chasers, 4), dtype=np.float32)
        ped_pos_map = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
        c_theta = cos(self.robot.theta)
        s_theta = sin(self.robot.theta)
        
        for i in range (self.human_num + self.static_obstacle_num):
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
        
        # Add all active chasers as moving obstacles
        idx = self.human_num + self.static_obstacle_num
        for j, chaser in enumerate(self.chasers):
            if self.chaser_active[j]:
                dx = chaser.px - self.robot.px
                dy = chaser.py - self.robot.py
                px_local = dy * s_theta + dx * c_theta
                py_local = dy * c_theta - dx * s_theta
                vx_local = chaser.vy * s_theta + chaser.vx * c_theta
                vy_local = chaser.vy * c_theta - chaser.vx * s_theta
                self.humans_state_in_robot_frame[idx, :] = np.array([px_local, py_local, vx_local, vy_local], dtype=np.float32)
                if fabs(px_local) < self.laser_max_range and fabs(py_local) < self.laser_max_range:
                    r = int((self.laser_max_range - py_local) / self.range_resolution)
                    c = int((px_local + self.laser_max_range) / self.range_resolution)
                    if c < self.image_size and r < self.image_size and c >= 0 and r >= 0:
                        ped_pos_map[0, r, c] = vx_local
                        ped_pos_map[1, r, c] = vy_local
                idx += 1

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
    
    def chaser_reset(self):
        """
        Reset positions and velocities for all chasers.
        Ensures no chaser-chaser, chaser-robot, chaser-human, or chaser-obstacle collisions.
        """
        for chaser_idx in range(self.chaser_num):
            chaser = self.chasers[chaser_idx]
            check_flag = 0
            max_attempts = 1000
            attempt = 0
            
            while check_flag < 4 and attempt < max_attempts:
                attempt += 1
                check_flag = 0
                chaser.px = np.random.uniform(-self.x_range+self.robot.radius, self.x_range-self.robot.radius, 1)
                chaser.py = np.random.uniform(-self.y_range+self.robot.radius, self.y_range-self.robot.radius, 1)

                #check distance from robot goal
                if hypot(chaser.px - self.robot.gx, chaser.py - self.robot.gy) > 1.0:
                    check_flag += 1
                
                # Check distance from robot (must be > 2.0)
                if hypot(chaser.px - self.robot.px, chaser.py - self.robot.py) > 5.0:
                    check_flag += 1
                
                # Check distance from humans (must be > 1.0)
                collision_with_human = False
                for i in range(self.human_num):
                    if hypot(chaser.px - self.humans[i].px, chaser.py - self.humans[i].py) < 1.0:
                        collision_with_human = True
                        break
                if not collision_with_human:
                    check_flag += 1
                
                # Check distance from static obstacles (must be > 1.0)
                collision_with_obstacle = False
                for i in range(self.static_obstacle_num):
                    if hypot(chaser.px - self.static_obstacles[i, 0], chaser.py - self.static_obstacles[i, 1]) < 1.0:
                        collision_with_obstacle = True
                        break
                if not collision_with_obstacle:
                    check_flag += 1
                
                # Check distance from other chasers already placed (must be > 1.0)
                if check_flag == 4:  # Only check other chasers if other conditions passed
                    collision_with_other_chaser = False
                    for j in range(chaser_idx):  # Check against previously placed chasers
                        if hypot(chaser.px - self.chasers[j].px, chaser.py - self.chasers[j].py) < 3.0:
                            collision_with_other_chaser = True
                            break
                    if collision_with_other_chaser:
                        check_flag = 0
            
            if attempt >= max_attempts:
                print(f"Warning: Could not find safe position for chaser {chaser_idx} after {max_attempts} attempts")
            
            chaser.theta = np.random.uniform(-np.pi, np.pi, 1)
            chaser.vx = 0.0
            chaser.vy = 0.0
            chaser.gx = self.robot.px
            chaser.gy = self.robot.py
            
            # Reset Ackermann-specific parameters if needed
            if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                chaser.current_steering_angle = 0.0
                chaser.current_velocity = 0.0
                chaser.current_angular_velocity = 0.0
        
        # Update backward compatibility reference
        if self.chaser_num > 0:
            self.chaser = self.chasers[0]
        
        return
    
    def reset_robot_goal(self):
        while True:
            random_goal_x = np.random.uniform(-self.x_range + self.robot.radius, self.x_range - self.robot.radius, 1)
            random_goal_y = np.random.uniform(-self.y_range + self.robot.radius, self.y_range - self.robot.radius, 1)
            if hypot(random_goal_x - self.robot.px, random_goal_y - self.robot.py) < 4.0 or \
                hypot(random_goal_x - self.robot.px, random_goal_y - self.robot.py) < hypot(random_goal_x - self.chaser.px, random_goal_y - self.chaser.py):
                continue
            else:
                for i in range(self.human_num):
                    if hypot(random_goal_x - self.humans[i].px, random_goal_y - self.humans[i].py) < 2*self.robot.radius:
                        continue
                    else:
                        for j in range(self.static_obstacle_num):
                            if hypot(random_goal_x - self.static_obstacles[j, 0], random_goal_y - self.static_obstacles[j, 1]) < 2*self.robot.radius:
                                continue
                            else:
                                self.robot.gx = random_goal_x
                                self.robot.gy = random_goal_y
                                return
            
    def cal_dwa_action(self):
        """Calculate robot action using Dynamic Window Approach (DWA)"""
        if self.action_choices is None:
            print("Warning: action_choices is None, returning zero action")
            return np.zeros(2)
        
        action_cost = 99999.9
        action_dwa = np.zeros(2)
        
        for i in range(self.action_choices.shape[0]):
            robot_vel = self.action_choices[i]
            
            robot_x = self.robot.px
            robot_y = self.robot.py
            robot_theta = self.robot.theta
            collision = False
            dis_human_and_obstacle = 99999.9
            
            # Simulate robot trajectory for this action
            for j in range(self.dwa_horizon):
                robot_theta = robot_theta + robot_vel[1] * self.time_step
                if robot_theta > np.pi:
                    robot_theta -= (2.0 * np.pi)
                elif robot_theta < -np.pi:
                    robot_theta += (2.0 * np.pi)
                
                # Update robot position - differential drive model
                robot_x = robot_x + robot_vel[0] * self.time_step * cos(robot_theta)
                robot_y = robot_y + robot_vel[0] * self.time_step * sin(robot_theta)

                # Check collision with humans
                for k in range(self.human_num):
                    human_x = self.humans[k].px + (j + 1) * self.humans[k].vx * self.time_step
                    human_y = self.humans[k].py + (j + 1) * self.humans[k].vy * self.time_step
                    dis_human_temp = hypot(human_x - robot_x, human_y - robot_y)
                    if dis_human_temp <= self.humans[k].radius + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_human_temp)

                if collision:
                    break

                # Check collision with static obstacles
                for k in range(self.static_obstacle_num):
                    obstacle_x = self.static_obstacles[k, 0]
                    obstacle_y = self.static_obstacles[k, 1]
                    dis_obstacle_temp = hypot(obstacle_x - robot_x, obstacle_y - robot_y)
                    if dis_obstacle_temp <= self.static_obstacles[k, 2] + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_obstacle_temp)

                if collision:
                    break
                
                # Check collision with chaser (treat as moving obstacle)
                chaser_x = self.chaser.px + (j + 1) * self.chaser.vx * self.time_step
                chaser_y = self.chaser.py + (j + 1) * self.chaser.vy * self.time_step
                dis_chaser_temp = hypot(chaser_x - robot_x, chaser_y - robot_y)
                if dis_chaser_temp <= self.chaser.radius + self.robot.radius:
                    collision = True
                    break
                dis_human_and_obstacle = min(dis_human_and_obstacle, dis_chaser_temp)

                if collision:
                    break

            if collision:
                continue
                
            # Calculate cost for this action
            dis_goal = hypot(self.robot.gx - robot_x, self.robot.gy - robot_y)
            action_cost_temp = (self.dwa_obstacle_cost_weight / (dis_human_and_obstacle + self.dwa_safety_margin) + 
                               dis_goal * self.dwa_goal_cost_weight)
            
            if action_cost > action_cost_temp:
                action_cost = action_cost_temp
                action_dwa = robot_vel
        
        return action_dwa
    
    def cal_orca_action(self):
        """Calculate robot action using ORCA (Optimal Reciprocal Collision Avoidance)"""
        # Create observable states for all agents
        ob = [human.get_observable_state() for human in self.humans]
        
        # Add static obstacles as observable states
        for k in range(self.static_obstacle_num):
            ob.append(ObservableState(
                self.static_obstacles[k, 0], 
                self.static_obstacles[k, 1], 
                0.0, 0.0, self.static_obstacles[k, 2])
            )
        
        # Add chaser as moving obstacle
        ob.append(ObservableState(
            self.chaser.px,
            self.chaser.py,
            self.chaser.vx,
            self.chaser.vy,
            self.chaser.radius
        ))
        
        # Create joint state for robot
        state = self.robot.get_joint_state(ob)
        
        # Get ORCA velocity
        action_temp = self.robot_orca_policy.predict(state)
        
        # Convert to linear and angular velocity
        linear_v = hypot(action_temp[0], action_temp[1])
        target_theta = atan2(action_temp[1], action_temp[0])
        delta_theta = target_theta - self.robot.theta
        if delta_theta > np.pi:
            delta_theta -= (2.0 * np.pi)
        elif delta_theta < -np.pi:
            delta_theta += (2.0 * np.pi)
        angular_v = delta_theta / self.time_step
        
        action = np.array([linear_v, angular_v], dtype=np.float32)
        
        # Clip action to stay within defined action range bounds
        action[0] = np.clip(action[0], self.action_range[0, 0], self.action_range[1, 0])  # linear velocity
        action[1] = np.clip(action[1], self.action_range[0, 1], self.action_range[1, 1])  # angular velocity
        
        return action
    
    def cal_rgl_action(self):
        """Calculate robot action using RGL (uses ORCA with coordinate-based observations)"""
        # Create coordinate-based observation
        ob = self.get_coordinate_observation()
        
        # Create joint state for robot
        state = self.robot.get_joint_state(ob)
        
        # Use ORCA policy (same as RGL environment does)
        action_temp = self.robot_orca_policy.predict(state)
        
        # Convert to linear and angular velocity
        linear_v = hypot(action_temp[0], action_temp[1])
        target_theta = atan2(action_temp[1], action_temp[0])
        delta_theta = target_theta - self.robot.theta
        if delta_theta > np.pi:
            delta_theta -= (2.0 * np.pi)
        elif delta_theta < -np.pi:
            delta_theta += (2.0 * np.pi)
        angular_v = delta_theta / self.time_step
        
        action = np.array([linear_v, angular_v], dtype=np.float32)
        
        return action
    
    def get_coordinate_observation(self):
        """Get coordinate-based observation compatible with RGL environment"""
        ob = []
        
        # Add humans with padding to human_num_max
        for k in range(self.human_num):
            ob.append(self.humans[k].get_observable_state())
        for i in range(self.human_num_max - self.human_num):
            ob.append(ObservableState(0.0, 0.0, 0.0, 0.0, 0.0))
        
        # Add static obstacles with padding to static_obstacle_num_max
        for k in range(self.static_obstacle_num):
            ob.append(ObservableState(self.static_obstacles[k, 0], 
                                      self.static_obstacles[k, 1], 
                                      0.0, 0.0, 
                                      self.static_obstacles[k, 2]))
        for i in range(self.static_obstacle_num_max - self.static_obstacle_num):
            ob.append(ObservableState(0.0, 0.0, 0.0, 0.0, 0.0))
        
        # Add chaser as additional dynamic obstacle
        ob.append(ObservableState(
            self.chaser.px,
            self.chaser.py,
            self.chaser.vx,
            self.chaser.vy,
            self.chaser.radius
        ))
        
        return ob

    
    def step(self, action, chaser_actions, eval=False, save_data=False):
        """
        Step the environment forward by one timestep.
        
        Args:
            action: Robot action (vx, vy) or (v, omega)
            chaser_actions: List of chaser actions, one per chaser [(vx, vy), ...] or numpy array of shape (chaser_num, 2)
            eval: Whether in evaluation mode
            save_data: Whether to log state data
        
        Returns:
            For non-RGL modes:
                obs_image, robot_goal_emotion_state, reward, done, info,
                closest_human_distance,
                chaser_obs_images (list), chaser_goal_emotion_states (list),
                chaser_rewards (list), chaser_infos (list)
        """
        # Check robot policy mode and compute action if not DRL
        # For RGL in eval/test mode, use provided action from RGL agent instead of computing ORCA action
        if self.robot_policy_mode == 'dwa':
            action = self.cal_dwa_action()
        elif self.robot_policy_mode == 'orca':
            action = self.cal_orca_action()
        elif self.robot_policy_mode == 'rgl':
            # In test/eval mode, use the provided action from RGL agent
            # Otherwise, compute action using ORCA (for training compatibility)
            if eval and np.linalg.norm(action) > 1e-6:  # If action is provided and non-zero
                pass  # Use the provided action
            else:
                action = self.cal_rgl_action()
        # For 'drl' mode, use the provided action as-is
        
        # Convert chaser_actions to list of numpy arrays if needed
        if isinstance(chaser_actions, np.ndarray):
            chaser_actions = [chaser_actions[i] for i in range(self.chaser_num)]
        
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
        chaser_action_copies = [np.array([chaser_action[0], chaser_action[1]]) for chaser_action in chaser_actions]
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
            
            # Compute new poses for all chasers
            chaser_new_poses = []  # List of (x, y, theta, ...) tuples for each chaser
            for chaser_idx, chaser_action in enumerate(chaser_actions):
                chaser = self.chasers[chaser_idx]
                if not self.chaser_active[chaser_idx]:
                    # Inactive chaser doesn't move (removed after collision)
                    if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                        chaser_new_poses.append((chaser.px, chaser.py, chaser.theta, 0.0, 0.0, 0.0))
                    else:
                        chaser_new_poses.append((chaser.px, chaser.py, chaser.theta))
                else:
                    if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                        cx, cy, ctheta, c_vel, c_steer, c_thetadot = chaser.compute_pose(chaser_action)
                        chaser_new_poses.append((cx, cy, ctheta, c_vel, c_steer, c_thetadot))
                    else:
                        ctheta = chaser.theta + chaser_action[1] * self.time_step
                        if ctheta > np.pi:
                            ctheta -= (2.0 * np.pi)
                        elif ctheta < -np.pi:
                            ctheta += (2.0 * np.pi)
                        if self.chaser_model == 'differential':
                            cx = chaser.px + chaser_action[0] * self.time_step * cos(ctheta)
                            cy = chaser.py + chaser_action[0] * self.time_step * sin(ctheta)
                        else:
                            cx = chaser.px
                            cy = chaser.py
                        chaser_new_poses.append((cx, cy, ctheta))
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
        
        # Update chaser action copies
        for chaser_idx, chaser in enumerate(self.chasers):
            if len(chaser_new_poses[chaser_idx]) >= 3:
                cx, cy = chaser_new_poses[chaser_idx][0], chaser_new_poses[chaser_idx][1]
                chaser_action_copies[chaser_idx][0] = hypot(cy - chaser.py, cx - chaser.px) / self.time_step
            
        # update robot states
        if self.robot_test_model == 'ackermann' or self.robot_model == 'ackermann':
            self.robot.update_states(robot_x, robot_y, robot_theta, action_copy, 
                                   forward_velocity=actual_velocity,
                                   steering_angle=actual_steering, 
                                   theta_dot=actual_theta_dot)
        else:
            self.robot.update_states(robot_x, robot_y, robot_theta, action_copy, differential=True)
        
        # Update all chaser states
        for chaser_idx, chaser in enumerate(self.chasers):
            if self.chaser_active[chaser_idx]:
                pose = chaser_new_poses[chaser_idx]
                if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                    chaser.update_states(pose[0], pose[1], pose[2], chaser_action_copies[chaser_idx],
                                        forward_velocity=pose[3],
                                        steering_angle=pose[4],
                                        theta_dot=pose[5])
                else:
                    chaser.update_states(pose[0], pose[1], pose[2], chaser_action_copies[chaser_idx], differential=True)
            # Update chaser goal to always chase robot (even for inactive chasers, for consistency)
            chaser.gx = self.robot.px
            chaser.gy = self.robot.py
        
        # Update backward compatibility references
        if self.chaser_num > 0:
            self.chaser = self.chasers[0]
            self.chaser_action_last = chaser_action_copies[0]
       
        for i in range(self.human_num):
            self.humans[i].update_states(human_actions[i])

        # get new laser scan and grid map
        self.get_frame() 
        
        # Get frames for all chasers
        for chaser_idx in range(self.chaser_num):
            self.get_chaser_frame(chaser_idx)
            self.chaser_frames_list[chaser_idx].append(self.chaser_single_frames[chaser_idx])
            self.chaser_lidar_frames_list[chaser_idx].append(self.chaser_scan_currents[chaser_idx] / self.laser_max_range)
        
        self.frames.append(self.single_frame)
        self.lidar_frames.append(self.scan_current / self.laser_max_range)
        
        # Update backward compatibility
        if self.chaser_num > 0:
            self.chaser_frames = self.chaser_frames_list[0]
            self.chaser_lidar_frames = self.chaser_lidar_frames_list[0]
            self.chaser_single_frame = self.chaser_single_frames[0]
            self.chaser_scan_current = self.chaser_scan_currents[0]
        
        assert len(self.frames) == self.frame_stack
        assert len(self.lidar_frames) == self.frame_stack
        lidar_image = np.concatenate(list(self.frames), axis=0)
        
        obs_image = self.construct_obs()
        
        # Construct observations for all chasers
        chaser_obs_images = []
        for chaser_idx in range(self.chaser_num):
            chaser_obs_images.append(self.construct_chaser_obs(chaser_idx))
        
        self.global_time += self.time_step
        
        # Robot reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        reaching_goal = goal_dist < (self.robot.radius - 0.1)

        # --- Per-chaser collision/reward/info computation (check catches BEFORE robot collision) ---
        chaser_rewards = []
        chaser_infos = []
        chaser_collisions = []
        chaser_collision_layers = []
        chaser_catches = []
        chaser_poses = []  # Store chaser poses for later use
        
        # First, check if any chaser caught the robot (before checking robot collisions)
        # This prevents robot collision from being set when a chaser catches it
        any_chaser_caught = False
        for chaser_idx, chaser in enumerate(self.chasers):
            # Check collisions for this chaser
            chaser_collision = self.is_chaser_collision(0, chaser_idx)
            chaser_collision_layer = self.is_chaser_collision(1, chaser_idx)
            chaser_collisions.append(chaser_collision)
            chaser_collision_layers.append(chaser_collision_layer)
            
            # Check if chaser caught robot
            # Only active (not removed), non-collided chasers can catch
            pose = chaser_new_poses[chaser_idx]
            chaser_x, chaser_y = pose[0], pose[1]
            chaser_poses.append((chaser_x, chaser_y))
            chaser_catch = (hypot(chaser_x - robot_x, chaser_y - robot_y) < self.robot.radius + chaser.radius and
                           self.chaser_active[chaser_idx] and
                           not chaser_collision)
            chaser_catches.append(chaser_catch)
            if chaser_catch:
                any_chaser_caught = True

        # Robot collision detection
        # If a chaser caught the robot, don't count it as a robot collision
        collision = self.is_collision(0) and not any_chaser_caught
        collision_layer = self.is_collision(1)
        
        # Compute robot reward
        dis_goal_reward = self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        self.goal_distance_last = goal_dist
        
        if self.use_angular:
            angular_reward = fabs(action[1]) * self.angular_penalty if fabs(action[1]) > 1.0 else 0.0
        else:
            angular_reward = 0.0
        
        self.action_last = action

        if self.velocity_obstacle():
            vo_penalty = self.vo_penalty
        else:
            vo_penalty = 0.0
            
        reward = collision_layer * self.collision_layer_penalty + dis_goal_reward + angular_reward + vo_penalty
        
        if collision and self.training_object == 'robot':
            reward = self.collision_penalty
            done = True
            info = Collision()
            if any_chaser_caught:
                print("implementation error: robot collision and chaser catch at the same time")
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
        
        for chaser_idx, chaser in enumerate(self.chasers):
            chaser_collision = chaser_collisions[chaser_idx]
            chaser_collision_layer = chaser_collision_layers[chaser_idx]
            chaser_catch = chaser_catches[chaser_idx]
            chaser_x, chaser_y = chaser_poses[chaser_idx]
            
            # Compute chaser goal distance and reward
            chaser_goal_dist = hypot(chaser_x - self.robot.gx, chaser_y - self.robot.gy)
            chaser_goal_reward = self.goal_distance_factor * (self.chaser_distance_lasts[chaser_idx] - chaser_goal_dist)
            self.chaser_distance_lasts[chaser_idx] = chaser_goal_dist
            
            # Compute chaser angular penalty
            if self.use_angular:
                chaser_angular_reward = fabs(chaser_actions[chaser_idx][1]) * self.angular_penalty if fabs(chaser_actions[chaser_idx][1]) > 1.0 else 0.0
            else:
                chaser_angular_reward = 0.0
            
            # Placeholder for velocity obstacle check (would need per-chaser implementation)
            chaser_vo_penalty = 0.0
            
            chaser_reward = chaser_collision_layer * self.collision_layer_penalty + chaser_goal_reward + chaser_angular_reward + chaser_vo_penalty
            
            # Determine chaser outcome
            if chaser_collision:
                chaser_reward = self.collision_penalty
                chaser_info = Collision()
                if self.training_object == 'chaser':
                    done = True
            elif chaser_catch:
                chaser_reward = self.success_reward
                if not done:
                    reward = self.collision_penalty
                    done = True
                chaser_info = ReachGoal()
            elif chaser_collision_layer:
                chaser_info = Danger(0.1)
            else:
                chaser_info = Nothing()
            
            chaser_rewards.append(chaser_reward)
            chaser_infos.append(chaser_info)
            
            # Remove collided chasers from environment
            if chaser_collision:
                if self.training_object == 'robot':
                    # Remove (deactivate) collided chaser in robot training mode
                    self.chaser_active[chaser_idx] = False
                # If chaser training, episode ends (done already set to True above)
            
            # Update per-chaser action last
            self.chaser_action_lasts[chaser_idx] = chaser_actions[chaser_idx]
        
        # Check chaser-chaser collisions and remove both chasers
        chaser_chaser_collisions = self.get_chaser_chaser_collisions()
        if len(chaser_chaser_collisions) > 0:
            if self.training_object == 'robot':
                # Remove both collided chasers
                for (i, j) in chaser_chaser_collisions:
                    self.chaser_active[i] = False
                    self.chaser_active[j] = False
            else:
                # End episode if chaser training
                done = True
        
        # Backward compatibility
        if self.chaser_num > 0:
            self.chaser_action_last = self.chaser_action_lasts[0]
            self.chaser_goal_distance_last = self.chaser_distance_lasts[0]

        # Compute closest distance to any pedestrian (humans only)
        closest_human_distance = inf
        for i in range(self.human_num):
            dis_center = hypot(self.robot.px - self.humans[i].px, self.robot.py - self.humans[i].py)
            if dis_center < closest_human_distance:
                closest_human_distance = dis_center
        if self.human_num == 0:
            closest_human_distance = inf

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

        # get the robot goal state observation
        robot_goal_emotion_state = np.array([r / self.square_width, t / np.pi, 
                                             self.action_last[0] / self.action_range[1, 0],
                                             self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
        
        # Compute goal states for all chasers
        chaser_goal_emotion_states = []
        for chaser_idx, chaser in enumerate(self.chasers):
            chaser_dx = self.robot.px - chaser.px
            chaser_dy = self.robot.py - chaser.py
            chaser_theta = chaser.theta
            chaser_y_rel = chaser_dy * cos(chaser_theta) - chaser_dx * sin(chaser_theta)
            chaser_x_rel = chaser_dy * sin(chaser_theta) + chaser_dx * cos(chaser_theta)
            chaser_r = hypot(chaser_x_rel, chaser_y_rel)
            chaser_t = atan2(chaser_y_rel, chaser_x_rel)

            chaser_goal_emotion_states.append(np.array([chaser_r / self.square_width, chaser_t / np.pi, 
                                                 self.chaser_action_lasts[chaser_idx][0] / self.chaser_action_range[1, 0],
                                                 self.chaser_action_lasts[chaser_idx][1] / self.chaser_action_range[1, 1]], dtype=np.float32))
        
        if save_data:
            self.global_step += 1
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, action[0], action[1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx[0], self.robot.gy[0]] if isinstance(self.robot.gx, np.ndarray) else [self.robot.gx, self.robot.gy], dtype=np.float32)
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

            # Log all chasers
            for chaser_idx, chaser in enumerate(self.chasers):
                self.log_env['chasers'][self.global_step, chaser_idx] = np.array([chaser.px, chaser.py, chaser_actions[chaser_idx][0], chaser_actions[chaser_idx][1], chaser.theta], dtype=np.float32)

        # Return appropriate observation format based on robot policy mode
        if self.robot_policy_mode == 'rgl':
            # For RGL mode, return coordinate-based state like RGL environment
            coordinate_obs = self.get_coordinate_observation()
            robot_state = self.robot.get_joint_state(coordinate_obs)
            return robot_state, reward, done, info, closest_human_distance, chaser_infos
        else:
            # For DRL, DWA, and ORCA modes, return image-based observations
            return obs_image, robot_goal_emotion_state, reward, done, info, closest_human_distance, chaser_obs_images, chaser_goal_emotion_states, chaser_rewards, chaser_infos
    
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
        if seed >= 0:
            np.random.seed(seed)
        self.generate_random_static_obstacle()
        self.generate_random_human_position()
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)

        if self.random_chaser_reset:
            self.chaser_reset()
        else:
            # Fixed position reset for all chasers
            for chaser_idx, chaser in enumerate(self.chasers):
                # Space chasers evenly around a circle
                angle = chaser_idx * (2 * np.pi / self.chaser_num)
                chaser.set(self.circle_radius * np.sin(angle), -self.circle_radius * np.cos(angle), 
                          self.robot.px, self.robot.py, 0.0, 0.0, 0.0)
                # Reset Ackermann-specific parameters for chaser if needed
                if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                    chaser.current_steering_angle = 0.0
                    chaser.current_velocity = 0.0
                    chaser.current_angular_velocity = 0.0
            # Update backward compatibility reference
            if self.chaser_num > 0:
                self.chaser = self.chasers[0]

        if self.random_robot_goal:
            self.reset_robot_goal()

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
            if self.robot_test_model == 'ackermann' or self.robot_model == 'ackermann':
                self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), 
                                       forward_velocity=0.0, steering_angle=0.0, theta_dot=0.0)
            else:
                self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), differential=True)
        
        self.goal_distance_last = self.robot.get_goal_distance()
        # Initialize chaser goal distances
        for chaser_idx, chaser in enumerate(self.chasers):
            self.chaser_distance_lasts[chaser_idx] = hypot(chaser.px - self.robot.gx, chaser.py - self.robot.gy)
        # Reset chaser active states (all chasers start active)
        self.chaser_active = [True] * self.chaser_num
        # Backward compatibility
        self.chaser_goal_distance_last = self.chaser_distance_lasts[0] if self.chaser_num > 0 else None

        # Initialize ORCA policy parameters if using ORCA or RGL
        if self.robot_policy_mode in ['orca', 'rgl']:
            self.robot_orca_policy.time_step = self.time_step
            self.robot_orca_policy.max_speed = self.robot.v_pref
            self.robot_orca_policy.radius = self.robot.radius
            self.robot_orca_policy.max_robot_speed = self.robot.v_pref

        # 3,5 save
        # np.random.seed(5)


        self.get_frame() 
        
        # Get frames for all chasers
        for chaser_idx in range(self.chaser_num):
            self.get_chaser_frame(chaser_idx)
        
        # Initialize robot frames
        for _ in range(self.frame_stack):
            self.frames.append(self.single_frame)
        
        # Initialize per-chaser frames
        for chaser_idx in range(self.chaser_num):
            for _ in range(self.frame_stack):
                self.chaser_frames_list[chaser_idx].append(self.chaser_single_frames[chaser_idx])
        
        # Update backward compatibility
        if self.chaser_num > 0:
            self.chaser_frames = self.chaser_frames_list[0]
            self.chaser_single_frame = self.chaser_single_frames[0]
        
        assert len(self.frames) == self.frame_stack
        lidar_image = np.concatenate(list(self.frames), axis=0)

        # Initialize robot lidar frames
        for _ in range(self.frame_stack):
            self.lidar_frames.append(self.scan_current / self.laser_max_range)
        
        # Initialize per-chaser lidar frames
        for chaser_idx in range(self.chaser_num):
            for _ in range(self.frame_stack):
                self.chaser_lidar_frames_list[chaser_idx].append(self.chaser_scan_currents[chaser_idx] / self.laser_max_range)
        
        # Update backward compatibility
        if self.chaser_num > 0:
            self.chaser_lidar_frames = self.chaser_lidar_frames_list[0]
        
        assert len(self.lidar_frames) == self.frame_stack
        
        obs_image = self.construct_obs()
        
        # Construct observations for all chasers
        chaser_obs_images = []
        for chaser_idx in range(self.chaser_num):
            chaser_obs_images.append(self.construct_chaser_obs(chaser_idx))

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
        
        # Compute goal states for all chasers
        chaser_goal_emotion_states = []
        for chaser_idx, chaser in enumerate(self.chasers):
            chaser_dx = self.robot.px - chaser.px
            chaser_dy = self.robot.py - chaser.py
            chaser_theta = chaser.theta
            chaser_y_rel = chaser_dy * cos(chaser_theta) - chaser_dx * sin(chaser_theta)
            chaser_x_rel = chaser_dy * sin(chaser_theta) + chaser_dx * cos(chaser_theta)
            chaser_r = hypot(chaser_x_rel, chaser_y_rel)
            chaser_t = atan2(chaser_y_rel, chaser_x_rel)

            chaser_goal_emotion_states.append(np.array([chaser_r / self.square_width, chaser_t / np.pi,
                                                    self.chaser_action_lasts[chaser_idx][0] / self.chaser_action_range[1, 0],
                                                    self.chaser_action_lasts[chaser_idx][1] / self.chaser_action_range[1, 1]], dtype=np.float32))
       
        if save_data:
            self.log_env['ypr'] = -100.0 * np.ones((self.max_episode_step + 1, 3), dtype=np.float32)
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            # Update chaser logging to handle multiple chasers
            self.log_env['chasers'] = -100.0 * np.ones((self.max_episode_step + 1, self.chaser_num, 5), dtype=np.float32)
            self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 4), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)
            self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx[0], self.robot.gy[0]] if isinstance(self.robot.gx, np.ndarray) else [self.robot.gx, self.robot.gy], dtype=np.float32)
            
            # Log all chasers
            for chaser_idx, chaser in enumerate(self.chasers):
                self.log_env['chasers'][self.global_step, chaser_idx] = np.array([chaser.px, chaser.py, 0.0, 0.0, chaser.theta], dtype=np.float32)
            
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

        return obs_image, robot_goal_emotion_state, chaser_obs_images, chaser_goal_emotion_states

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

    def get_full_state(self, chaser_idx=None):
        """
        Returns the full state of the environment for ORCA policy
        This follows the same pattern as how observations are provided for human agents
        
        Args:
            chaser_idx (int, optional): Index of chaser requesting observation. 
                                       If provided, other chasers (excluding self) are included.
                                       Robot is NOT included (ORCA must not avoid robot).
        """
        # Create observable states for all humans
        observation = []
        for human in self.humans:
            observation.append(ObservableState(
                human.px, human.py,
                human.vx, human.vy,
                human.radius
            ))
            
        # Add static obstacles as observable states
        for i in range(self.static_obstacle_num):
            observation.append(ObservableState(
                self.static_obstacles[i, 0],
                self.static_obstacles[i, 1],
                0.0, 0.0,
                self.static_obstacles[i, 2]
            ))
        
        # Add other active chasers if chaser_idx is provided (excluding self and inactive chasers)
        # Do NOT add robot - ORCA must not see robot as obstacle to maintain chasing behavior
        if chaser_idx is not None and hasattr(self, 'chasers'):
            for j, chaser in enumerate(self.chasers):
                if j != chaser_idx and self.chaser_active[j]:
                    observation.append(ObservableState(
                        chaser.px, chaser.py,
                        chaser.vx, chaser.vy,
                        chaser.radius
                    ))
        
        return observation

    def export_initial_state(self, filepath):
        """Export current initial state to NPZ file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Environment metadata
            environment_type = 'adv'
            timestamp = datetime.now().isoformat()
            
            # Robot state [px, py, gx, gy, vx, vy, theta, radius, v_pref]
            robot_state = np.array([
                self.robot.px, self.robot.py, self.robot.gx, self.robot.gy,
                self.robot.vx, self.robot.vy, self.robot.theta,
                self.robot.radius, self.robot.v_pref
            ], dtype=np.float32)
            
            # Chasers state [n_chasers x 9] -> [px, py, gx, gy, vx, vy, theta, radius, v_pref]
            chasers_data = []
            for chaser in self.chasers:
                chaser_data = [
                    chaser.px, chaser.py, chaser.gx, chaser.gy,
                    chaser.vx, chaser.vy, chaser.theta,
                    chaser.radius, chaser.v_pref
                ]
                chasers_data.append(chaser_data)
            chasers_state = np.array(chasers_data, dtype=np.float32)
            
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
                self.human_num, self.static_obstacle_num, self.chaser_num
            ], dtype=np.float32)
            
            # Additional adversarial-specific parameters
            adv_params = np.array([
                self.random_chaser_reset,
                self.random_robot_goal,
                self.training_object == 'robot'  # 1.0 for robot, 0.0 for chaser
            ], dtype=np.float32)
            
            # Save to NPZ file
            np.savez_compressed(
                filepath,
                environment_type=np.array([environment_type], dtype='U20'),
                timestamp=np.array([timestamp], dtype='U30'),
                robot=robot_state,
                chasers=chasers_state,
                humans=humans_state,
                static_obstacles=static_obstacles_state,
                env_params=env_params,
                adv_params=adv_params
            )
            
            print(f"Adversarial environment state exported to: {filepath}")
            
        except Exception as e:
            print(f"Error exporting adversarial environment state: {e}")
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
            print(f"Error importing adversarial environment state: {e}")
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
            self.static_obstacles = None
            self.log_env = {}
            
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
            
            # Set robot state from imported data
            robot_data = state_data['robot']
            self.robot.set(
                px=float(robot_data[0]), py=float(robot_data[1]),
                gx=float(robot_data[2]), gy=float(robot_data[3]),
                vx=float(robot_data[4]), vy=float(robot_data[5]),
                theta=float(robot_data[6]), v_pref=float(robot_data[8])
            )
            self.robot.radius = float(robot_data[7])
            
            # Set chasers state from imported data
            chasers_data = state_data['chasers']
            loaded_chaser_num = len(chasers_data)
            
            # Ensure we have the correct number of chasers
            if loaded_chaser_num != self.chaser_num:
                print(f"Warning: State has {loaded_chaser_num} chasers but environment expects {self.chaser_num}. Adjusting.")
            
            for i, chaser_data in enumerate(chasers_data[:self.chaser_num]):
                if i < len(self.chasers):
                    self.chasers[i].set(
                        px=float(chaser_data[0]), py=float(chaser_data[1]),
                        gx=float(chaser_data[2]), gy=float(chaser_data[3]),
                        vx=float(chaser_data[4]), vy=float(chaser_data[5]),
                        theta=float(chaser_data[6]), v_pref=float(chaser_data[8])
                    )
                    self.chasers[i].radius = float(chaser_data[7])
                    
                    # Reset Ackermann-specific parameters for chaser if needed
                    if self.chaser_model == 'ackermann' or self.chaser_test_model == 'ackermann':
                        self.chasers[i].current_steering_angle = 0.0
                        self.chasers[i].current_velocity = 0.0
                        self.chasers[i].current_angular_velocity = 0.0
            
            # Update backward compatibility reference
            if self.chaser_num > 0:
                self.chaser = self.chasers[0]
            
            # Handle digit environment if present
            if self.digit_env is not None:
                self.digit_env.reset(robot=np.array([self.robot.px, self.robot.py], dtype=np.float32))
                sleep(self.digit_env.cfg.control.control_dt)
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
                if self.robot_test_model == 'ackermann' or self.robot_model == 'ackermann':
                    self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), 
                                           forward_velocity=0.0, steering_angle=0.0, theta_dot=0.0)
                else:
                    self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), differential=True)
            
            self.goal_distance_last = self.robot.get_goal_distance()
            
            # Initialize chaser goal distances and action lasts
            for chaser_idx, chaser in enumerate(self.chasers):
                self.chaser_distance_lasts[chaser_idx] = hypot(chaser.px - self.robot.gx, chaser.py - self.robot.gy)
                self.chaser_action_lasts[chaser_idx] = np.zeros(2)
            
            # Reset chaser active states (all chasers start active)
            self.chaser_active = [True] * self.chaser_num
            
            # Backward compatibility
            if self.chaser_num > 0:
                self.chaser_goal_distance_last = self.chaser_distance_lasts[0]
                self.chaser_action_last = self.chaser_action_lasts[0]
            
            # Initialize ORCA policy parameters if using ORCA or RGL
            if hasattr(self, 'robot_policy_mode') and self.robot_policy_mode in ['orca', 'rgl']:
                self.robot_orca_policy.time_step = self.time_step
                self.robot_orca_policy.max_speed = self.robot.v_pref
                self.robot_orca_policy.radius = self.robot.radius
                self.robot_orca_policy.max_robot_speed = self.robot.v_pref
            
            # Get frame and prepare observation
            self.get_frame()
            
            # Get frames for all chasers
            for chaser_idx in range(self.chaser_num):
                self.get_chaser_frame(chaser_idx)
            
            for _ in range(self.frame_stack):
                self.frames.append(self.single_frame)
                self.lidar_frames.append(self.scan_current / self.laser_max_range)
                # Initialize per-chaser frames
                for chaser_idx in range(self.chaser_num):
                    self.chaser_frames_list[chaser_idx].append(self.chaser_single_frames[chaser_idx])
                    self.chaser_lidar_frames_list[chaser_idx].append(self.chaser_scan_currents[chaser_idx] / self.laser_max_range)
            
            # Update backward compatibility
            if self.chaser_num > 0:
                self.chaser_frames = self.chaser_frames_list[0]
                self.chaser_lidar_frames = self.chaser_lidar_frames_list[0]
                self.chaser_single_frame = self.chaser_single_frames[0]
                self.chaser_scan_current = self.chaser_scan_currents[0]
            
            assert len(self.frames) == self.frame_stack
            assert len(self.lidar_frames) == self.frame_stack
            
            obs_image = self.construct_obs()
            
            # Construct observations for all chasers
            chaser_obs_images = []
            for chaser_idx in range(self.chaser_num):
                chaser_obs_images.append(self.construct_chaser_obs(chaser_idx))
            
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
            
            # Calculate chaser goal emotion states for all chasers
            chaser_goal_emotion_states = []
            for chaser_idx, chaser in enumerate(self.chasers):
                chaser_dx = chaser.gx - chaser.px
                chaser_dy = chaser.gy - chaser.py
                chaser_theta = chaser.theta
                chaser_y_rel = chaser_dy * cos(chaser_theta) - chaser_dx * sin(chaser_theta)
                chaser_x_rel = chaser_dy * sin(chaser_theta) + chaser_dx * cos(chaser_theta)
                chaser_r = hypot(chaser_x_rel, chaser_y_rel)
                chaser_t = atan2(chaser_y_rel, chaser_x_rel)
                
                chaser_goal_emotion_states.append(np.array([
                    chaser_r / self.square_width, chaser_t / np.pi,
                    self.chaser_action_lasts[chaser_idx][0] / self.chaser_action_range[1, 0],
                    self.chaser_action_lasts[chaser_idx][1] / self.chaser_action_range[1, 1]
                ], dtype=np.float32))
            
            print(f"Adversarial environment reset from imported state successfully")
            return obs_image, robot_goal_emotion_state, chaser_obs_images, chaser_goal_emotion_states
            
        except Exception as e:
            print(f"Error resetting adversarial environment from imported state: {e}")
            raise

    def _validate_imported_state(self, state_data):
        """Validate that imported state is safe and valid"""
        try:
            # Check required keys (support both old 'chaser' and new 'chasers' format)
            required_keys = ['environment_type', 'robot', 'humans', 'static_obstacles', 'env_params']
            for key in required_keys:
                if key not in state_data:
                    raise ValueError(f"Missing required key in state data: {key}")
            
            # Support both old and new format
            if 'chasers' not in state_data and 'chaser' not in state_data:
                raise ValueError("Missing chaser data in state file (neither 'chasers' nor 'chaser' found)")
            
            # Check environment type
            env_type = str(state_data['environment_type'][0])
            if env_type != 'adv':
                raise ValueError(f"Environment type mismatch. Expected 'adv', got '{env_type}'")
            
            # Validate robot state
            robot_data = state_data['robot']
            if robot_data.shape != (9,):
                raise ValueError(f"Invalid robot state shape. Expected (9,), got {robot_data.shape}")
            
            # Validate chaser state
            chaser_data = state_data['chaser']
            if chaser_data.shape != (9,):
                raise ValueError(f"Invalid chaser state shape. Expected (9,), got {chaser_data.shape}")
            
            # Check robot position bounds
            robot_px, robot_py = robot_data[0], robot_data[1]
            if abs(robot_px) > self.square_width or abs(robot_py) > self.square_width:
                raise ValueError(f"Robot position out of bounds: ({robot_px}, {robot_py})")
            
            # Check chaser position bounds
            chaser_px, chaser_py = chaser_data[0], chaser_data[1]
            if abs(chaser_px) > self.square_width or abs(chaser_py) > self.square_width:
                raise ValueError(f"Chaser position out of bounds: ({chaser_px}, {chaser_py})")
            
            # Validate robot attributes
            robot_radius, robot_v_pref = robot_data[7], robot_data[8]
            if not (0.1 <= robot_radius <= 1.0):
                raise ValueError(f"Invalid robot radius: {robot_radius}")
            if not (0.1 <= robot_v_pref <= 3.0):
                raise ValueError(f"Invalid robot v_pref: {robot_v_pref}")
            
            # Validate chaser attributes
            chaser_radius, chaser_v_pref = chaser_data[7], chaser_data[8]
            if not (0.1 <= chaser_radius <= 1.0):
                raise ValueError(f"Invalid chaser radius: {chaser_radius}")
            if not (0.1 <= chaser_v_pref <= 3.0):
                raise ValueError(f"Invalid chaser v_pref: {chaser_v_pref}")
            
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
            
            # Add chaser
            all_entities.append({
                'type': 'chaser',
                'px': chaser_data[0], 'py': chaser_data[1], 'radius': chaser_data[7]
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
            
            print(f"Adversarial state validation passed successfully")
            
        except Exception as e:
            print(f"Adversarial state validation failed: {e}")
            raise

