from math import hypot, cos, sin, pi, atan, tan
import numpy as np
from utils.state import FullState, ObservableState, JointState # Assuming these are in the same parent directory

class AckermannRobot():
    def __init__(self, radius=0.3, v_pref=0.4, 
                 axle_spacing=0.332, wheel_spacing=0.332, steering_axle_spacing=0.17,
                 wheel_radius=0.0625, maximum_steering_angle=0.7, time_step=0.1,
                 max_acceleration=2, max_angular_acceleration=2.0, max_steering_rate=1):
        """
        Initialize an Ackermann steering robot.

        Args:
            radius (float): Radius of the robot's footprint for collision checking.
            v_pref (float): Preferred (max) speed of the robot.
            axle_spacing (float): Distance between front and rear axles (wheelbase L).
            wheel_spacing (float): Distance between the center of the two wheels on an axle (track width W).
                                   Not directly used in the simple bicycle model kinematics but good to have.
            wheel_radius (float): Radius of the wheels.
            time_step (float): Simulation time step.
            max_acceleration (float): Maximum acceleration allowed in m/s².
            max_angular_acceleration (float): Maximum angular acceleration in rad/s².
            max_steering_rate (float): Maximum rate of change of steering angle in rad/s.
        """
        self.radius = radius
        self.v_pref = v_pref
        self.px = 0.0
        self.py = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.time_step = time_step
        self.current_velocity = 0.0  # Track current forward velocity for acceleration limits
        self.max_acceleration = max_acceleration  # Maximum acceleration in m/s²
        self.current_steering_angle = 0.0  # Track current steering angle
        self.current_angular_velocity = 0.0  # Track angular velocity (theta_dot)
        self.max_angular_acceleration = max_angular_acceleration  # Maximum angular acceleration in rad/s²
        self.max_steering_rate = max_steering_rate  # Maximum rate of change for steering angle in rad/s
        self.steering_axle_spacing = steering_axle_spacing
        self.maximum_steering_angle = maximum_steering_angle
        # Ackermann specific parameters
        self.axle_spacing = axle_spacing  # L, wheelbase
        self.wheel_spacing = wheel_spacing # W, track width
        self.wheel_radius = wheel_radius
        self.max_steering_angle = maximum_steering_angle

    def set(self, px, py, gx, gy, vx, vy, theta, v_pref=None):
        """Set the robot's state manually."""
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.current_velocity = hypot(vx, vy)  # Update forward velocity based on vx, vy
        # Reset steering angle and angular velocity when state is manually set
        self.current_steering_angle = 0.0
        self.current_angular_velocity = 0.0
        if v_pref is not None:
            self.v_pref = v_pref

    def get_full_state(self):
        """Get the full state of the robot."""
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, 
                         self.gx, self.gy, self.v_pref, self.theta)

    def get_observable_state(self):
        """Get the observable state of the robot (e.g., for other agents)."""
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_position(self):
        """Get the robot's current (px, py) position."""
        return self.px, self.py

    def get_goal_distance(self):
        """Calculate the Euclidean distance to the goal."""
        return hypot(self.gx - self.px, self.gy - self.py)
    
    def get_sfm_state(self): # May need review if SFM is used differently for Ackermann
        """Get state relevant for Social Force Model (if applicable)."""
        return [self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.v_pref]

    def compute_pose(self, action):
        """
        Compute the new pose (px_new, py_new, theta_new) based on an action.
        Action is [forward_velocity, steering_angle].
        """
        requested_velocity = action[0]
        requested_steering_angle = action[1]
        
        # Apply steering rate limits - limit how quickly steering angle can change
        max_steering_change = self.max_steering_rate * self.time_step
        steering_change = requested_steering_angle - self.current_steering_angle
        
        if abs(steering_change) > max_steering_change:
            steering_change = np.sign(steering_change) * max_steering_change
            
        # Apply the limited steering change
        steering_angle = self.current_steering_angle + steering_change
        
        # Clamp steering angle to its physical limits
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Apply acceleration limits
        max_velocity_change = self.max_acceleration * self.time_step
        velocity_change = requested_velocity - self.current_velocity
        
        # Limit the velocity change based on acceleration constraint
        if abs(velocity_change) > max_velocity_change:
            velocity_change = np.sign(velocity_change) * max_velocity_change
        
        # Apply the limited velocity change
        forward_velocity = self.current_velocity + velocity_change

        # Calculate new angular velocity (theta_dot)
        L = self.axle_spacing
        K = self.steering_axle_spacing
        tan_alpha = tan(steering_angle)
        if abs(tan_alpha) < 1e-6:
            theta_dot = 0.0
        else:
            if steering_angle >= 0:
                R = L / tan_alpha + K / 2
            else:
                R = L / tan_alpha - K / 2
            theta_dot = forward_velocity / R

        

        # Update pose using current theta for calculating dx, dy component of velocity
        # Then update theta based on theta_dot
        px_new = self.px + forward_velocity * cos(self.theta) * self.time_step
        py_new = self.py + forward_velocity * sin(self.theta) * self.time_step
        theta_new = self.theta + theta_dot * self.time_step

        # Normalize theta to be within [-pi, pi]
        while theta_new > pi:
            theta_new -= 2 * pi
        while theta_new < -pi:
            theta_new += 2 * pi
            
        return px_new, py_new, theta_new, forward_velocity, steering_angle, theta_dot

    def update_states(self, px_new, py_new, theta_new, action, forward_velocity=None, 
                     steering_angle=None, theta_dot=None):
        """
        Update the robot's state variables after a pose computation.
        px_new, py_new, theta_new are the results from compute_pose.
        action is the original [forward_velocity, steering_angle] command.
        forward_velocity, steering_angle, and theta_dot are the actual applied values
        (considering acceleration and steering rate limits).
        """
        self.px = px_new
        self.py = py_new
        self.theta = theta_new # New orientation

        # Update velocity state
        if forward_velocity is not None:
            self.current_velocity = forward_velocity
        else:
            self.current_velocity = action[0]
            
        # Update steering angle state    
        if steering_angle is not None:
            self.current_steering_angle = steering_angle
        else:
            self.current_steering_angle = action[1]
            
        # Update angular velocity state
        if theta_dot is not None:
            self.current_angular_velocity = theta_dot
            
        # Update world-frame velocities based on the current velocity and new orientation
        self.vx = self.current_velocity * cos(self.theta)
        self.vy = self.current_velocity * sin(self.theta)

    # The original update_vel might not be directly applicable for Ackermann control
    # def update_vel(self, action):
    #     self.vx = action[0]
    #     self.vy = action[1] 