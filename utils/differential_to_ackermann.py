from math import atan
import numpy as np


class DifferentialToAckermannConverter():
    def __init__(self, axle_spacing, max_steering_angle, min_velocity=0.05, epsilon=1e-6):
        """
        Initialize the converter from differential drive to Ackermann steering commands.
        
        Args:
            axle_spacing (float): Distance between front and rear axles (wheelbase L).
            max_steering_angle (float): Maximum steering angle in radians.
            min_velocity (float): Minimum velocity threshold for handling small velocities with rotation.
                                  Default is 0.05 m/s.
            epsilon (float): Small value to prevent division by zero. Default is 1e-6.
        """
        self.L = axle_spacing
        self.max_steering_angle = max_steering_angle
        self.min_velocity = min_velocity
        self.epsilon = epsilon
    
    def convert(self, action):
        """
        Convert differential drive command to Ackermann steering command.
        
        Args:
            action: Array-like [forward_velocity, angular_velocity]
                   - forward_velocity: Linear velocity in m/s
                   - angular_velocity: Angular velocity in rad/s
        
        Returns:
            Array [forward_velocity, steering_angle] for Ackermann robot
                   - forward_velocity: Adjusted linear velocity in m/s
                   - steering_angle: Steering angle in radians
        """
        # Extract differential drive commands
        v = action[0]
        omega = action[1]
        
        # Clamp negative velocity to zero (Ackermann can't go backward easily)
        if v < 0:
            v = 0.0
        
        # Handle small velocity with rotation: boost to minimum velocity
        # to approximate in-place rotation with small v and large steering angle
        if v < self.min_velocity and omega != 0:
            v = self.min_velocity
        
        # Calculate steering angle using kinematic relationship
        # omega = v / R, where R = L / tan(steering_angle)
        # Therefore: steering_angle = atan(omega * L / v)
        # Add epsilon to denominator to prevent division by zero
        steering_angle = atan(omega * self.L / (v + self.epsilon))
        
        # Clamp steering angle to physical limits
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        return np.array([v, steering_angle])



