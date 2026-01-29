import numpy as np
import math
import rvo2

#social force model 
# https://github.com/lc6chang/Social_Force_Model
class SFM:
    def __init__(self):
        
        self.name = 'SFM'
        self.A = 2000
        self.B = -0.08
        self.force_dis = 1.4
        self.max_speed = 1
        self.phase = None


    def set_phase(self, phase):
        self.phase = phase

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        sum_of_fij = np.zeros(2)
        for human_state in state.human_states:
            d = (((self_state.position[0] - human_state.position[0]))**2 + ((self_state.position[1] - human_state.position[1]))**2)**0.5
            if d >= self.force_dis:  #if distance is over force_dis, neglect the social force
                continue
            fij = self.A * math.exp((d-self_state.radius - human_state.radius)/self.B)
            sum_of_fij[0] = sum_of_fij[0] + fij * (self_state.position[0] - human_state.position[0])
            sum_of_fij[1] = sum_of_fij[1] + fij * (self_state.position[1] - human_state.position[1])
        a_x = ((self_state.mass * (pref_vel[0] - self_state.velocity[0]) / 0.5) + sum_of_fij[0]) / self_state.mass
        a_y = ((self_state.mass * (pref_vel[1] - self_state.velocity[1]) / 0.5) + sum_of_fij[1]) / self_state.mass

       
        action = np.array([a_x, a_y])
        self.last_state = state

        return action
