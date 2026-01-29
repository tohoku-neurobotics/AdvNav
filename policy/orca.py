import numpy as np
import rvo2

class ORCA:
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.safety_space = 0.2
        self.neighbor_dist = 10
        self.max_neighbors = 30
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.time_step = None
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None
        self.phase = None
        self.max_robot_speed = None
        self.last_state = None


    def set_phase(self, phase):
        self.phase = phase

    def predict(self, state, has_robot=False):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state
        if has_robot:
            robot_state = state.robot_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            if has_robot:
                robot_state = state.robot_state
                self.sim.addAgent(robot_state.position, *params, robot_state.radius + 0.01 + self.safety_space,
                                self.max_robot_speed, robot_state.velocity)
            self.sim.addAgent(self_state.position, *params, self_state.radius + 0.01 + self.safety_space,
                            self_state.v_pref, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                # Check if this is a static obstacle (zero velocity)
                is_static = np.linalg.norm(human_state.velocity) < 0.001
                # Use zero max_speed for static obstacles
                agent_max_speed = 0.0 if is_static else self.max_speed
                self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                                agent_max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                # Update max speed if this is a static obstacle
                is_static = np.linalg.norm(human_state.velocity) < 0.001
                if not is_static:
                    self.sim.setAgentPosition(i + 1, human_state.position)
                    self.sim.setAgentVelocity(i + 1, human_state.velocity)
            if has_robot:
                self.sim.setAgentPosition(len(state.human_states), robot_state.position)
                self.sim.setAgentVelocity(len(state.human_states), robot_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = self.sim.getAgentVelocity(0)
        self.last_state = state

        return action
