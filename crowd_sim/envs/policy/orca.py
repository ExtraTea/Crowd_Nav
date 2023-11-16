import numpy as np

from rvo.simulator import Simulator
from rvo.vector import Vector2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class ORCA(Policy):
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
        self.max_neighbors = None
        self.radius = None
        self.max_speed = 1 # the ego agent assumes that all other agents have this max speed
        self.sim = None
        self.safety_space = 0.01
        self.kinematics = 'holonomic'


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
        # max number of humans = current number of humans
        self.max_neighbors = len(state.human_states)
        self.radius = state.self_state.radius
        # params = self.config.orca.neighbor_dist, self.max_neighbors, self.config.orca.time_horizon, self.config.orca.time_horizon_obst
        if self.sim is not None and self.sim.num_agents != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        del self.sim
        self.sim=None
        if self.sim is None:
            # print("hoge")
            self.sim = Simulator()
            self.sim.set_agent_defaults(100.0, 10, 5.0, 5.0, 5.46, 3.0, Vector2(0.0, 0.0))
            # self.sim.addAgent((self_state.px, self_state.py), *params, self_state.radius + 0.01 + self.safety_space,
            #                   self_state.v_pref, (self_state.vx, self_state.vy))
            
            self.sim.add_agent(Vector2(self_state.px, self_state.py),self_state.radius + 0.01 + self.safety_space, Vector2(self_state.vx, self_state.vy))
            # self.sim.agents_[0].max_speed_ = self_state.v_pref
            # self.sim.agents_[0].velocity_ = Vector2(self_state.vx, self_state.vy)
            # print(0)
            
            for i, human_state in enumerate(state.human_states):
                # self.sim.addAgent((human_state.px, human_state.py), *params, human_state.radius + 0.01 + self.config.orca.safety_space,
                #                   self.max_speed, (human_state.vx, human_state.vy))
                self.sim.add_agent(Vector2(human_state.px, human_state.py), human_state.radius + 0.01 + 0.15, Vector2(human_state.vx, human_state.vy))
                self.sim.agents_[i+1].max_speed_ = self.max_speed
                # self.sim.agents_[i+1].velocity_ = Vector2(human_state.vx, human_state.vy)
                # print((self_state.px-human_state.px)**2+(self_state.py-human_state.py)**2)
                # print(self_state.px, self_state.py, 0, human_state.px, human_state.py, i+1)
                # print(i+1)
                # self.sim.set_agent_pref_velocity(-1, human_state.v_pref)
            
        # else:
        #     self.sim.setAgentPosition(0, (self_state.px, self_state.py))
        #     self.sim.setAgentVelocity(0, (self_state.vx, self_state.vy))
        #     for i, human_state in enumerate(state.human_states):
        #         self.sim.setAgentPosition(i + 1, (human_state.px, human_state.py))
        #         self.sim.setAgentVelocity(i + 1, (human_state.vx, human_state.vy))


        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.set_agent_pref_velocity(0, Vector2(pref_vel[0], pref_vel[1]))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.set_agent_pref_velocity(i + 1, Vector2(0, 0))

        self.sim.step()
        action = ActionXY(self.sim.agents_[0].velocity_.x, self.sim.agents_[0].velocity_.y)
        self.last_state = state

        return action

ORCA()