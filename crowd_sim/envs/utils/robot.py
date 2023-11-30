from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.policy import orca
from crowd_sim.envs.utils.action import ActionRot
import math
import numpy as np
class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
    
    def imact(self, ob):
        def convert_velocity_to_rotation_and_magnitude(vx, vy):
            rotation = math.atan2(vy, vx)  # 角度をラジアンで計算
            velocity = math.sqrt(vx ** 2 + vy ** 2)  # 速度の大きさを計算
            return rotation, velocity
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        pol = orca.ORCA()
        action = pol.predict(state)
        rotation, velocity = convert_velocity_to_rotation_and_magnitude(action.vx, action.vy)
        action = np.array([velocity, rotation])
        # action = np.array([1,math.pi/2])
        return action

