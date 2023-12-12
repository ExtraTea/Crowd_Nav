import logging
import gymnasium as gym
import matplotlib.lines as mlines
import numpy as np
# import rvo2
from rvo.simulator import Simulator
from rvo.vector import Vector2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
import math
from gymnasium.spaces import Box
# from gymnasium.spaces import 
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from time import sleep
from matplotlib import pyplot as plt
import random as rand
class CrowdSim(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30  # または適切なフレームレート
    }

    def __init__(self, render_mode=None):
        super(CrowdSim, self).__init__()
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.count = 0
        # for gym
        # d = {}
        # d['robot_rotated_node'] = gym.spaces.Box(low=-1000, high=1000, shape=(6,), dtype=np.float64)
        # pedestrian_obs_dim = 6
        # num_humans = 5
        # d['pedestrian_node'] = gym.spaces.Box(low=-1000, high=1000, shape=(num_humans, pedestrian_obs_dim, ), dtype=np.float64)
        # d['angular_map'] = gym.spaces.Box(low=0, high = 5, shape = (72,), dtype=np.float64)
        # d['robot_node'] = gym.spaces.Box(low=-1000, high=1000, shape=(1, pedestrian_obs_dim, ), dtype=np.float64)
        # self.observation_space = gym.spaces.Dict(d)
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(114, ), dtype=np.float64)
        high = 1000 * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float64)
        self.set_robot()
        def sample_truncated_normal(mean, lower, upper):
            while True:
                sample = np.random.normal(mean, 1)  # 標準偏差は1と仮定
                if lower <= sample <= upper:
                    return sample
        # for wall obstacle
        self.bottom = rand.uniform(-5000, 5000)
        self.top = self.bottom + sample_truncated_normal(7, 5, 16)
        self.endline = 30
        self.walls = np.array([
            [-self.endline, self.bottom, -self.endline, self.top],  # Left wall
            [self.endline, self.bottom, self.endline, self.top],    # Right wall
            [-self.endline, self.bottom, self.endline, self.bottom],# Bottom wall
            [-self.endline, self.top, self.endline, self.top]       # Top wall
        ])

        self.fig, self.ax = plt.subplots(figsize=(7, 7))

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        # logging.info('human number: {}'.format(self.human_num))
        # if self.randomize_attributes:
        #     logging.info("Randomize human's radius and preferred speed")
        # else:
        #     logging.info("Not randomize human's radius and preferred speed")
        # logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        # logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self):
        from crowd_sim.envs.utils.robot import Robot
        import configparser
        config_file_path = r'/home/dai/sotsuron/original_crowdnav/CrowdNav/crowd_nav/configs/env.config'
        
        # configparserのインスタンスを作成し、設定ファイルを読み込む
        env_config = configparser.RawConfigParser()
        env_config.read(config_file_path)
        self.robot = Robot(env_config, 'robot')
        self.configure(env_config)
        self.robot.set_policy(SARL())
        

    def generate_random_human_position(self, human_num, rule='wall_crossing'):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_wall_crossing_human())
        elif rule == 'wall_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_wall_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_wall_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        
        while True:
            # 出発地点と目的地の範囲設定
            if np.random.random() < 0.5:
                # -6 < x < -3 から出発し、3 < x < 6 にゴール
                px = np.random.uniform(-7, -3)
                gx = np.random.uniform(3, 7)
            else:
                # 3 < x < 6 から出発し、-6 < x < -3 にゴール
                px = np.random.uniform(3, 7)
                gx = np.random.uniform(-7, -3)

            # -3 < y < 3 の範囲でランダムに選択
            py = np.random.uniform(self.bottom, self.top)
            gy = np.random.uniform(self.bottom, self.top)

            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                    collide = True
                    break

            if not collide:
                break

        human.set(px, py, gx, gy, 0, 0, 0)
        return human


    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human
   
    def get_human_times(self):
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = Simulator()
        sim.set_agent_defaults(100.0, 10, 5.0, 5.0, 5.46, 3.0, Vector2(0.0, 0.0))
        sim.add_agent(Vector2(self.robot.px, self.robot.py),self.robot.radius + 0.01 + self.safety_space, Vector2(self.robot.vx, self.robot.vy))
        for i, human in enumerate(self.humans):
            sim.add_agent(Vector2(human.px, human.py), human.radius + 0.01 + self.config.orca.safety_space, Vector2(human.vx, human.vy))
            sim.agents_[i+1].max_speed_ = self.max_speed
        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.set_agent_pref_velocity(i, tuple(vel_pref))
            sim.step()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time
            
            # for visualization
            self.robot.set_position(sim.get_agent_position(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.get_agent_position(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        del sim
        return self.human_times

    def reset(self, phase='train', test_case=None, options=None, seed=None):
        import numpy as np
        self.bottom = rand.uniform(-5, 5)
        self.top = self.bottom + rand.uniform(3,9)
        def set_robot_position(self):
            # ロボットの出発地点と目的地の範囲設定
            if np.random.random() < 0.5:
                # -6 < x < -3 から出発し、3 < x < 6 にゴール
                px = np.random.uniform(-8, -3)
                gx = np.random.uniform(3, 8)
            else:
                # 3 < x < 6 から出発し、-6 < x < -3 にゴール
                px = np.random.uniform(3, 8)
                gx = np.random.uniform(-8, -3)

            # -3 < y < 3 の範囲でランダムに選択
            py = np.random.uniform(self.bottom + self.robot.radius, self.top - self.robot.radius)
            gy = np.random.uniform(self.bottom + self.robot.radius, self.top - self.robot.radius)

            # ロボットの初期速度と向き
            vx = 0
            vy = 0
            theta = np.pi / 2

            self.robot.set(px, py, gx, gy, vx, vy, theta)

        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.set_robot()
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            # self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
            self.human_times = [0] * (self.human_num)
        # if not self.robot.policy.multiagent_training:
        #     self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            # self.robot.set(0, -self.circle_radius, 2, self.circle_radius, 0, 0, np.pi / 2)
            set_robot_position(self)
            if self.case_counter[phase] >= 0:
                # np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    # human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    human_num = self.human_num
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            # create anglular map
            angular_map = self.create_angular_map(self.robot.px, self.robot.py, self.robot.theta)
            robot_obs = self.robot.get_full_state()
            robot_obs = np.array([
                            robot_obs.px, 
                            robot_obs.py, 
                            robot_obs.vx, 
                            robot_obs.vy, 
                            robot_obs.radius, 
                            math.atan2(robot_obs.face_orientation.y_, robot_obs.face_orientation.x_)
                        ], dtype=np.float64)
            robot_obs = np.expand_dims(robot_obs, axis=0)
            pedestrian_observations = []
            for human in self.humans:
                observable_state = human.get_observable_state()
                # ObservableState オブジェクトから必要な情報を抽出
                obs_array = np.array([
                    observable_state.px, 
                    observable_state.py, 
                    observable_state.vx, 
                    observable_state.vy, 
                    observable_state.radius, 
                    math.atan2(observable_state.face_orientation.y_, observable_state.face_orientation.x_)
                ], dtype=np.float64)
                pedestrian_observations.append(obs_array)

            # すべての観測値を一つの配列に結合
            # pedestrian_channel = np.concatenate(pedestrian_observations, axis=0)
            pedestrian_channel = np.array(pedestrian_observations)
            dg1 = np.sqrt((self.robot.px - self.robot.gx) ** 2 + (self.robot.py - self.robot.gy) ** 2)
            vel_pref = 0
            # alpha is the differenve of the current heading to the angle to the goal
            alpha = np.arctan2(self.robot.gy - self.robot.py, self.robot.gx - self.robot.px) - self.robot.theta
            #vx_rc and vy_rc is the robot’s current velocity in the robot centered frame. x axis is directed at the goal
            vx_rc = self.robot.vx * np.cos(alpha) + self.robot.vy * np.sin(alpha)
            vy_rc = -self.robot.vx * np.sin(alpha) + self.robot.vy * np.cos(alpha)
            robot_channel = np.array([dg1, vel_pref, alpha, self.robot.radius, vx_rc, vy_rc], dtype=np.float64)
            combined_array = np.concatenate([
                robot_channel.flatten(),     # 1次元に平坦化
                pedestrian_channel.flatten(),     # 1次元に平坦化
                angular_map.flatten(),     # 既に1次元
                robot_obs.flatten()      # 1次元に平坦化
            ])
            # obs = {"robot_rotated_node": robot_channel, "pedestrian_node": pedestrian_channel, 'angular_map': angular_map, 'robot_node': robot_obs}
            obs = combined_array

        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
        
        return obs, {}

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            # if self.robot.visible:
            if True:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob, self.bottom, self.top))
        # # for imitation learning
        # ob = [other_human.get_observable_state() for other_human in self.humans]
        # robot_action = self.robot.imact(ob, self.bottom, self.top)
        # action = robot_action


        dmin = float('inf')
        dmin=[]
        collision = False
        info = None
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                acv = action[0]
                acr = action[1]
                action = ActionRot(acv, acr)
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius

            # 人間の視線方向とロボットの方向との間の角度を計算
            gaze_direction = np.array([human.face_orientation.x, human.face_orientation.y])
            robot_direction = np.array([px, py])
            dot_product = np.dot(gaze_direction, robot_direction)
            norm_product = np.linalg.norm(gaze_direction) * np.linalg.norm(robot_direction)
            
            # 避けるべきゼロ割を防ぐ
            if norm_product != 0:
                angle = np.arccos(dot_product / norm_product)
            else:
                angle = np.pi  # 180度：ロボットは視界内にいないと仮定
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            else:
                if abs(angle) <= 90:
                    dmin.append(closest_dist)

        # collision against wall
        px, py = self.robot.px, self.robot.py

        # 左の壁に衝突しているか判定
        if px < -30 + self.robot.radius:
            collision = True

        # 右の壁に衝突しているか判定
        if px > 30 - self.robot.radius:
            collision = True

        # 上の壁に衝突しているか判定
        if py > self.top - self.robot.radius:
            collision = True

        # 下の壁に衝突しているか判定
        if py < self.bottom + self.robot.radius:
            collision = True

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius
        truncated = False
        if self.global_time >= self.time_limit - 1:
            reward = 0.0
            done = True
            info = {'event': 'timeout', 'action': np.array([action.v, action.r])}
            truncated = True
        elif collision:
            reward = self.collision_penalty
            done = True
            info = {'event': 'collision', 'action': np.array([action.v, action.r])}
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = {'event': 'reaching_goal', 'action': np.array([action.v, action.r])}
        elif any(dist < self.discomfort_dist for dist in dmin):
            # self.discomfort_distより小さいすべての距離に対して報酬を計算
            rewards = [(dist - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step for dist in dmin if dist < self.discomfort_dist]

            # 複数の報酬がある場合は、それらを合計する
            total_reward = sum(rewards) if rewards else 0

            # 何かしらの距離がself.discomfort_distより小さい場合のみ、イベントとして記録
            if rewards:
                done = False
                info = {'event': 'danger', 'min_dists': [dist for dist in dmin if dist < self.discomfort_dist], 'action': np.array([action.v, action.r])}
                reward = total_reward
        else:
            reward = 0.0
            done = False
            info = {'event': 'nothing', 'action': np.array([action.v, action.r])}
        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                angular_map = self.create_angular_map(self.robot.px, self.robot.py, self.robot.theta)
                robot_obs = self.robot.get_full_state()
                robot_obs = np.array([
                            robot_obs.px, 
                            robot_obs.py, 
                            robot_obs.vx, 
                            robot_obs.vy, 
                            robot_obs.radius, 
                            math.atan2(robot_obs.face_orientation.y_, robot_obs.face_orientation.x_)
                        ], dtype=np.float64)
                robot_obs = np.expand_dims(robot_obs, axis=0)
                pedestrian_observations = []
                for human in self.humans:
                    observable_state = human.get_observable_state()
                    # ObservableState オブジェクトから必要な情報を抽出
                    obs_array = np.array([
                        observable_state.px, 
                        observable_state.py, 
                        observable_state.vx, 
                        observable_state.vy, 
                        observable_state.radius, 
                        math.atan2(observable_state.face_orientation.y_, observable_state.face_orientation.x_)
                    ], dtype=np.float64)
                    pedestrian_observations.append(obs_array)

                # すべての観測値を一つの配列に結合
                # pedestrian_channel = np.concatenate(pedestrian_observations, axis=0)
                pedestrian_channel = np.array(pedestrian_observations)
                dg1 = np.sqrt((self.robot.px - self.robot.gx) ** 2 + (self.robot.py - self.robot.gy) ** 2)
                vel_pref = 0
                # alpha is the differenve of the current heading to the angle to the goal
                alpha = np.arctan2(self.robot.gy - self.robot.py, self.robot.gx - self.robot.px) - self.robot.theta
                #vx_rc and vy_rc is the robot’s current velocity in the robot centered frame. x axis is directed at the goal
                vx_rc = self.robot.vx * np.cos(alpha) + self.robot.vy * np.sin(alpha)
                vy_rc = -self.robot.vx * np.sin(alpha) + self.robot.vy * np.cos(alpha)
                robot_channel = np.array([dg1, vel_pref, alpha, self.robot.radius, vx_rc, vy_rc], dtype=np.float64)

                combined_array = np.concatenate([
                    robot_channel.flatten(),     # 1次元に平坦化
                    pedestrian_channel.flatten(),     # 1次元に平坦化
                    angular_map.flatten(),     # 既に1次元
                    robot_obs.flatten()      # 1次元に平坦化
                ])
                # obs = {"robot_rotated_node": robot_channel, "pedestrian_node": pedestrian_channel, 'angular_map': angular_map, 'robot_node': robot_obs}
                obs = combined_array
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                angular_map = self.create_angular_map(self.robot.px, self.robot.py, self.robot.theta)
                robot_obs = self.robot.get_full_state()
                robot_obs = np.array([
                            robot_obs.px, 
                            robot_obs.py, 
                            robot_obs.vx, 
                            robot_obs.vy, 
                            robot_obs.radius, 
                            math.atan2(robot_obs.face_orientation.y_, robot_obs.face_orientation.x_)
                        ], dtype=np.float64)
                robot_obs = np.expand_dims(robot_obs, axis=0)
                pedestrian_observations = []
                for human in self.humans:
                    observable_state = human.get_observable_state()
                    # ObservableState オブジェクトから必要な情報を抽出
                    obs_array = np.array([
                        observable_state.px, 
                        observable_state.py, 
                        observable_state.vx, 
                        observable_state.vy, 
                        observable_state.radius, 
                        math.atan2(observable_state.face_orientation.y_, observable_state.face_orientation.x_)
                    ], dtype=np.float64)
                    pedestrian_observations.append(obs_array)

                # すべての観測値を一つの配列に結合
                # pedestrian_channel = np.concatenate(pedestrian_observations, axis=0)
                pedestrian_channel = np.array(pedestrian_observations)
                dg1 = np.sqrt((self.robot.px - self.robot.gx) ** 2 + (self.robot.py - self.robot.gy) ** 2)
                vel_pref = 0
                # alpha is the differenve of the current heading to the angle to the goal
                alpha = np.arctan2(self.robot.gy - self.robot.py, self.robot.gx - self.robot.px) - self.robot.theta
                #vx_rc and vy_rc is the robot’s current velocity in the robot centered frame. x axis is directed at the goal
                vx_rc = self.robot.vx * np.cos(alpha) + self.robot.vy * np.sin(alpha)
                vy_rc = -self.robot.vx * np.sin(alpha) + self.robot.vy * np.cos(alpha)
                robot_channel = np.array([dg1, vel_pref, alpha, self.robot.radius, vx_rc, vy_rc], dtype=np.float64)

                combined_array = np.concatenate([
                    robot_channel.flatten(),     # 1次元に平坦化
                    pedestrian_channel.flatten(),     # 1次元に平坦化
                    angular_map.flatten(),     # 既に1次元
                    robot_obs.flatten()      # 1次元に平坦化
                ])
                # obs = {"robot_rotated_node": robot_channel, "pedestrian_node": pedestrian_channel, 'angular_map': angular_map, 'robot_node': robot_obs}
                obs = combined_array            
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        # rewards = rewards.astype(np.float32)
        return obs, reward, done, truncated, info

    def render(self, mode='human', output_file=None):
        from matplotlib import animation, patches
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        wall_color = 'gray'  # 壁の色

        if mode == 'human':
            self.ax.clear()
            self.ax.set_xlim(-8, 8)
            self.ax.set_ylim(self.bottom-2, self.bottom+14)

            # 壁の描画
            # 左の壁
            left_wall = patches.Rectangle((-self.endline, self.bottom), 0.1, self.top - self.bottom, edgecolor=wall_color, facecolor=wall_color)
            self.ax.add_patch(left_wall)

            # 右の壁
            right_wall = patches.Rectangle((self.endline - 0.1, self.bottom), 0.1, self.top - self.bottom, edgecolor=wall_color, facecolor=wall_color)
            self.ax.add_patch(right_wall)

            # 上の壁
            top_wall = patches.Rectangle((-self.endline, self.top - 0.1), 2 * self.endline, 0.1, edgecolor=wall_color, facecolor=wall_color)
            self.ax.add_patch(top_wall)

            # 下の壁
            bottom_wall = patches.Rectangle((-self.endline, self.bottom), 2 * self.endline, 0.1, edgecolor=wall_color, facecolor=wall_color)
            self.ax.add_patch(bottom_wall)

            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
                # Determine the direction of the human's gaze
                gaze_direction_x = human.face_orientation.x
                gaze_direction_y = human.face_orientation.y
                # print(gaze_direction_x, gaze_direction_y)
                # Calculate the end point of the arrow based on the gaze direction
                arrow_end_x = human.get_position()[0] + gaze_direction_x
                arrow_end_y = human.get_position()[1] + gaze_direction_y

                # Draw an arrow to represent the gaze direction
                gaze_arrow = patches.FancyArrow(human.get_position()[0], human.get_position()[1], arrow_end_x - human.get_position()[0], arrow_end_y - human.get_position()[1], width=0.1, head_width=0.3, head_length=0.3, color=arrow_color)
                self.ax.add_patch(gaze_arrow)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.draw()
            plt.pause(0.001)



        
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
                plt.legend([robot], ['Robot'], fontsize=16)
            self.count +=1
            print(self.count)
            if self.count%64==0:
                plt.show()
            
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError

    def create_angular_map(self, robot_x, robot_y, robot_theta):
        def calculate_wall_distance(px, py, ray_angle):
            x1, y1, x2, y2 = self.walls.T
            denominator = (x1 - x2) * np.sin(ray_angle) - (y1 - y2) * np.cos(ray_angle)
            valid = denominator != 0

            t = ((x1 - px) * np.sin(ray_angle) - (y1 - py) * np.cos(ray_angle)) / denominator
            u = -((x1 - x2) * (y1 - py) - (y1 - y2) * (x1 - px)) / denominator

            valid &= (t >= 0) & (t <= 1) & (u >= 0)
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            distances = np.sqrt((ix - px) ** 2 + (iy - py) ** 2)

            min_distance = np.min(distances[valid]) if np.any(valid) else float('inf')
            return min_distance
        
        angles = np.arange(0, 360, 5)
        ray_angles = np.radians(angles + robot_theta)
        angular_map = np.ones(angles.size) * 5  # Initialize the map, every 5 degrees, with a maximum of 5 meters
        for i, ray_angle in enumerate(ray_angles):
            distance = calculate_wall_distance(robot_x, robot_y, ray_angle)
            angular_map[i] = min(distance, 5)  # Limit distance to 5 meters

        return angular_map
        