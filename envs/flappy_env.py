# Helpers
import os
import sys
sys.path.append('/Users/mintaekim/Desktop/Hybrid Robotics Lab/Flappy/Integrated/Flappy_Integrated/flappy_v2/envs')
import numpy as np
from typing import Dict, Union
from collections import deque
# Gym
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.utils import seeding
# Mujoco
from mujoco_gym.mujoco_env import MujocoEnv
# Flappy
from dynamics import Flappy
from parameter import Simulation_Parameter
from action_filter import ActionFilterButter
from env_randomize import EnvRandomizer
from utility_functions import *
import utility_trajectory as ut
from rotation_transformations import *


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 5.0,}

TRAJECTORY_TYPES = {"linear": 0, "circular": 1, "setpoint": 2}

class FlappyEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        max_timesteps = 3000,
        is_visual     = False,
        randomize     = False,
        debug         = False,
        lpf_action    = False,
        traj_type     = False,
        # MujocoEnv
        xml_file: str = "Flappy_v6.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs
    ):
        # Dynamics simulator
        p = Simulation_Parameter()
        self.sim = Flappy(p=p, render=is_visual)

        # Frequency
        self.max_timesteps         = max_timesteps
        self.timestep              = 0.0
        self.sim_freq              = self.sim.freq # 2000Hz
        self.dt                    = 1.0 / self.sim_freq # 1/2000
        self.policy_freq           = 30 # 30Hz but the real control frequency is not exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.policy_freq
        self.secs_per_env_step     = self.num_sims_per_env_step / self.sim_freq
        self.policy_freq           = int(1.0/self.secs_per_env_step)

        self.is_visual          = is_visual
        self.randomize          = randomize
        self.debug              = debug
        self.traj_type          = traj_type
        self.noisy              = False
        self.randomize_dynamics = False # True to randomize dynamics
        self.lpf_action         = lpf_action # action filter 

        # Observation, need to be reduce later for smoothness
        self.n_state            = 41 # NOTE: change to the number of states *we can measure*
        self.n_action           = 7  # NOTE: change to the number of action
        self.history_len_short  = 4
        self.history_len_long   = 10
        self.history_len        = 4
        self.previous_obs       = deque(maxlen=self.history_len)
        self.previous_act       = deque(maxlen=self.history_len)
        
        self.action_space = Box(low=-100, high=100, shape=(self.n_action,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(462,)) # NOTE: change to the actual number of obs to actor policy
        self.observation_space_policy = Box(low=-np.inf, high=np.inf, shape=(462,)) # NOTE: change to the actual number of obs to actor policy
        self.observation_space_value_func = Box(low=-np.inf, high=np.inf, shape=(462,)) # NOTE: change to the actual number of obs to the value function
        
        # NOTE: the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.pos_lb = np.array([-np.inf, -np.inf, 0])  # fight space dimensions: xyz
        self.pos_ub = np.array([np.inf, np.inf, 4])  # fight space dimensions
        self.vel_lb = np.array([-2, -2, -2])
        self.vel_ub = np.array([2, 2, 2])

        # NOTE: define your action bounds
        self.action_lower_bounds = np.array([8,1,0.1,2,0.1,0.1,2,2,0.15])
        self.action_upper_bounds = np.array([8.5,1.5,0.2,2.5,0.2,0.2,2.5,2.5,0.2])
        
        # Info for normalizing the state
        self._init_action_filter()
        # self._init_env_randomizer() # NOTE: Take out dynamics randomization first 
        self._seed()
        self.reset()

        # MujocoEnv
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=self.observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    def _init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut        = None,
            highcut       = [4],
            sampling_rate = self.policy_freq,
            order         = 2,
            num_joints    = self.n_action,
        )

    def _seed(self, seed=None):
        self.np_random, _seeds = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, randomize=None):
        if randomize is None:
            randomize = self.randomize
        self._reset_env(randomize)
        self.action_filter.reset()
        # self.env_randomizer.randomize_dynamics()
        # self._set_dynamics_properties()
        self._update_data(step=False)
        return self._get_obs(step=False)
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _init_env_randomizer(self):
        self.env_randomizer = EnvRandomizer(self.sim)

    def _set_dynamics_properties(self):
        if self.randomize_dynamics:
            self.sim.set_dynamics()

    def _reset_env(self, randomize=False):
        self.timestep    = 0 # discrete timestep, k
        self.time_in_sec = 0.0 # time 
        # NOTE: set robot init position
        self.goal = np.concatenate([np.zeros(3), [2.0, 0.0, 0.0]]) # goal is x y z vx vy vz
        self.sim.reset()
        # use action
        self.last_act   = np.zeros(self.n_action)
        self.reward     = None
        self.terminated = None
        self.info       = {}

    def _get_obs(self, action=0.0, step=False):
        obs_curr    = self.obs_states      
        obs_curr_gt = self.gt_states
        obs_command = self.goal
        if self.timestep == 0:
            [self.previous_obs.append(obs_curr) for i in range(self.history_len)]
            [self.previous_act.append(np.zeros(self.n_action)) for i in range(self.history_len)]
        obs_prev = np.concatenate([np.array(self.previous_obs, dtype=object).flatten(), np.array(self.previous_act, dtype=object).flatten()])
        obs_pol  = np.concatenate([obs_prev, obs_curr, obs_command])
        obs_vf   = np.concatenate([obs_prev, obs_curr_gt, obs_command])
        if step:
            self.previous_obs.append(obs_curr)
            self.previous_act.append(action)
        return obs_vf, obs_pol

    def _act_norm2actual(self, act):
        return self.action_lower_bounds + (act + 1)/2.0 * (self.action_upper_bounds - self.action_lower_bounds)

    def step(self, action_normalized, restore=False):
        assert action_normalized.shape[0] == self.n_action and -1.0 <= action_normalized.all() <= 1.0
        action = self._act_norm2actual(action_normalized)
        if self.timestep == 0:
            self.action_filter.init_history(action)
        # post-process action
        if self.lpf_action:
            action_filtered = self.action_filter.filter(action)
            # u_to_apply = action_in_SI*self._acs_alpha + self.last_acs*(1-self._acs_alpha)
        else:
            action_filtered = np.copy(action)

        for i in range(self.num_sims_per_env_step):
            self.sim.step(action_filtered)

        self._update_data(step=True)
        self.last_act = action
        obs_vf, obs_pol = self._get_obs(action=action, step=True)
        reward, reward_dict = self._get_reward(action)
        self.info["reward_dict"] = reward_dict
        terminated = self._terminated()
        truncated = False
        

        return obs_vf, reward, terminated, truncated, self.info

    def _update_data(self, step=True):
        # NOTE: modify obs states, ground truth states 
        self.obs_states = self.sim.get_obseverable()
        self.gt_states = self.sim.states
        if step:
            self.timestep += 1
            self.time_in_sec += self.secs_per_env_step
            # self.time_in_sec = self.sim.time
            # self.reference_generator.update_ref_env(self.time_in_sec)

    def _get_reward(self, action):
        names = ['position_error', 'velocity_error', 'attitude_error', 'input', 'delta_acs']

        w_position    = 80.0 # 80.0
        w_velocity    = 20.0
        w_orientation = 20.0
        w_input       = 10.0
        w_delta_act   = 10.0 #40.0 #3.0 #5.0

        reward_weights = np.array([w_position, w_velocity, w_orientation, w_input, w_delta_act])
        weights = reward_weights / np.sum(reward_weights)  # weight can be adjusted later

        scale_pos       = 10.0
        scale_vel       = 1.0
        scale_ori       = 1.0
        scale_input     = 5.0
        scale_delta_act = 1.0  # 2.0

        desired_pos = np.array([0.0, 0.0, 2.0]).reshape(3,1) # x y z 
        desired_vel = np.array([0.0, 0.0, 0.0]).reshape(3,1) # vx vy vz
        desired_ori = np.array([0.0, 0.0, 0.0]).reshape(3,1) # roll, pitch, yaw
        current_pos = self.sim.get_position()
        current_vel = self.sim.get_velocity()
        current_att = self.sim.get_orientation() # euler_mes
        
        pos_err = np.linalg.norm(current_pos - desired_pos) 
        r_pos = np.exp(-scale_pos * pos_err)

        vel_err = np.linalg.norm(current_vel- desired_vel) 
        r_vel = np.exp(-scale_vel * vel_err)  # scale_vel need to be adjust later

        ori_err = np.linalg.norm(current_att- desired_ori)
        r_ori = np.exp(-scale_ori * ori_err)

        input_err = np.linalg.norm(action) 
        r_input = np.exp(-scale_input * input_err)

        delta_act_err = np.linalg.norm(action - self.last_act) 
        r_delta_act = np.exp(-scale_delta_act * delta_act_err)

        rewards = np.array([r_pos, r_vel, r_ori, r_input, r_delta_act])
        total_reward = np.sum(weights * rewards)
        reward_dict = dict(zip(names, weights * rewards))

        return total_reward, reward_dict

    def _terminated(self):
        pos_ub = np.array([20., 20., 20.])
        pos_lb = np.array([-20., -20., 0])

        if not ((self.sim.robot_pos <= pos_ub).all() and (self.sim.robot_pos >= pos_lb).all()):
            print("Out of bounds ", self.timestep)
            if self.debug: print("Out of position bounds")
            return True
        if self.timestep >= self.max_timesteps:
            if self.debug: print("Max step reached: {}".format(self.max_timesteps))
            return True
        else:
            return False

    def test(self, model):
        for i in range(5):
            obs = self.reset()
            # self.debug = True
            self.max_timesteps = 0.5 * self.max_timesteps

            log = {
                "t": np.empty((0, 1)),
                "x": np.empty((0, 6)),
                "xd": self.goal,
                "u": np.empty((0, 3)),
            }
            t = 0
            log["x"] = np.append(
                log["x"],
                np.array(
                    [np.concatenate((self.estimator.pos(), self.estimator.vel()))]
                ),
                axis=0,
            )
            log["t"] = np.append(log["t"], t)
            total_reward = 0
            for i in range(5000):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, info = self.step(action)
                total_reward += reward
                t = t + self.dt
                log["x"] = np.append(
                    log["x"],
                    np.array(
                        [np.concatenate((self.estimator.pos(), self.estimator.vel()))]
                    ),
                    axis=0,
                )
                log["t"] = np.append(log["t"], t)
                log["u"] = np.append(log["u"], np.array([self.last_act]), axis=0)
                if terminated:
                    print(f"total reward: {total_reward}")
                    total_reward = 0
                    self.plot(log)
                    obs = self.reset()
                    log = {
                        "t": np.empty((0, 1)),
                        "x": np.empty((0, 6)),
                        "xd": self.goal,
                        "u": np.empty((0, 3)),
                    }
                    t = 0
                    log["x"] = np.append(
                        log["x"],
                        np.array(
                            [
                                np.concatenate(
                                    (self.estimator.pos(), self.estimator.vel())
                                )
                            ]
                        ),
                        axis=0,
                    )
                    log["t"] = np.append(log["t"], t)
            self.debug = False
            print(f"testing complete")
            self.plot(log)
        return log

    def close(self):
        return