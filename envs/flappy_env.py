# Helpers
import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2/envs')
import numpy as np
from typing import Dict, Union
from collections import deque
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import time
# Gym
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import utils
from gymnasium.utils import seeding
# Mujoco
import mujoco as mj
from mujoco_gym.mujoco_env import MujocoEnv
# Flappy
from dynamics import Flappy
from parameter import Simulation_Parameter
from aero_force import aero
from action_filter import ActionFilterButter
from env_randomize import EnvRandomizer
from utility_functions import *
import utility_trajectory as ut
from rotation_transformations import *
from R_body import R_body


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0,}
TRAJECTORY_TYPES = {"linear": 0, "circular": 1, "setpoint": 2}

class FlappyEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 3000,
        is_visual     = False,
        randomize     = False,
        debug         = False,
        lpf_action    = True,
        traj_type     = False,
        # MujocoEnv
        xml_file: str = "../assets/Flappy_v3.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs
    ):
        # Dynamics simulator
        self.p = Simulation_Parameter()
        self.sim = Flappy(p=self.p, render=is_visual)

        # Frequency
        self.max_timesteps         = max_timesteps
        self.timestep              = 0.0
        self.sim_freq              = self.sim.freq # NOTE: 2000Hz for hard coding
        # self.dt                    = 1.0 / self.sim_freq # NOTE: 1/2000s for hard coding
        self.policy_freq           = 30 # NOTE: 30Hz but the real control frequency might not be exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.policy_freq # 2000//30 = 66
        self.secs_per_env_step     = self.num_sims_per_env_step / self.sim_freq # 66/2000 = 0.033s
        self.policy_freq           = int(1.0/self.secs_per_env_step) # 1000/33 = 30Hz

        self.is_visual          = is_visual
        self.randomize          = randomize
        self.debug              = debug
        self.is_plotting        = True
        self.traj_type          = traj_type
        self.noisy              = False
        self.randomize_dynamics = False # True to randomize dynamics
        self.lpf_action         = lpf_action # Low Pass Filter 

        # Observation, need to be reduce later for smoothness
        self.n_state            = 84 # NOTE: change to the number of states *we can measure*
        self.n_action           = 7  # NOTE: change to the number of action
        self.history_len_short  = 4
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.previous_obs       = deque(maxlen=self.history_len)
        self.previous_act       = deque(maxlen=self.history_len)
        
        self.action_space = Box(low=-100, high=100, shape=(self.n_action,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(41,)) # NOTE: change to the actual number of obs to actor policy
        self.observation_space_policy = Box(low=-np.inf, high=np.inf, shape=(454,)) # NOTE: change to the actual number of obs to actor policy
        self.observation_space_value_func = Box(low=-np.inf, high=np.inf, shape=(454,)) # NOTE: change to the actual number of obs to the value function
        
        # NOTE: the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.pos_lb = np.array([-20, -20, 0]) # fight space dimensions: xyz
        self.pos_ub = np.array([20, 20, 20])
        self.vel_lb = np.array([-2, -2, -2])
        self.vel_ub = np.array([2, 2, 2])

        self.action_lower_bounds = np.array([-30,0,0,0,0,0,0])
        self.action_upper_bounds = np.array([0,2,2,2,2,0.5,0.5])
        self.action_bounds_scale = 0.2
        self.action_lower_bounds_actual = self.action_lower_bounds + self.action_bounds_scale * self.action_upper_bounds
        self.action_upper_bounds_actual = (1 - self.action_bounds_scale) * self.action_upper_bounds
        
        # MujocoEnv
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        self.body_list = ["Base","L1","L2","L3","L4","L5","L6","L7",
                          "L1R","L2R","L3R","L4R","L5R","L6R","L7R"]
        self.joint_list = ['J1','J2','J3','J5','J6','J7','J10',
                           'J1R','J2R','J3R','J5R','J6R','J7R','J10R']
        self.bodyID_dic, self.jntID_dic, self.posID_dic, self.jvelID_dic = self.get_bodyIDs(self.body_list)
        self.jID_dic = self.get_jntIDs(self.joint_list)
        
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=self.observation_space, default_camera_config=default_camera_config, **kwargs)
        
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        self.xa = np.zeros(3 * self.p.n_Wagner) # aero()

        # Info for normalizing the state
        self._init_action_filter()
        # self._init_env_randomizer() # NOTE: Take out dynamics randomization first 
        self._seed()
        self.reset()
        self._init_env()

        self.reward_plot = []

    @property
    def dt(self) -> float:
        # return self.model.opt.timestep * self.frame_skip # 4e-3
        return 2e-5

    def _init_env(self):
        print("Environment created")
        action = self.action_space.sample()
        print("Sample action: {}".format(action))
        print("Control range: {}".format(self.model.actuator_ctrlrange))
        print("Time step(dt): {}".format(self.dt))
        
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
        obs = self._get_obs()
        info = self._get_reset_info
        return obs, info
    
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

    def _init_env_randomizer(self):
        self.env_randomizer = EnvRandomizer(self.sim)

    def _set_dynamics_properties(self):
        if self.randomize_dynamics:
            self.sim.set_dynamics()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def _act_norm2actual(self, act):
        return self.action_lower_bounds_actual + (act + 1)/2.0 * (self.action_upper_bounds_actual - self.action_lower_bounds_actual)

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

        self.do_simulation(action_filtered, self.frame_skip)
        self._update_data(step=True)
        self.last_act = action

        # obs_vf, obs_pol = self._get_obs(action=action, step=True)
        obs = self._get_obs()
        reward, reward_dict = self._get_reward(action)
        self.info["reward_dict"] = reward_dict

        if self.render_mode == "human":
            self.render()

        terminated = self._terminated()
        truncated = False
        self.reward_plot.append(reward)
        
        return obs, reward, terminated, truncated, self.info
    
    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        for _ in range(self.num_sims_per_env_step):
            self.data.ctrl[:] = ctrl
            self.data.ctrl[0] = -29.8451302

            # self.xd, R_body = self._get_original_states()
            # fa, ua, self.xd = aero(self.model, self.data, self.xa, self.xd, R_body)
            # # Apply Aero forces
            # self.data.qfrc_applied[self.jvelID_dic["L5"]] = ua[0]
            # self.data.qfrc_applied[self.jvelID_dic["L6"]] = ua[1]
            # self.data.qfrc_applied[0:6] = ua[2:8]
            # # Integrate Aero States
            # self.xa = self.xa + fa * self.dt

            mj.mj_step(self.model, self.data, nstep=n_frames)
            mj.mj_rnePostConstraint(self.model, self.data)

    def _get_original_states(self):
        # Takes mujoco states vectors and converts to MATLAB states vectors defined in func_eom
        qpos = self.data.qpos
        qvel = self.data.qvel     
        N = len(qpos)

        xd = np.zeros(22)
        xd[0] = qpos[self.posID_dic["L5"]]
        xd[1] = qpos[self.posID_dic["L6"]]
        if N == 21:
            xd[2:5] = qpos[0:3]
        else:
            xd[2:5] = np.array([0,0,0.5])

        xd[5] = qvel[self.jvelID_dic["L5"]]
        xd[6] = qvel[self.jvelID_dic["L6"]]
        
        if N == 21:
            xd[7:10] = qvel[0:3]
            xd[10:13] = qvel[3:6]
        else:
            xd[7:10] = np.zeros(3)
            xd[10:13] = np.zeros(3)

        R_B = R_body(self.model, self.data)
        xd[13:23] = np.transpose(R_B).flatten()
        return xd, R_B

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
        names = ['position_error', 'velocity_error', 'orientation_error', 'input', 'delta_acs']

        w_position    = 1.0
        w_velocity    = 1.0
        w_orientation = 0.2
        w_input       = 0.002
        w_delta_act   = 0.01

        reward_weights = np.array([w_position, w_velocity, w_orientation, w_input, w_delta_act])
        weights = reward_weights / np.sum(reward_weights)  # weight can be adjusted later

        scale_pos       = 1.0
        scale_vel       = 1.0
        scale_ori       = 1.0
        scale_input     = 1.0
        scale_delta_act = 1.0

        desired_pos = np.array([0.0, 0.0, 0.5]).reshape(3,1) # x y z 
        desired_vel = np.array([0.0, 0.0, 0.0]).reshape(3,1) # vx vy vz
        desired_ori = np.array([0.0, 0.0, 0.0]).reshape(3,1) # roll, pitch, yaw
        current_pos = self.data.qpos
        current_vel = self.data.qvel
        current_ori = quat2euler_raw(self.data.qpos[3:7]) # euler_mes
        
        pos_err = np.linalg.norm(current_pos - desired_pos) 
        r_pos = np.exp(-scale_pos * pos_err)

        vel_err = np.linalg.norm(current_vel- desired_vel) 
        r_vel = np.exp(-scale_vel * vel_err)  # scale_vel need to be adjust later

        ori_err = np.linalg.norm(current_ori- desired_ori)
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
        # if not((self.data.qpos[0:3] <= self.pos_ub).all() 
        #         and (self.data.qpos[0:3] >= self.pos_lb).all()):
        #     print("Out of position bounds ", self.data.qpos[0:3], self.timestep)
        #     return True
        # if not((self.data.qvel[0:3] <= self.vel_ub).all() 
        #         and (self.data.qvel[0:3] >= self.vel_lb).all()):
        #     print("Out of velocity bounds ", self.data.qvel[0:3], self.timestep)
        #     return True
        if self.timestep >= self.max_timesteps:
            print("Max step reached: {}".format(self.max_timesteps))
            return True
        else:
            return False

    def get_bodyIDs(self, body_list):
        bodyID_dic = {}
        jntID_dic = {}
        posID_dic = {}
        jvelID_dic = {}
        for bodyName in body_list:
            mjID = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, bodyName)
            jntID = self.model.body_jntadr[mjID]   # joint ID
            jvelID = self.model.body_dofadr[mjID]  # joint velocity
            posID = self.model.jnt_qposadr[jntID]  # joint position
            bodyID_dic[bodyName] = mjID
            jntID_dic[bodyName] = jntID
            posID_dic[bodyName] = posID
            jvelID_dic[bodyName] = jvelID
        return bodyID_dic, jntID_dic, posID_dic, jvelID_dic

    def get_jntIDs(self, jnt_list):
        jointID_dic = {}
        for jointName in jnt_list:
            jointID = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jointName)
            jointID_dic[jointName] = jointID
        return jointID_dic

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
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()