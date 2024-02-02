import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility_functions import *
from func_eom import *
from func_initial_joint_angles import *
from symbolic_functions.func_wing_kinematic_vel import func_wing_kinematic_vel

class Flappy():

    def __init__(self, p, render):
        self.p             = p
        self.dt            = p.dt # 1/2000
        self.sim_freq      = int(1.0 / self.dt)
        self.render        = render
        self.flapping_freq = 4.75
        self._init_const()
        self.reset()
        self.get_obseverable()

    # region: Properties

    @property
    def freq(self):
        return np.copy(self.sim_freq)

    @property
    def states(self):
        return np.copy(self._states)

    @property
    def robot_pos(self):
        return self.get_position() 
    
    @property
    def robot_vel(self):
        return self.get_velocity() 
    
    @property
    def robot_ori(self):
        return self.get_orientation()

    def get_position(self):
        position = self.xd[2:5]
        return position 

    def get_velocity(self):
        velocity = self.xd[7:10]
        return velocity

    def get_orientation(self):      
        R_body = (self.xd[13:22].reshape(3,3)).T # body to inertial
        pitch = np.arcsin(np.clip(-R_body.T[0,2],-1,1))
        yaw = np.arctan2(R_body.T[0,1], R_body.T[0,0])
        roll = np.arctan2(R_body.T[1, 2], R_body.T[2,2])
        orientation = np.array([pitch, yaw, roll])
        return orientation

    def get_orientation_vel(self):
        pitch_rate=self.xd[10]
        yaw_rate=self.xd[11]
        roll_rate=self.xd[12]
        return pitch_rate, yaw_rate, roll_rate
    
    def get_obseverable(self):
        return np.copy(self.states) # NOTE: HACK there, need to replace to what we can measure onboard
    
    def inputs(self):
        self.u_gain = np.zeros(9,)

    # endregion

    def _init_const(self):
        self.wc = np.exp(-self.p.wc_lp * 2 * np.pi * self.dt)

    def _init_param(self):
        self.pitch_error = 0.
        self.roll_error = 0.
        self.vel_error = 0.

        self.pitch_error_i = 0.
        self.roll_error_i = 0.
        self.vel_error_i = 0.

        self.pitch_error_d = 0.
        self.roll_error_d = 0.
	
    def _init_states(self):
        xk_init = np.zeros(14,)
        xd_init = np.zeros(22,)
        xa_init = np.zeros(3 * (self.p.n_blade - self.p.n_blade_tail),)

        Rb0 = rot_y(self.p.pitch_init) @ rot_x(self.p.roll_init)  # initial pitch and roll angle
        xd_init[13:22] = Rb0.T.reshape(9,)
        xk_init[0:7] = func_initial_joint_angles(self.p.theta1_init, self.p.wing_conformation).reshape(7,)

        # Joint velocities
        kinematic_vel_out = func_wing_kinematic_vel(xk_init[0:7], self.p.flapping_freq * 2 * np.pi, self.p.wing_conformation.flatten())
        A_kv = kinematic_vel_out[0:7,:]
        h_kv = kinematic_vel_out[7,:]
        u_kv = kinematic_vel_out[8,:]
        xk_init[7:14] = np.linalg.solve(A_kv, -h_kv + u_kv)
    
    	# Wing initial states(dynamics), follows the same values as KS states
        xd_init[0:2] = xk_init[3:5]
        xd_init[5:7] = xk_init[10:12]
    	# Body initial velocity
        xd_init[7:10] = self.p.body_init_vel.reshape(3,)
   
        return xk_init, xd_init, xa_init
    
    def reset(self):
        self._init_param()
        self.xk, self.xd, self.xa = self._init_states()
        self._states = np.concatenate([self.xk, self.xd, self.xa])
    
    def step(self, u_gain):
        ###################################
        ##### original PID controller #####
        ###################################
        # u1 becomes to (9,), including p.KP_P, p.KI_P, p.KD_P, p.KP_R, p.KI_R, p.KD_R, p.KP_V, p.KI_V, kd
        # u_gain=np.array([KP_P,KI_P,KD_P, KP_R,KI_R,KD_R, KP_V,KI_V,kd])
        
        R_body = self.xd[13:22].reshape(3,3).T  # %body to inertial
        pitch = np.arcsin(np.clip(-R_body.T[0,2],-1,1))
        yaw = np.arctan2(R_body.T[0,1], R_body.T[0,0])
        roll = np.arctan2(R_body.T[1,2], R_body.T[2,2])

        # Prevent roll - over from -pi to pi
        if (roll > np.pi * 0.99):
            pitch = -np.pi - pitch

        vel_forward = self.xd[7]

        # Roll and pitch error
        pitch_error_old = self.pitch_error
        roll_error_old = self.roll_error
        self.pitch_error = self.p.pitch_ref - pitch
        self.roll_error = self.p.roll_ref - roll
        self.vel_error = self.p.vel_ref - vel_forward

        # Derivative, use low pass to prevent huge spikes
        pitch_rate = (self.pitch_error - pitch_error_old) / self.p.dt
        roll_rate = (self.roll_error - roll_error_old) / self.p.dt
        self.pitch_error_d = self.wc * self.pitch_error_d + (1 - self.wc) * pitch_rate
        self.roll_error_d = self.wc * self.roll_error_d + (1 - self.wc) * roll_rate

        # Integral
        self.pitch_error_i = self.pitch_error_i + self.pitch_error * self.p.dt
        self.roll_error_i = self.roll_error_i + self.roll_error * self.p.dt
        self.vel_error_i = self.vel_error_i + self.vel_error * self.p.dt

        # Desired u should be obtainbed by DRL, u_pitch, u_roll, u_vel should be intialized
        # u_gain=np.array([KP_P, KI_P, KD_P, KP_R, KI_R, KD_R, KP_V, KI_V, kd])
        # u_pitch = p.pitch_dir * (p.KP_P * self.pitch_error + p.KI_P * self.pitch_error_i + p.KD_P * self.pitch_error_d)
        u_pitch = self.p.pitch_dir * (u_gain[0] * self.pitch_error + u_gain[1] * self.pitch_error_i + u_gain[2] * self.pitch_error_d)
        u_roll = self.p.roll_dir * (u_gain[3] * self.roll_error + u_gain[4] * self.roll_error_i + u_gain[5] * self.roll_error_d)
        u_vel = u_gain[6] * self.vel_error + u_gain[7] * self.vel_error_i
        # u_roll = p.roll_dir * (p.KP_R * self.roll_error + p.KI_R * self.roll_error_i + p.KD_R * self.roll_error_d)
        # u_vel = p.KP_V * self.vel_error + p.KI_V * self.vel_error_i

        # Pitch output forces
        x1 = u_vel + u_pitch
        x2 = u_vel - u_pitch
        x1 = np.clip(x1, 0.0001, self.p.thruster_max_force)
        x2 = np.clip(x2, 0.0001, self.p.thruster_max_force)

        # Roll output forces
        y1 = u_roll
        y2 = -u_roll
        y1 = np.clip(y1, 0.0001, self.p.thruster_max_force)
        y2 = np.clip(y2, 0.0001, self.p.thruster_max_force)

        # if (time(i) > p.wait_time):
        #     u1 = p.kd * (p.flapping_freq * 2 * np.pi - xk_init[7])
        # else:
        #     u1 = 0

        f_thruster = np.zeros((3,4))  # inertial force

        f1 = R_body @ (np.array([x1, 0, 0])).reshape(3,1)
        f2 = R_body @ (np.array([x2, 0, 0])).reshape(3,1)
        f3 = R_body @ (np.array([0, -y1, 0])).reshape(3,1)
        f4 = R_body @ (np.array([0, y2, 0])).reshape(3,1)

        f_thruster = np.concatenate([f1, f2, f3, f4], axis=1) # 3x4
        t_thruster = np.zeros((3,4))
        yaw_rate = self.xd[12]
        kd = u_gain[8]
        yaw_damper = -kd * yaw_rate
        #yaw_damper = -u_gain[8] * yaw_rate
        yaw_damping = np.array([0, 0, yaw_damper])

        u1 = 0
        # sim
        # tik = time.time()
        fk1, fd1, fa1 = func_eom(self.xk, self.xd, self.xa, u1, f_thruster, t_thruster, self.p, yaw_damping)
        # tok1 = time.time()
        fk2, fd2, fa2 = func_eom(self.xk + fk1 * self.dt / 2, self.xd + fd1 * self.dt / 2, self.xa + fa1 * self.dt / 2, u1, f_thruster, t_thruster, self.p, yaw_damping)
        # tok2 = time.time()
        fk3, fd3, fa3 = func_eom(self.xk + fk2 * self.dt / 2, self.xd + fd2 * self.dt / 2, self.xa + fa2 * self.dt / 2, u1, f_thruster,t_thruster, self.p, yaw_damping)
        # tok3 = time.time()
        fk4, fd4, fa4 = func_eom(self.xk + fk3 * self.dt, self.xd + fd3 * self.dt, self.xa + fa3 * self.dt, u1, f_thruster, t_thruster, self.p, yaw_damping)
        # tok4 = time.time()
        # print('step time:', tok1-tik, tok2-tik, tok3-tik, tok4-tik)

        self.xk = self.xk + (fk1/6 + fk2/3 + fk3/3 + fk4/6) * self.dt
        self.xd = self.xd + (fd1/6 + fd2/3 + fd3/3 + fd4/6) * self.dt
        self.xa = self.xa + (fa1/6 + fa2/3 + fa3/3 + fa4/6) * self.dt
        self._states = np.concatenate([self.xk, self.xd, self.xa])

