import numpy as np
from numba import jit
import sys, os
sys.path.append('/Users/mintaekim/Desktop/Hybrid Robotics Lab/Flappy/Integrated/Flappy_Integrated/flappy_v2/envs')
sys.path.append('/Users/mintaekim/Desktop/Hybrid Robotics Lab/Flappy/Integrated/Flappy_Integrated/flappy_v2')
from symbolic_functions.func_wing_tail import *
from func_create_wing_segment import func_create_wing_segment
from parameter import Simulation_Parameter
from utility_functions import *
#from utility_functions.R_body import R_body_f
#from utility_functions.rotation_transformations import *

@jit(nopython=True,fastmath=True)
def lift_coeff(a, params):
    return params[0] + params[1] * np.sin(np.deg2rad(params[2] * np.rad2deg(a) + params[3]))

@jit(nopython=True,fastmath=True)
def drag_coeff(a, params):
    return params[0] + params[1] * np.cos(np.deg2rad(params[2] * np.rad2deg(a) + params[3]))

def original_states(model, data):
    #takes mujoco states vectors and converts to MATLAB states vectors defined in func_eom
    qvel = data.qvel
    qpos = data.qpos
    N = len(qpos)

    xd = np.array([0.0]*22)
    xd[0] = qpos[posID_dic["L5"]]
    xd[1] = qpos[posID_dic["L6"]]
    if N == 21:
        xd[2:5] = list(qpos[0:3])
    else:
        xd[2:5] = [0,0,0.5]

    xd[5] = qvel[jvelID_dic["L5"]]
    xd[6] = qvel[jvelID_dic["L6"]]
    
    if N==21:
        xd[7:10] = list(qvel[0:3])
        xd[10:13] = list(qvel[3:6])
    else:
        xd[7:10] = [0,0,0]
        xd[10:13] = [0,0,0]

    R_B= R_body(model,data)
    xd[13:23] = list(np.transpose(R_B).flatten())
    return xd, R_B

def aero(model, data, xa):
    xd, R_body = original_states(model, data)

    # initialize output
    xa = xa.T
    fa = np.zeros_like(xa)  # output dxa/dt
    ua = 0  # sum of generalized aerodynamics forces
    xa_m = xa.reshape(3,p.nWagner).T  # reshape xa for easy access

    # --------------------------------------------------------------------------
    # calculate blade element inertial position, velocities,
    # and Ba (mapping for generalized forces)
    # --------------------------------------------------------------------------

    # Ba is the inertial force to generalized force transformation matrix for
    # calculating ua
    pos_s = np.zeros((3,p.n_blade))  # inertial position
    vel_s = np.zeros((3,p.n_blade))  # inertial velocity
    Ba_s = np.zeros((8,3,p.n_blade))  # ua = Ba*fa, fa = inertial aero force
    vel_s_surf = np.zeros((3,p.n_blade))  # velocity about the wing axis [front, left, normal]
    e_n = np.zeros((3,p.n_blade))  # wing's surface normal direction
    aoa = np.zeros(p.n_blade)  # angle of attack (free stream)
    #aoa_d = np.zeros(n_blade)  # change in angle of attack due to downwash
    U = np.zeros(p.n_blade)  # effective air speed
    e_effvel = np.zeros((3,p.n_blade))

    strip_id, strip_dc, strip_c, strip_theta = func_create_wing_segment(p)

    for i in range(p.n_blade):
        # check which wing segment this blade element index belongs to
        if strip_id[i] == 1:  # hL
            pos, vel, Ba = func_wing_hL(xd, strip_dc[0:2, i], p.wing_conformation.flatten())
            Rw = rot_x(xd[0])
        elif strip_id[i] == 2:  # rL
            pos, vel, Ba = func_wing_rL(xd, strip_dc[0:2, i], p.wing_conformation.flatten())
            Rw = rot_x(xd[1])
        elif strip_id[i] == 3:  # hR
            pos, vel, Ba = func_wing_hR(xd, strip_dc[0:2, i], p.wing_conformation.flatten())
            Rw = rot_x(-xd[0])
        elif strip_id[i] == 4:  # rR
            pos, vel, Ba = func_wing_rR(xd, strip_dc[0:2, i], p.wing_conformation.flatten())
            Rw = rot_x(-xd[1])
        elif strip_id[i] == 5:  # body-fixed (e.g., tail)
            pos, vel, Ba = func_tail(xd, strip_dc[0:3, i])
            Rw = np.eye(3)

        # wing velocities about wing segment's axis [front, left, normal]
        # local effective velocity (wing frame)
        vel_surf = np.dot(np.transpose(Rw), np.dot(np.transpose(R_body), vel - p.airspeed))
        vel_surf_norm = np.linalg.norm(vel_surf)  # air velocity magnitude

        # effective velocity direction (x, z)
        if np.linalg.norm(vel_surf[[0, 2]]) < 1e-6:
            ev = np.zeros(3)  # prevent dividing by zero
        else:
            ev = vel_surf / vel_surf_norm  # unit vector of air flow direction

        alpha_inf = np.arctan2(-vel_surf[2], vel_surf[0])[0]  # angle of attack

        # record values
        pos_s[:, i] = pos.flatten()
        vel_s[:, i] = np.transpose(vel - p.airspeed)
        vel_s_surf[:, i] = np.transpose(vel_surf)
        Ba_s[:, :, i] = Ba
        e_n[:, i] = np.dot(np.dot(R_body, Rw), np.array([0, 0, 1]))
        aoa[i] = alpha_inf
        e_effvel[:, i] = ev.flatten()  # effective velocity direction (local)

        if vel_surf_norm < 1e-4:
            U[i] = 1e-4  # prevents dividing by zero
        else:
            U[i] = vel_surf_norm
    # -----------------------------------------------------------------------
    #  Determine An and bn for An*an_dot = bn to solve for fa
    #  Follows Boutet formulations
    #------------------------------------------------------------------------

    # Calculate An
    An = np.zeros((p.nWagner, p.nWagner))
    n = np.arange(1, p.nWagner+1)
    # sin_n_strip_theta = np.sin(np.outer(np.arange(1, nWagner + 1), strip_theta))
    for i in range(p.nWagner):
        An[i, :] = p.a0 * p.c0 / U[i] * np.sin(n * strip_theta[i])
        # An[i, :] = a0 * c0 / U[i] * sin_n_strip_theta[i]

    # Calculate bn
    an = xa_m[:, 0]
    bn = np.zeros((p.nWagner,1))
    fa_m = np.zeros((p.nWagner,3))
    for i in range(p.nWagner):
        # aero states and effective air speed
        z1 = xa_m[i, 1]
        z2 = xa_m[i, 2]
        # downwash due to vortex
        wy = -p.a0 * p.c0 * U[i] / 4 / p.span_max * np.dot((n * np.sin(n * strip_theta[i])) / np.sin(strip_theta[i]),an)
        # effective downwash
        wn = vel_s_surf[2, i]
        w = wn + wy
        # alpha_downwash = np.arctan2(-w, vel_surf[0])[0] - aoa[i]
        bn[i] = -p.a0 * p.c0 / strip_c[i] * np.dot(np.sin(n * strip_theta[i]), an) + p.a0 * (w * p.Phi_0 / U[i] + p.phi_a[0] * p.phi_b[0] / (strip_c[i] / 2) * z1 + p.phi_a[1] * p.phi_b[1] * z2)
        # dz1/dt and dz2/dt
        fa_m[i,1] = U[i] * wn + p.phi_b[0] / (strip_c[i] / 2) * z1 #
        fa_m[i,2] = U[i] * wy + p.phi_b[1] / (strip_c[i] / 2) * z2 #

    # calculate aerodynamic states rate of change for an
    andot = np.linalg.solve(An, bn)  # dan/dt
    fa_m[:, 0] = np.transpose(andot)
    # restructure fa_m back into vector format
    fa = np.transpose(fa_m).flatten()
    
    #---------------------------------------------------------------
    # Calculate aerodynamics lift and drag
    for i in range(p.n_blade):
        # strip width
        if strip_id[i] == 1 or strip_id[i] == 3:
            # proximal wing
            delta_span = p.span_prox / p.n_blade_prox
        elif strip_id[i] == 2 or strip_id[i] == 4:
            # distal wing
            delta_span = p.span_dist / p.n_blade_dist
        else:
            # tail
            delta_span = p.tail_width / p.n_blade_tail

        # rotation matrix from body axis to wing axis
        if strip_id[i] == 1:  # hL
            Rw = rot_x(xd[0])
        elif strip_id[i] == 2:  # rL
            Rw = rot_x(xd[1])
        elif strip_id[i] == 3:  # hR
            Rw = rot_x(-xd[0])
        elif strip_id[i] == 4:  # rR
            Rw = rot_x(-xd[1])
        elif strip_id[i] == 5:  # tail
            Rw = np.eye(3)

        # lift coefficient (wagner for the wing, if enabled)
        if i < p.nWagner:
            #Gamma = 0.5 * a0 * c0 * U[i] * np.sin(strip_theta[i]) * an
            C_lift = -p.a0 * np.dot(np.sin(n*strip_theta[i]),(an + np.dot(strip_c[i] / U[i],fa_m[:, 0])))
        else:
            # quasi-steady model
            #Gamma = 0
            C_lift = lift_coeff(aoa[i], p.aero_model_lift)

        # Drag coefficient (quasi-steady)
        C_drag = drag_coeff(aoa[i], p.aero_model_drag)

        e_lift = np.cross(e_effvel[:, i], np.array([0, 1, 0]))
        e_drag = -e_effvel[:, i]

        lift = p.air_density / 2 * U[i] ** 2 * C_lift * delta_span * strip_c[i] * e_lift  # Wagner
        drag = p.air_density / 2 * U[i] ** 2 * C_drag * delta_span * strip_c[i] * e_drag  # quasisteady

        # Combine the drag and lift directions in inertial frame
        f_aero = np.dot(np.dot(R_body, Rw), (drag + lift))
        ua += np.dot(Ba_s[:, :, i], f_aero) #shape (1,8) for d2(theta5, theta6, x, y, z,roll,pitch,yaw)/dt2
 
    # Dynamics EOM, Md*accel + hd = ua + Jc'*lambda
    return fa, ua