from rotation_transformations import *
from utility_functions import *

def R_body(model, data):
    
    qpos = data.qpos
    quat = qpos[3:7]
    #check wether this is still the case when the freejoint is disabled.
    #The data.qpos vector has only 14 element and not 21 anymore. But removing the freejoint should delete 6DOF, not 7.
    #so mujoco might compute euler angles directly, and not quaternions, which should be given by qpos[3:6]
    N=len(data.qpos)
    
    if N == 21:
        angles = quat2euler(quat)
    else:
        angles = [0,0,0]
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]
    
    zrot = rot_z(yaw)
    yrot = rot_y(pitch)
    xrot = rot_x(roll)
    R_body = np.dot(zrot,np.dot(yrot,xrot))
    
    return R_body