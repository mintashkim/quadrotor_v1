import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def func_wing_hL(in1 = None, in2 = None, in3 = None):

    L3c = in3[6]
    P5_1 = in3[19]
    P5_2 = in3[20]
    Rb_1_1 = in1[13]
    Rb_1_2 = in1[16]
    Rb_1_3 = in1[19]
    Rb_2_1 = in1[14]
    Rb_2_2 = in1[17]
    Rb_2_3 = in1[20]
    Rb_3_1 = in1[15]
    Rb_3_2 = in1[18]
    Rb_3_3 = in1[21]
    alpha_3 = in3[7]
    da_hL_1 = in2[0]
    da_hL_2 = in2[1]
    offset_x = in3[23]
    theta_WL_1 = in1[0]
    thetaD_WL_1 = in1[5]
    w_1 = in1[10]
    w_2 = in1[11]
    w_3 = in1[12]
    xD_1 = in1[7]
    xD_2 = in1[8]
    xD_3 = in1[9]
    x_1 = in1[2]
    x_2 = in1[3]
    x_3 = in1[4]
    
    cos_theta_WL_1 = np.cos(theta_WL_1)
    sin_theta_WL_1 = np.sin(theta_WL_1)
    cos_alpha_3 = np.cos(alpha_3)
    sin_alpha_3 = np.sin(alpha_3)

    pos_aero_hum_L = np.array([[x_1 + Rb_1_2 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_3 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_1 * (da_hL_1 + offset_x)],[x_2 + Rb_2_2 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_3 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_1 * (da_hL_1 + offset_x)],[x_3 + Rb_3_2 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_3 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_1 * (da_hL_1 + offset_x)]])
    
    # if nargout > 1
    mt1 = [xD_1 + (da_hL_1 + offset_x) * (Rb_1_2 * w_3 - Rb_1_3 * w_2) + (Rb_1_1 * w_2 - Rb_1_2 * w_1) * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - (Rb_1_1 * w_3 - Rb_1_3 * w_1) * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_1_2 * (thetaD_WL_1 * sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1) + Rb_1_3 * (thetaD_WL_1 * cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1)]
    mt2 = [xD_2 + (da_hL_1 + offset_x) * (Rb_2_2 * w_3 - Rb_2_3 * w_2) + (Rb_2_1 * w_2 - Rb_2_2 * w_1) * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - (Rb_2_1 * w_3 - Rb_2_3 * w_1) * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_2_2 * (thetaD_WL_1 * sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1) + Rb_2_3 * (thetaD_WL_1 * cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1)]
    mt3 = [xD_3 + (da_hL_1 + offset_x) * (Rb_3_2 * w_3 - Rb_3_3 * w_2) + (Rb_3_1 * w_2 - Rb_3_2 * w_1) * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - (Rb_3_1 * w_3 - Rb_3_3 * w_1) * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_3_2 * (thetaD_WL_1 * sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1) + Rb_3_3 * (thetaD_WL_1 * cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1)]
    vel_aero_hum_L = np.array([mt1, mt2, mt3]).reshape(3,1)
    
    # if nargout > 2
    mt4 = [- Rb_1_2 * (sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_3 * (cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),0.0,1.0,0.0,0.0,- Rb_1_2 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_3 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),Rb_1_1 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_1_3 * (da_hL_1 + offset_x),- Rb_1_1 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_2 * (da_hL_1 + offset_x),- Rb_2_2 * (sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_3 * (cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),0.0,0.0,1.0,0.0]
    mt5 = [- Rb_2_2 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_3 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),Rb_2_1 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_2_3 * (da_hL_1 + offset_x),- Rb_2_1 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_2 * (da_hL_1 + offset_x),- Rb_3_2 * (sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_3 * (cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),0.0,0.0,0.0,1.0,- Rb_3_2 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_3 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1)]
    mt6 = [Rb_3_1 * (P5_2 + sin_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_3_3 * (da_hL_1 + offset_x),- Rb_3_1 * (P5_1 + cos_theta_WL_1 * (da_hL_2 + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_2 * (da_hL_1 + offset_x)]
    Ba_hL = np.array([*mt4, *mt5, *mt6]).reshape(3,8).T
   
    return pos_aero_hum_L, vel_aero_hum_L, Ba_hL

@jit(nopython=True, fastmath=True)
def func_wing_rL(in1 = None, in2 = None, in3 = None):

    L3a = in3[4]
    L3c = in3[6]
    P5_1 = in3[19]
    P5_2 = in3[20]
    Rb_1_1 = in1[13]
    Rb_1_2 = in1[16]
    Rb_1_3 = in1[19]
    Rb_2_1 = in1[14]
    Rb_2_2 = in1[17]
    Rb_2_3 = in1[20]
    Rb_3_1 = in1[15]
    Rb_3_2 = in1[18]
    Rb_3_3 = in1[21]
    alpha_3 = in3[7]
    da_rL_1 = in2[0]
    da_rL_2 = in2[1]
    offset_x = in3[23]
    theta_WL_1 = in1[0]
    theta_WL_2 = in1[1]
    thetaD_WL_1 = in1[5]
    thetaD_WL_2 = in1[6]
    w_1 = in1[10]
    w_2 = in1[11]
    w_3 = in1[12]
    xD_1 = in1[7]
    xD_2 = in1[8]
    xD_3 = in1[9]
    x_1 = in1[2]
    x_2 = in1[3]
    x_3 = in1[4]
    
    cos_theta_WL_1 = np.cos(theta_WL_1)
    sin_theta_WL_1 = np.sin(theta_WL_1)
    cos_alpha_3 = np.cos(alpha_3)
    sin_alpha_3 = np.sin(alpha_3)

    pos_aero_rad_L = np.array([[x_1 + Rb_1_2 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_3 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_1 * (da_rL_1 + offset_x)],[x_2 + Rb_2_2 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_3 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_1 * (da_rL_1 + offset_x)],[x_3 + Rb_3_2 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_3 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_1 * (da_rL_1 + offset_x)]])

    # if nargout > 1
    mt1 = [xD_1 - (Rb_1_1 * w_3 - Rb_1_3 * w_1) * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + (Rb_1_1 * w_2 - Rb_1_2 * w_1) * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + (da_rL_1 + offset_x) * (Rb_1_2 * w_3 - Rb_1_3 * w_2) + Rb_1_3 * (da_rL_2 * thetaD_WL_2 * np.cos(theta_WL_2) + thetaD_WL_1 * cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) - Rb_1_2 * (da_rL_2 * thetaD_WL_2 * np.sin(theta_WL_2) + thetaD_WL_1 * sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1)]
    mt2 = [xD_2 - (Rb_2_1 * w_3 - Rb_2_3 * w_1) * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + (Rb_2_1 * w_2 - Rb_2_2 * w_1) * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + (da_rL_1 + offset_x) * (Rb_2_2 * w_3 - Rb_2_3 * w_2) + Rb_2_3 * (da_rL_2 * thetaD_WL_2 * np.cos(theta_WL_2) + thetaD_WL_1 * cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) - Rb_2_2 * (da_rL_2 * thetaD_WL_2 * np.sin(theta_WL_2) + thetaD_WL_1 * sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1)]
    mt3 = [xD_3 - (Rb_3_1 * w_3 - Rb_3_3 * w_1) * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + (Rb_3_1 * w_2 - Rb_3_2 * w_1) * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + (da_rL_1 + offset_x) * (Rb_3_2 * w_3 - Rb_3_3 * w_2) + Rb_3_3 * (da_rL_2 * thetaD_WL_2 * np.cos(theta_WL_2) + thetaD_WL_1 * cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) - Rb_3_2 * (da_rL_2 * thetaD_WL_2 * np.sin(theta_WL_2) + thetaD_WL_1 * sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1)]
    vel_aero_rad_L = np.array([mt1, mt2, mt3]).reshape(3,1)

    # if nargout > 2
    mt4 = [- Rb_1_2 * (sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_3 * (cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),Rb_1_3 * da_rL_2 * np.cos(theta_WL_2) - Rb_1_2 * da_rL_2 * np.sin(theta_WL_2),1.0,0.0,0.0,Rb_1_3 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_1_2 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1),Rb_1_1 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_1_3 * (da_rL_1 + offset_x),- Rb_1_1 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_2 * (da_rL_1 + offset_x)]
    mt5 = [- Rb_2_2 * (sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_3 * (cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),Rb_2_3 * da_rL_2 * np.cos(theta_WL_2) - Rb_2_2 * da_rL_2 * np.sin(theta_WL_2),0.0,1.0,0.0,Rb_2_3 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_2_2 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1),Rb_2_1 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_2_3 * (da_rL_1 + offset_x),- Rb_2_1 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_2 * (da_rL_1 + offset_x)]
    mt6 = [- Rb_3_2 * (sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_3 * (cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),Rb_3_3 * da_rL_2 * np.cos(theta_WL_2) - Rb_3_2 * da_rL_2 * np.sin(theta_WL_2),0.0,0.0,1.0,Rb_3_3 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_3_2 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1),Rb_3_1 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_3_3 * (da_rL_1 + offset_x),- Rb_3_1 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + da_rL_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_2 * (da_rL_1 + offset_x)]
    Ba_rL = np.array([*mt4, *mt5, *mt6]).reshape(3,8).T

    return pos_aero_rad_L, vel_aero_rad_L, Ba_rL

@jit(nopython=True, fastmath=True)
def func_wing_hR(in1 = None, in2 = None, in3 = None): 

    L3c = in3[6]
    P5_1 = in3[19]
    P5_2 = in3[20]
    Rb_1_1 = in1[13]
    Rb_1_2 = in1[16]
    Rb_1_3 = in1[19]
    Rb_2_1 = in1[14]
    Rb_2_2 = in1[17]
    Rb_2_3 = in1[20]
    Rb_3_1 = in1[15]
    Rb_3_2 = in1[18]
    Rb_3_3 = in1[21]
    alpha_3 = in3[7]
    da_hR_1 = in2[0]
    da_hR_2 = in2[1]
    offset_x = in3[23]
    theta_WL_1 = in1[0]
    thetaD_WL_1 = in1[5]
    w_1 = in1[10]
    w_2 = in1[11]
    w_3 = in1[12]
    xD_1 = in1[7]
    xD_2 = in1[8]
    xD_3 = in1[9]
    x_1 = in1[2]
    x_2 = in1[3]
    x_3 = in1[4]

    cos_theta_WL_1 = np.cos(theta_WL_1)
    sin_theta_WL_1 = np.sin(theta_WL_1)
    cos_alpha_3 = np.cos(alpha_3)
    sin_alpha_3 = np.sin(alpha_3)

    pos_aero_hum_R = np.array([[x_1 + Rb_1_3 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_2 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_1 * (da_hR_1 + offset_x)],[x_2 + Rb_2_3 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_2 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_1 * (da_hR_1 + offset_x)],[x_3 + Rb_3_3 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_2 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_1 * (da_hR_1 + offset_x)]])

    # if nargout > 1
    mt1 = [xD_1 + (da_hR_1 + offset_x) * (Rb_1_2 * w_3 - Rb_1_3 * w_2) + (Rb_1_1 * w_2 - Rb_1_2 * w_1) * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_1_2 * (thetaD_WL_1 * sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1) - Rb_1_3 * (thetaD_WL_1 * cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) - (Rb_1_1 * w_3 - Rb_1_3 * w_1) * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1)]
    mt2 = [xD_2 + (da_hR_1 + offset_x) * (Rb_2_2 * w_3 - Rb_2_3 * w_2) + (Rb_2_1 * w_2 - Rb_2_2 * w_1) * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_2_2 * (thetaD_WL_1 * sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1) - Rb_2_3 * (thetaD_WL_1 * cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) - (Rb_2_1 * w_3 - Rb_2_3 * w_1) * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1)]
    mt3 = [xD_3 + (da_hR_1 + offset_x) * (Rb_3_2 * w_3 - Rb_3_3 * w_2) + (Rb_3_1 * w_2 - Rb_3_2 * w_1) * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_3_2 * (thetaD_WL_1 * sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) - L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1) - Rb_3_3 * (thetaD_WL_1 * cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) - (Rb_3_1 * w_3 - Rb_3_3 * w_1) * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1)]
    vel_aero_hum_R = np.array([mt1,mt2,mt3]).reshape(3,1)

    # if nargout > 2
    mt4 = [- Rb_1_2 * (sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) - L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_1_3 * (cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1),0.0,1.0,0.0,0.0,- Rb_1_2 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_3 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1),Rb_1_1 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_1_3 * (da_hR_1 + offset_x),- Rb_1_1 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_2 * (da_hR_1 + offset_x)]
    mt5 = [- Rb_2_2 * (sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) - L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_2_3 * (cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1),0.0,0.0,1.0,0.0,- Rb_2_2 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_3 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1),Rb_2_1 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_2_3 * (da_hR_1 + offset_x),- Rb_2_1 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_2 * (da_hR_1 + offset_x)]
    mt6 = [- Rb_3_2 * (sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) - L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_3_3 * (cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1),0.0,0.0,0.0,1.0,- Rb_3_2 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_3 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1),Rb_3_1 * (P5_2 - sin_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_3_3 * (da_hR_1 + offset_x),- Rb_3_1 * (- P5_1 + cos_theta_WL_1 * (da_hR_2 - L3c * sin_alpha_3) + L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_2 * (da_hR_1 + offset_x)]
    Ba_hR = np.array([*mt4, *mt5, *mt6]).reshape(3,8).T

    return pos_aero_hum_R, vel_aero_hum_R, Ba_hR

@jit(nopython=True, fastmath=True)
def func_wing_rR(in1 = None, in2 = None, in3 = None):

    L3a = in3[4]
    L3c = in3[6]
    P5_1 = in3[19]
    P5_2 = in3[20]
    Rb_1_1 = in1[13]
    Rb_1_2 = in1[16]
    Rb_1_3 = in1[19]
    Rb_2_1 = in1[14]
    Rb_2_2 = in1[17]
    Rb_2_3 = in1[20]
    Rb_3_1 = in1[15]
    Rb_3_2 = in1[18]
    Rb_3_3 = in1[21]
    alpha_3 = in3[7]
    da_rR_1 = in2[0]
    da_rR_2 = in2[1]
    offset_x = in3[23]
    theta_WL_1 = in1[0]
    theta_WL_2 = in1[1]
    thetaD_WL_1 = in1[5]
    thetaD_WL_2 = in1[6]
    w_1 = in1[10]
    w_2 = in1[11]
    w_3 = in1[12]
    xD_1 = in1[7]
    xD_2 = in1[8]
    xD_3 = in1[9]
    x_1 = in1[2]
    x_2 = in1[3]
    x_3 = in1[4]

    cos_theta_WL_1 = np.cos(theta_WL_1)
    sin_theta_WL_1 = np.sin(theta_WL_1)
    cos_alpha_3 = np.cos(alpha_3)
    sin_alpha_3 = np.sin(alpha_3)
    pos_aero_rad_R = np.array([[x_1 - Rb_1_2 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_3 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_1 * (da_rR_1 + offset_x)],[x_2 - Rb_2_2 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_3 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_1 * (da_rR_1 + offset_x)],[x_3 - Rb_3_2 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_3 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_1 * (da_rR_1 + offset_x)]])

    # if nargout > 1
    mt1 = [xD_1 + (Rb_1_1 * w_3 - Rb_1_3 * w_1) * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + (Rb_1_1 * w_2 - Rb_1_2 * w_1) * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + (da_rR_1 + offset_x) * (Rb_1_2 * w_3 - Rb_1_3 * w_2) - Rb_1_3 * (da_rR_2 * thetaD_WL_2 * np.cos(theta_WL_2) - thetaD_WL_1 * cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) + Rb_1_2 * (- da_rR_2 * thetaD_WL_2 * np.sin(theta_WL_2) + thetaD_WL_1 * sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1)]
    mt2 = [xD_2 + (Rb_2_1 * w_3 - Rb_2_3 * w_1) * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + (Rb_2_1 * w_2 - Rb_2_2 * w_1) * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + (da_rR_1 + offset_x) * (Rb_2_2 * w_3 - Rb_2_3 * w_2) - Rb_2_3 * (da_rR_2 * thetaD_WL_2 * np.cos(theta_WL_2) - thetaD_WL_1 * cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) + Rb_2_2 * (- da_rR_2 * thetaD_WL_2 * np.sin(theta_WL_2) + thetaD_WL_1 * sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1)]
    mt3 = [xD_3 + (Rb_3_1 * w_3 - Rb_3_3 * w_1) * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + (Rb_3_1 * w_2 - Rb_3_2 * w_1) * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) + (da_rR_1 + offset_x) * (Rb_3_2 * w_3 - Rb_3_3 * w_2) - Rb_3_3 * (da_rR_2 * thetaD_WL_2 * np.cos(theta_WL_2) - thetaD_WL_1 * cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * sin_theta_WL_1) + Rb_3_2 * (- da_rR_2 * thetaD_WL_2 * np.sin(theta_WL_2) + thetaD_WL_1 * sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * thetaD_WL_1 * cos_alpha_3 * cos_theta_WL_1)]
    vel_aero_rad_R = np.array([mt1,mt2,mt3]).reshape(3,1)

    # if nargout > 2
    mt4 = [Rb_1_2 * (sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_1_3 * (cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),- Rb_1_3 * da_rR_2 * np.cos(theta_WL_2) - Rb_1_2 * da_rR_2 * np.sin(theta_WL_2),1.0,0.0,0.0,- Rb_1_3 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_1_2 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1),Rb_1_1 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_1_3 * (da_rR_1 + offset_x),Rb_1_1 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_1_2 * (da_rR_1 + offset_x)]
    mt5 = [Rb_2_2 * (sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_2_3 * (cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),- Rb_2_3 * da_rR_2 * np.cos(theta_WL_2) - Rb_2_2 * da_rR_2 * np.sin(theta_WL_2),0.0,1.0,0.0,- Rb_2_3 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_2_2 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1),Rb_2_1 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_2_3 * (da_rR_1 + offset_x),Rb_2_1 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_2_2 * (da_rR_1 + offset_x)]
    mt6 = [Rb_3_2 * (sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) + L3c * cos_alpha_3 * cos_theta_WL_1) + Rb_3_3 * (cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - L3c * cos_alpha_3 * sin_theta_WL_1),- Rb_3_3 * da_rR_2 * np.cos(theta_WL_2) - Rb_3_2 * da_rR_2 * np.sin(theta_WL_2),0.0,0.0,1.0,- Rb_3_3 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) - Rb_3_2 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1),Rb_3_1 * (P5_2 + sin_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.sin(theta_WL_2) + L3c * cos_alpha_3 * cos_theta_WL_1) - Rb_3_3 * (da_rR_1 + offset_x),Rb_3_1 * (P5_1 + cos_theta_WL_1 * (L3a + L3c * sin_alpha_3) - da_rR_2 * np.cos(theta_WL_2) - L3c * cos_alpha_3 * sin_theta_WL_1) + Rb_3_2 * (da_rR_1 + offset_x)]
    Ba_rR = np.array([*mt4, *mt5, *mt6]).reshape(3,8).T
    
    return pos_aero_rad_R, vel_aero_rad_R, Ba_rR

@jit(nopython=True, fastmath=True)
def func_tail(in1 = None, in2 = None): 

    Rb_1_1 = in1[13]
    Rb_1_2 = in1[16]
    Rb_1_3 = in1[19]
    Rb_2_1 = in1[14]
    Rb_2_2 = in1[17]
    Rb_2_3 = in1[20]
    Rb_3_1 = in1[15]
    Rb_3_2 = in1[18]
    Rb_3_3 = in1[21]
    dtail_1 = in2[0]
    dtail_2 = in2[1]
    dtail_3 = in2[2]
    w_1 = in1[10]
    w_2 = in1[11]
    w_3 = in1[12]
    xD_1 = in1[7]
    xD_2 = in1[8]
    xD_3 = in1[9]
    x_1 = in1[2]
    x_2 = in1[3]
    x_3 = in1[4]

    pos_aero_tail = np.array([[x_1 + Rb_1_1 * dtail_1 + Rb_1_2 * dtail_2 + Rb_1_3 * dtail_3],[x_2 + Rb_2_1 * dtail_1 + Rb_2_2 * dtail_2 + Rb_2_3 * dtail_3],[x_3 + Rb_3_1 * dtail_1 + Rb_3_2 * dtail_2 + Rb_3_3 * dtail_3]])
    
    #if nargout > 1
    vel_aero_tail = np.array([[xD_1 + dtail_3 * (Rb_1_1 * w_2 - Rb_1_2 * w_1) - dtail_2 * (Rb_1_1 * w_3 - Rb_1_3 * w_1) + dtail_1 * (Rb_1_2 * w_3 - Rb_1_3 * w_2)],[xD_2 + dtail_3 * (Rb_2_1 * w_2 - Rb_2_2 * w_1) - dtail_2 * (Rb_2_1 * w_3 - Rb_2_3 * w_1) + dtail_1 * (Rb_2_2 * w_3 - Rb_2_3 * w_2)],[xD_3 + dtail_3 * (Rb_3_1 * w_2 - Rb_3_2 * w_1) - dtail_2 * (Rb_3_1 * w_3 - Rb_3_3 * w_1) + dtail_1 * (Rb_3_2 * w_3 - Rb_3_3 * w_2)]])
    
    #if nargout > 2
    Ba_tail = np.array([0.0,0.0,1.0,0.0,0.0,- Rb_1_2 * dtail_3 + Rb_1_3 * dtail_2,Rb_1_1 * dtail_3 - Rb_1_3 * dtail_1,- Rb_1_1 * dtail_2 + Rb_1_2 * dtail_1,0.0,0.0,0.0,1.0,0.0,- Rb_2_2 * dtail_3 + Rb_2_3 * dtail_2,Rb_2_1 * dtail_3 - Rb_2_3 * dtail_1,- Rb_2_1 * dtail_2 + Rb_2_2 * dtail_1,0.0,0.0,0.0,0.0,1.0,- Rb_3_2 * dtail_3 + Rb_3_3 * dtail_2,Rb_3_1 * dtail_3 - Rb_3_3 * dtail_1,- Rb_3_1 * dtail_2 + Rb_3_2 * dtail_1]).reshape(3,8).T

    return pos_aero_tail, vel_aero_tail, Ba_tail