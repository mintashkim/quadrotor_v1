#!/usr/bin/env pythono

import scipy as sp
import math
from logging import warning
import numpy as np
import warnings
import matplotlib.pyplot as plt


PI = sp.pi


class Trajectory(object):
    def __init__(self):
        self._tf = 10
        pass

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        raise NotImplementedError

    def plot(self, name=''):
        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        v = np.empty((0, 3))
        a = np.empty((0, 3))
        for t in T:
            x_, v_, a_ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            v = np.append(v, np.array([v_]), axis=0)
            a = np.append(a, np.array([a_]), axis=0)

        plt.figure(name)
        plt.subplot(311)
        plt.style.use('seaborn-whitegrid')
        plt.plot(T, x[:, 0], 'b', linewidth=2, label='x')
        plt.plot(T, x[:, 1], 'g', linewidth=2, label='y')
        plt.plot(T, x[:, 2], 'r', linewidth=2, label='z')
        plt.scatter(t, x[-1, 0], s=100, c='b', alpha=0.5)
        plt.scatter(t, x[-1, 1], s=100, c='g', alpha=0.5)
        plt.scatter(t, x[-1, 2], s=100, c='r', alpha=0.5)
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle=':')
        plt.title('positions')
        plt.legend()

        plt.subplot(312)
        plt.style.use('seaborn-whitegrid')
        plt.plot(T, v[:, 0], ':b', linewidth=2, label='x')
        plt.plot(T, v[:, 1], ':g', linewidth=2, label='y')
        plt.plot(T, v[:, 2], ':r', linewidth=2, label='z')
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle=':')
        plt.title('velocities')
        plt.legend()

        plt.subplot(313)
        plt.plot(T, a[:, 0], '--b', linewidth=2, label='x')
        plt.plot(T, a[:, 1], '--g', linewidth=2, label='y')
        plt.plot(T, a[:, 2], '--r', linewidth=2, label='z')
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle=':')
        plt.title('accel')
        plt.legend()

        plt.show()
        return


class SmoothTraj(Trajectory):
    def __init__(self, x0, xf, tf):
        self._x0 = x0
        self._xf = xf
        self._tf = tf
        self._pos_params = []
        self._vel_params = []
        self._acc_params = []

        self._t = lambda l: np.array([1., l, l**2, l**3, l**4, l**5])

        self.compute_traj_params()
        pass

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        if t >= self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        elif t <= 0:
            warnings.warn("cannot have t < 0")
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            l = t / self._tf
            return (np.array([self._t(l)])@self._pos_params)[0],\
                   (np.array([self._t(l)])@self._vel_params)[0],\
                   (np.array([self._t(l)])@self._acc_params)[0]


class SmoothTraj5(SmoothTraj):
    """
    given initial and final position
    velocity & accelerations are zero
    """

    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda l: np.array([1., l, l**2, l**3, l**4, l**5])

    def compute_traj_params(self):
        a = self._xf - self._x0
        #                  np.array([  c,           t,          t^2,        t^3,  t^4, t^5])
        self._pos_params = np.array(
            [self._x0,
             np.zeros(3, ),
             np.zeros(3, ), 10 * a, -15 * a, 6 * a])
        self._vel_params = np.array([
            np.zeros(3, ), 2 * np.zeros(3, ), 3 * 10 * a, 4 * -15 * a,
            5 * 6 * a,
            np.zeros(3)
        ])
        self._acc_params = np.array([
            np.zeros(3, ), 6 * 10 * a, 12 * -15 * a, 20 * 6 * a,
            np.zeros(3),
            np.zeros(3)
        ])
        pass


class SmoothTraj3(SmoothTraj):
    """
    given initial and final position
    velocity & accelerations are zero
    """

    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda l: np.array([1., l, l**2, l**3])

    def compute_traj_params(self):
        a = self._xf - self._x0
        #                  np.array([  c,           t,      t^2,  t^3])
        self._pos_params = np.array([self._x0, np.zeros(3, ), 3 * a, -2 * a])
        self._vel_params = np.array(
            [np.zeros(3, ), 6 * a, -6 * a,
             np.zeros(3)])
        self._acc_params = np.array([6 * a, -12 * a, np.zeros(3), np.zeros(3)])
        pass


class SmoothTraj1(SmoothTraj):
    """
    given initial and final position
    velocity & accelerations are zero
    """

    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda l: np.array([1., l])

    def compute_traj_params(self):
        a = self._xf - self._x0
        #                  np.array([  c,           t,      t^2,  t^3])
        self._pos_params = np.array([self._x0, a])
        self._vel_params = np.array([a, np.zeros(3)])
        self._acc_params = np.array([np.zeros(3), np.zeros(3)])
        pass


# class CircularTraj(Trajectory):
#     def __init__(self, r=1, origin=np.zeros(3), w=0.5 * np.pi):
#         self.r = r
#         self.origin = origin
#         self.w = w
#         super().__init__()

#     def get(self, t):
#         x = self.origin + self.r * np.array(
#             [np.cos(self.w * t), np.sin(self.w * t), 1])
#         v = self.r * np.array(
#             [-1 * self.w * np.sin(self.w * t), self.w * np.cos(self.w * t), 0])
#         a = self.r * np.array([
#             -1 * self.w**2 * np.cos(self.w * t),
#             -1 * self.w**2 * np.sin(self.w * t), 0
#         ])
#         # traj['d3x'] = r*np.array([w**3*math.sin(w*t), -1*w**3*math.cos(w*t), 0])
#         # traj['d4x'] =  r*np.array([w**4*math.cos(w*t), w**4*math.sin(w*t), 0])
#         # traj['d5x'] =  r*np.array([-w**5*math.sin(w*t), w**5*math.cos(w*t), 0])
#         # traj['d6x'] =  r*np.array([-w**6*math.cos(w*t), -w**6*math.sin(w*t), 0])
#         return x, v, a


class SmoothSineTraj(SmoothTraj):
    def __init__(self, x0, xf, tf):
        self._pos_offset = np.zeros(3)
        self._pos_amp = np.zeros(3)
        self._vel_amp = np.zeros(3)
        self._acc_amp = np.zeros(3)
        super().__init__(x0, xf, tf)

    def compute_traj_params(self):
        self._pos_offset = 0.5*(self._xf + self._x0)
        self._pos_amp = 0.5*(self._xf - self._x0)
        self._vel_amp = 0.5*(self._xf - self._x0)*(np.pi/self._tf)
        self._acc_amp = -0.5*(self._xf - self._x0)*(np.pi/self._tf)**2

    def get(self, t):
        if t >= self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        elif t <= 0:
            warnings.warn("cannot have t < 0")
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            x = self._pos_offset + self._pos_amp * \
                np.sin(t*np.pi/self._tf - np.pi/2)
            v = self._vel_amp * np.cos(t*np.pi/self._tf - np.pi/2)
            a = self._acc_amp * np.sin(t*np.pi/self._tf - np.pi/2)
            return x, v, a


class PolyTraj5(SmoothTraj):
    """
    given initial and final position
    velocity & accelerations are zero
    """

    def __init__(self, x0, xf, tf, v0=np.zeros(3), vf=np.zeros(3), a0=np.zeros(3), af=np.zeros(3)):
        self._v0 = v0
        self._vf = vf
        self._a0 = a0
        self._af = af
        super().__init__(x0, xf, tf)
        self._t = lambda l: np.array([1., l, l**2, l**3, l**4, l**5])

    def solve_params(self, p0, v0, a0, p1, v1, a1):
        b = np.array([[p1-p0-v0-0.5*a0*a0],
                      [v1-v0-a0],
                      [a1-a0]])
        A = np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]])
        x = np.linalg.pinv(A)@b
        return x

    def get(self, t):
        if t >= self._tf:
            return self._xf, self._vf, self._af
        elif t <= 0:
            return self._x0, self._v0, self._a0
        else:
            l = t / self._tf
            return (np.array([self._t(l)])@self._pos_params)[0],\
                   (np.array([self._t(l)])@self._vel_params)[0],\
                   (np.array([self._t(l)])@self._acc_params)[0]

    def compute_traj_params(self):
        a = self._xf - self._x0
        #  np.array([ c, t, t^2, t^3, t^4, t^5])
        a3 = np.zeros(3)
        a4 = np.zeros(3)
        a5 = np.zeros(3)
        for i in range(3):
            p = self.solve_params(self._x0[i],
                                  self._v0[i],
                                  self._a0[i],
                                  self._xf[i],
                                  self._vf[i],
                                  self._af[i])
            a3[i], a4[i], a5[i] = p[0][0], p[1][0], p[2][0]

        self._pos_params = np.array(
            [self._x0, self._v0, 0.5*self._a0, a3, a4, a5])
        self._vel_params = np.array([
            self._v0, self._a0, 3. * a3, 4.*a4, 5.*a5, np.zeros(3)
        ])
        self._acc_params = np.array([
            self._a0, 6.*a3, 12.*a4, 20.*a3, np.zeros(3), np.zeros(3)
        ])
        return



class RandomTrajectory(SmoothTraj):
    """
    given initial and final position
    velocity & accelerations are zero
    """

    def __init__(self, x0, tf, v0=np.zeros(3), a0=np.zeros(3), dt=0.005):
        self._v0 = v0
        self._a0 = a0
        super().__init__(x0, np.zeros(3), tf)
        N = int(1./dt)
        self.T = np.linspace(0, tf, N+1)
        self._jerk = -1+2*np.random.rand(3,N+1)
        
        pass
    
    def get(self, t):
        if t <= 0:
            return self._x0, self._v0, self._a0
        else:
            l = t / self._tf
            return (np.array([self._t(l)])@self._pos_params)[0],\
                   (np.array([self._t(l)])@self._vel_params)[0],\
                   (np.array([self._t(l)])@self._acc_params)[0]

    def compute_traj_params(self):
        #  np.array([ c, t, t^2, t^3, t^4, t^5])
        a3 =  2 * np.random.rand(3)
        a4 = 2 * np.random.rand(3)
        a5 = 2 * np.random.rand(3)

        self._pos_params = np.array(
            [self._x0, self._v0, 0.5*self._a0, a3, a4, a5])
        self._vel_params = np.array([
            self._v0, self._a0, 3. * a3, 4.*a4, 5.*a5, np.zeros(3)
        ])
        self._acc_params = np.array([
            self._a0, 6.*a3, 12.*a4, 20.*a3, np.zeros(3), np.zeros(3)
        ])
        return

def setpoint(t, sp=np.array([0., 0., 1.0])):

    traj = dict()
    traj['x'] = sp
    traj['dx'] = np.zeros(3)
    traj['d2x'] = np.zeros(3)
    traj['d3x'] = np.zeros(3)
    traj['d4x'] = np.zeros(3)
    traj['d5x'] = np.zeros(3)
    traj['d6x'] = np.zeros(3)

    return traj


def circleXY(t, r=1, c=np.zeros(3), w=0.1*PI):

    traj = dict()
    traj['x'] = c + r*np.array([math.cos(w*t), math.sin(w*t), 0])
    traj['dx'] = r*np.array([-1*w*math.sin(w*t), w*math.cos(w*t), 0])
    traj['d2x'] = r*np.array([-1*w**2*math.cos(w*t), -1*w**2*math.sin(w*t), 0])
    traj['d3x'] = r*np.array([w**3*math.sin(w*t), -1*w**3*math.cos(w*t), 0])
    traj['d4x'] = r*np.array([w**4*math.cos(w*t), w**4*math.sin(w*t), 0])
    traj['d5x'] = r*np.array([-w**5*math.sin(w*t), w**5*math.cos(w*t), 0])
    traj['d6x'] = r*np.array([-w**6*math.cos(w*t), -w**6*math.sin(w*t), 0])

    return traj


def polynomialTraj5(p0=np.zeros(3), v0=np.zeros(3), a0=np.zeros(3), pf=np.zeros(3),
                    vf=np.zeros(3), af=np.zeros(3)):

    return

class CircularTraj(Trajectory):
    def __init__(self, center=np.zeros(3), radius=1, speed=1, th0 = 0 ,tf=10.):
        self._center = center
        self._radius = radius
        self._speed = speed
        self._tf = tf
        self._w = self._speed/self._radius
        self._th0 = th0
        self.compute_params()
          
    def compute_params(self):
        self._sint = lambda t: np.sin(self._th0 + self._w*t)
        self._cost = lambda t: np.cos(self._th0 + self._w*t)
        self._x = lambda t: self._center +  np.array([self._radius*self._cost(t), 
                                            self._radius*self._sint(t), 0.])
        self._dx = lambda t:  np.array([-self._radius*self._sint(t)*self._w, 
                                            self._radius*self._cost(t)*self._w, 0.])
        self._d2x = lambda t:  np.array([-self._radius*self._cost(t)*self._w**2, 
                                            -self._radius*self._sint(t)*self._w**2, 0.])

    def get(self, t):
        return self._x(t), self._dx(t), self._d2x(t)
          
class CrazyTrajectroy(Trajectory):
    def __init__(self, tf=10, ax=2, ay=2.5, az=1.5, 
                            f1 = 1/4, f2=1/5, f3=1/7,
                            phix=0., phiy = 0., phiz=0):
        super().__init__()
        self.ax = ax
        self.ay = ay
        self.az = az
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.phix = phix
        self.phiy = phiy
        self.phiz = phiz
        
    def get(self, t): 
        w1 = 2*np.pi*self.f1
        w2 = 2*np.pi*self.f2
        w3 = 2*np.pi*self.f3
        x = np.array([self.ax*(1-np.cos(w1*t+self.phix)),
                          self.ay*np.sin(w2*t+self.phiy),
                          self.az*np.cos(w3*t+self.phiz)])
        dx = np.array([self.ax*np.sin(w1*t)*w1, 
                        self.ay*np.cos(w2*t)*w2,
                        -self.az*np.sin(w3*t)*w3])
        d2x = np.array([self.ax*np.cos(w1*t)*w1*w1,
                          -self.ay*np.sin(w2*t)*w2*w2,
                          -self.az*np.cos(w3*t)*w3*w3])
        
        return x, dx, d2x
        

if __name__ == "__main__":
    # traj = SmoothTraj5(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 10)
    # traj.plot('SmoothTraj5')
    # traj = SmoothTraj3(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 10)
    # traj.plot('SmoothTraj3')
    # traj = SmoothTraj1(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 10)
    # traj.plot('SmoothTraj1')
    # traj = CircularTraj()
    # traj.plot('CircularTraj')
    # traj = SmoothSineTraj(-1 + 2 * np.random.rand(3),
    #                       10 * np.random.rand(3), 10)
    # traj.plot('SmoothSineTraj')
    # traj = PolyTraj5(-2 * np.random.rand(3),
    #                       2 * np.random.rand(3), 10,
    #                       v0=np.zeros(3),
    #                       vf=np.zeros(3),
    #                       a0=-1 + 2 * np.random.rand(3),
    #                       af=-1 + 2 * np.random.rand(3))
    # traj.plot('PolyTraj5')
    # traj = RandomTrajectory(-2 + 4 * np.random.rand(3), 5)
    # traj.plot('RandomTraj5')
    # traj = CircularTraj()
    # traj.plot('CircularTraj')
    
    
    plt.figure()
    
    for i in range(3):
        traj = CrazyTrajectroy(ax=2*np.random.rand(),
                                ay=2*np.random.rand(),
                                az=2*np.random.rand(),
                                f1=-0.5+1*np.random.rand(),
                                f2=-0.5+1*np.random.rand(),
                                f3=-0.5+1*np.random.rand(),
                                phix=-np.pi+2*np.pi*np.random.rand(),
                                phiy=-np.pi+2*np.pi*np.random.rand(),
                                phiz=-np.pi+2*np.pi*np.random.rand())
        T = np.linspace(0, traj._tf, 100)

        x = np.empty((0, 3))
        v = np.empty((0, 3))
        a = np.empty((0, 3))
        for t in T:
            x_, v_, a_ = traj.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            v = np.append(v, np.array([v_]), axis=0)
            a = np.append(a, np.array([a_]), axis=0)
        
        plt.subplot(231)
        plt.plot(x[:,0], x[:,1])
        plt.subplot(232)
        plt.plot(x[:,1], x[:,2])
        plt.subplot(233)
        plt.plot(x[:,2], x[:,0])
        plt.subplot(234)
        plt.plot(T, x[:,0])
        plt.subplot(235)
        plt.plot(T, v[:,1])
        plt.subplot(236)
        plt.plot(T, a[:,2])
        
    plt.show()
    pass