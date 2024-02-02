from logging import warning
import numpy as np
import warnings
import matplotlib.pyplot as plt


class Trajectory(object):
    def __init__(self):
        self._tf = 10
        pass

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        raise NotImplementedError

    def plot(self):
        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        v = np.empty((0, 3))
        a = np.empty((0, 3))
        for t in T:
            x_, v_, a_ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            v = np.append(v, np.array([v_]), axis=0)
            a = np.append(a, np.array([a_]), axis=0)

        plt.figure()
        plt.subplot(311)
        plt.style.use('seaborn-whitegrid')
        plt.plot(T, x[:, 0], 'b', linewidth=2)
        plt.plot(T, x[:, 1], 'g', linewidth=2)
        plt.plot(T, x[:, 2], 'r', linewidth=2)
        plt.scatter(t, x[-1, 0], s=100, c='b', alpha=0.5)
        plt.scatter(t, x[-1, 1], s=100, c='g', alpha=0.5)
        plt.scatter(t, x[-1, 2], s=100, c='r', alpha=0.5)
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle=':')

        plt.subplot(312)
        plt.style.use('seaborn-whitegrid')
        plt.plot(T, v[:, 0], ':b', linewidth=2)
        plt.plot(T, v[:, 1], ':g', linewidth=2)
        plt.plot(T, v[:, 2], ':r', linewidth=2)
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle=':')

        plt.subplot(313)
        plt.plot(T, a[:, 0], '--b', linewidth=2)
        plt.plot(T, a[:, 1], '--g', linewidth=2)
        plt.plot(T, a[:, 2], '--r', linewidth=2)
        plt.grid(visible=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle=':')
        plt.show()
        return


class Setpoint(Trajectory):
    def __init__(self, setpoint):
        self._xf = setpoint
        super().__init__()
    
    def get(self, t):
        return self._xf, np.zeros(3), np.zeros(3)

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


class CircularTraj(Trajectory):
    def __init__(self, r=1, origin=np.zeros(3), w=0.5 * np.pi):
        self.r = r
        self.origin = origin
        self.w = w
        super().__init__()

    def get(self, t):
        x = self.origin + self.r * np.array(
            [np.cos(self.w * t), np.sin(self.w * t), 1])
        v = self.r * np.array(
            [-1 * self.w * np.sin(self.w * t), self.w * np.cos(self.w * t), 0])
        a = self.r * np.array([
            -1 * self.w**2 * np.cos(self.w * t),
            -1 * self.w**2 * np.sin(self.w * t), 0
        ])
        # traj['d3x'] = r*np.array([w**3*math.sin(w*t), -1*w**3*math.cos(w*t), 0])
        # traj['d4x'] =  r*np.array([w**4*math.cos(w*t), w**4*math.sin(w*t), 0])
        # traj['d5x'] =  r*np.array([-w**5*math.sin(w*t), w**5*math.cos(w*t), 0])
        # traj['d6x'] =  r*np.array([-w**6*math.cos(w*t), -w**6*math.sin(w*t), 0])
        return x, v, a

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
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            x = self._pos_offset + self._pos_amp * np.sin(t*np.pi/self._tf - np.pi/2)
            v = self._vel_amp * np.cos(t*np.pi/self._tf - np.pi/2)
            a = self._acc_amp * np.sin(t*np.pi/self._tf - np.pi/2)
            return x, v, a

class CrazyTrajectory(Trajectory):
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
    traj = SmoothTraj5(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 10)
    traj.plot()
    traj = SmoothTraj3(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 10)
    traj.plot()
    traj = SmoothTraj1(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 10)
    traj.plot()
    traj = CircularTraj()
    traj.plot()
    traj = SmoothSineTraj(-1 + 2 * np.random.rand(3), 10 * np.random.rand(3), 5)
    traj.plot()
