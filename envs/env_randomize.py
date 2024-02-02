import numpy as np

class EnvRandomizer(object):
    def __init__(self, sim) -> None:
        self._default_mass = sim.mass
        self._default_inertia = sim.inertia
        self.m = self._default_mass
        self.J = self._default_inertia

    def randomize_dynamics(self):
        self.m = self._default_mass * (1.0 + np.random.uniform(-0.3, 0.3))
        self.J = np.random.uniform(0.7, 1.3, (3, 3)) * self._default_inertia