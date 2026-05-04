import numpy as np


class FaultModel:

    def __init__(self):
        self.lambdas = np.ones(4)
        self.faulted_motor = None
        self.fault_magnitude = 1.0
        self.inject_time = 0.0
        self.active = False

    def reset(self, fault_range=(1.0, 1.0), inject_mid_episode=False,
              episode_duration=10.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        self.lambdas = np.ones(4)
        self.active = False

        min_lam, max_lam = fault_range

        if min_lam >= 1.0 and max_lam >= 1.0:
            self.faulted_motor = None
            self.fault_magnitude = 1.0
            self.inject_time = 0.0
            self.active = True
            return

        self.faulted_motor = rng.integers(0, 4)
        self.fault_magnitude = rng.uniform(min_lam, max_lam)

        if inject_mid_episode:
            self.inject_time = rng.uniform(0.2, 0.5) * episode_duration
            self.active = False
        else:
            self.inject_time = 0.0
            self.active = True
            self.lambdas[self.faulted_motor] = self.fault_magnitude

    def update(self, current_time):
        if not self.active and self.faulted_motor is not None:
            if current_time >= self.inject_time:
                self.active = True
                self.lambdas[self.faulted_motor] = self.fault_magnitude

    def apply(self, rpm_commands):
        return self.lambdas * rpm_commands

    def get_info(self):
        return {
            'faulted_motor': self.faulted_motor,
            'fault_magnitude': self.fault_magnitude,
            'fault_active': self.active,
            'lambdas': self.lambdas.copy(),
        }