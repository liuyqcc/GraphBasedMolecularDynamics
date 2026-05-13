import numpy as np


class PropagationClassical():
    def __init__(self):
        pass

    def update_velocity_half_step(self, v, dt, f, mass):
        """
        v(t+dt/2) = v(t) + a(t)*dt/2
        """
        v_new = v + f / mass * dt
        return v_new

    def update_coordinates(self, x, v, dt):
        """
        x(t+dt) = x(t) + v(t+dt/2)*dt
        """
        x_new = x + v*dt
        return x_new

