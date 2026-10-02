import numpy as np
from dataclasses import dataclass


@dataclass
class Params:
    m: float = 0.624
    R: float = 0.1194
    Cd: float = 0.47

    g: float = 9.81
    rho: float = 1.225

    hoop_height: float = 3.048
    hoop_radius: float = 0.2286
    rim_offset: float = 0.15

    backboard_x: float = 4.191
    backboard_y: float = 0.0
    backboard_width: float = 1.8288
    backboard_rise: float = 0.90
    backboard_drop: float = 0.15

    release_x: float = 0.15
    release_y: float = 0.0

    defender_radius: float = 0.28
    lane_half_width: float = 2.44

    @property
    def hoop_x(self) -> float:
        return self.backboard_x - self.rim_offset - self.hoop_radius

    @property
    def hoop_y(self) -> float:
        return self.backboard_y

    @property
    def board_top(self) -> float:
        return self.hoop_height + self.backboard_rise

    @property
    def board_bottom(self) -> float:
        return self.hoop_height - self.backboard_drop

    @property
    def board_half_width(self) -> float:
        return 0.5 * self.backboard_width


def drag_force(vel, wind, P: Params):
    vr = vel - wind
    speed = float(np.sqrt(vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]))
    if speed < 1e-9:
        return np.zeros(3)
    A = np.pi * P.R ** 2
    k = 0.5 * P.rho * P.Cd * A
    return -k * speed * vr


def derivatives(state, wind, P: Params):
    vel = state[3:]
    acc = drag_force(vel, wind, P) / P.m
    acc[2] -= P.g
    return np.concatenate((vel, acc))


def rk4_step(state, dt, wind, P: Params):
    k1 = derivatives(state, wind, P)
    k2 = derivatives(state + 0.5 * dt * k1, wind, P)
    k3 = derivatives(state + 0.5 * dt * k2, wind, P)
    k4 = derivatives(state + dt * k3, wind, P)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    P = Params()
    no_drag = Params(Cd=0.0)
    wind = np.zeros(3)
    vx0, vy0, vz0, z0 = 6.0, 1.5, 6.0, 2.0

    state = np.array([0.0, 0.0, z0, vx0, vy0, vz0])
    prev = state
    dt = 0.01
    for _ in range(400):
        prev = state
        state = rk4_step(state, dt, wind, no_drag)
        if state[2] <= 0.0:
            break

    frac = prev[2] / (prev[2] - state[2])
    x_land = prev[0] + frac * (state[0] - prev[0])
    y_land = prev[1] + frac * (state[1] - prev[1])
    t_exact = (vz0 + np.sqrt(vz0 ** 2 + 2.0 * no_drag.g * z0)) / no_drag.g

    print()
    print("Checking the RK4 integrator with drag switched off, where the")
    print("answer is known exactly.")
    print()
    print("A ball thrown from %.1f m up at (%.1f, %.1f, %.1f) m/s lands at"
          % (z0, vx0, vy0, vz0))
    print("x = %.6f m, y = %.6f m" % (x_land, y_land))
    print("The closed-form answer is x = %.6f m, y = %.6f m"
          % (vx0 * t_exact, vy0 * t_exact))
    print()
    print("That leaves %.3f mm of error in x and %.3f mm in y, which is the"
          % (abs(x_land - vx0 * t_exact) * 1000.0,
             abs(y_land - vy0 * t_exact) * 1000.0))
    print("integrator doing its job.")
    print()
