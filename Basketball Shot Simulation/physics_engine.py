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
    release_x: float = 0.15

    @property
    def hoop_x(self) -> float:
        return self.backboard_x - self.rim_offset - self.hoop_radius

def drag_force(vx, vy, wind_x, P: Params):
    vxr = vx - wind_x
    vyr = vy
    v = np.hypot(vxr, vyr)
    if v < 1e-9:
        return 0.0, 0.0
    A = np.pi * P.R ** 2
    k = 0.5 * P.rho * P.Cd * A
    return -k * v * vxr, -k * v * vyr

def derivatives(state, wind_x, P: Params):
    _, _, vx, vy = state
    Fx, Fy = drag_force(vx, vy, wind_x, P)
    return np.array([vx, vy, Fx / P.m, -P.g + Fy / P.m])

def rk4_step(state, dt, wind_x, P: Params):
    k1 = derivatives(state, wind_x, P)
    k2 = derivatives(state + 0.5 * dt * k1, wind_x, P)
    k3 = derivatives(state + 0.5 * dt * k2, wind_x, P)
    k4 = derivatives(state + dt * k3, wind_x, P)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == "__main__":
    P = Params()
    state = np.array([0.0, 2.0, 6.0, 6.0])
    dt = 0.01
    zero_drag = Params(Cd=0.0)
    for _ in range(400):
        state = rk4_step(state, dt, 0.0, zero_drag)
        if state[1] <= 0:
            break
    print("RK4 self-test (drag disabled) final state:", state)
    print("Params:", P)
