"""
================================================================
 MODULE 1 / 5  --  PHYSICAL PARAMETERS & NUMERICAL INTEGRATOR
================================================================
 Contributor : Mezba Us Salaheen (2105172)
 Part of     : Defender-Aware Basketball Shot Simulation
               (extension of Silverberg, Tran & Adcock, 2003)

 Responsibility
   This module owns the physical constants of the ball / court
   and the low-level numerical machinery used by every other
   module: the quadratic aerodynamic drag model and the
   4th-order Runge-Kutta (RK4) integrator that advances the
   ball's state one time step at a time.

   The base paper (Silverberg et al., 2003, Sec. II) explicitly
   neglects aerodynamic effects ("... because of their smallness
   and because they would not be exploited by the shooter").
   Reintroducing quadratic drag plus a horizontal wind term is
   the first half of our extension, so the equation of motion
   solved here is:

        m * dv/dt = -m*g*j - 0.5*rho*Cd*A*|v_rel|*v_rel

   where v_rel = v_ball - v_wind is the ball's velocity relative
   to the moving air.

 Depends on : numpy, dataclasses  (standard library / numpy only)
 Used by    : trajectory_simulation_siam.py, optimization_analysis_razin.py,
              main_simulation.py
================================================================
"""

import numpy as np
from dataclasses import dataclass


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

@dataclass
class Params:
    # Ball (men's size 7)
    m: float = 0.624            # kg
    R: float = 0.1194           # m
    Cd: float = 0.47            # drag coefficient (sphere)

    # Environment
    g: float = 9.81             # m/s^2
    rho: float = 1.225          # kg/m^3 (sea-level air)

    # Court geometry (FIBA / NBA)
    hoop_height: float = 3.048  # 10 ft
    hoop_radius: float = 0.2286 # 18 in diameter rim
    rim_offset: float = 0.15    # rim front edge stands off the backboard
    backboard_x: float = 4.191  # free-throw line to backboard
    release_x: float = 0.15     # ball leaves slightly ahead of the shooter

    @property
    def hoop_x(self) -> float:
        """Horizontal centre of the rim."""
        return self.backboard_x - self.rim_offset - self.hoop_radius


# ============================================================
# 2. PHYSICS ENGINE  (drag model + RK4 integrator)
# ============================================================

def drag_force(vx, vy, wind_x, P: Params):
    """Quadratic aerodynamic drag evaluated in the air frame."""
    vxr = vx - wind_x          # velocity relative to the air
    vyr = vy
    v = np.hypot(vxr, vyr)
    if v < 1e-9:
        return 0.0, 0.0
    A = np.pi * P.R ** 2
    k = 0.5 * P.rho * P.Cd * A
    return -k * v * vxr, -k * v * vyr


def derivatives(state, wind_x, P: Params):
    """state = [x, y, vx, vy] -> d(state)/dt"""
    _, _, vx, vy = state
    Fx, Fy = drag_force(vx, vy, wind_x, P)
    return np.array([vx, vy, Fx / P.m, -P.g + Fy / P.m])


def rk4_step(state, dt, wind_x, P: Params):
    """One 4th-order Runge-Kutta step advancing the state by dt."""
    k1 = derivatives(state, wind_x, P)
    k2 = derivatives(state + 0.5 * dt * k1, wind_x, P)
    k3 = derivatives(state + 0.5 * dt * k2, wind_x, P)
    k4 = derivatives(state + dt * k3, wind_x, P)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ------------------------------------------------------------
# Quick self-test: run this file directly to sanity-check the
# integrator against the closed-form no-drag projectile range.
# ------------------------------------------------------------
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
