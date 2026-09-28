"""
================================================================
 MODULE 2 / 5  --  TRAJECTORY SIMULATION & COLLISION LOGIC
================================================================
 Contributor : Siam Ahmed (2105155)
 Part of     : Defender-Aware Basketball Shot Simulation
               (extension of Silverberg, Tran & Adcock, 2003)

 Responsibility
   This module contains the single most important extension of
   the base paper: simulate(), which time-steps one full shot
   (release -> made / missed) using the RK4 integrator from
   physics_engine_mezba.py and resolves every physical event
   along the way:

     - defender contact  : NEW constraint not present in the
                            base paper. The defender is modelled
                            as a vertical reach barrier at x =
                            defender_x, height = defender_reach.
                            A ball crossing that line below the
                            reach height is "blocked" and
                            deflects backwards/downwards.
     - rim contact        : elastic bounce off the two iron
                             edges (restitution e = 0.45), same
                             spirit as Silverberg et al. Eq. (12)-(13)
                             but treated as a simple circular
                             contact instead of the full hoop-
                             coordinate formulation of the paper.
     - backboard contact  : partially-elastic bounce (restitution
                             0.55), analogous to Silverberg et al.
                             Eq. (2)-(3).
     - made / missed test : the ball must cross the rim plane
                             within the usable opening
                             (hoop_radius - ball_radius).

   The function returns a dictionary of the full (x, y, t)
   trajectory plus every diagnostic quantity needed by the UI
   and the optimizer: apex height, flight time, entry angle,
   defender clearance margin, and a human-readable shot status.

 Depends on : numpy, physics_engine_mezba.py (Params, rk4_step)
 Used by    : optimization_analysis_razin.py, main_simulation.py
================================================================
"""

import numpy as np

from physics_engine_mezba import Params, rk4_step


# ============================================================
# 3. TRAJECTORY SIMULATION
# ============================================================

def simulate(v0, theta_deg, wind_x, defender_x, defender_reach,
             release_height, P: Params, dt: float = 0.010,
             collide: bool = True):
    """
    Integrate one shot and classify the outcome.

    Returns a dict with the trajectory plus diagnostic quantities
    (apex, entry angle, flight time, clearance over the defender).
    """
    theta = np.radians(theta_deg)
    state = np.array([P.release_x, release_height,
                      v0 * np.cos(theta), v0 * np.sin(theta)])

    t_max = 5.0

    xs, ys, ts = [state[0]], [state[1]], [0.0]

    clears_defender = None
    defender_clearance = None
    hit_backboard = False
    hit_rim = False
    hit_ground = False
    made = False
    entry_angle = None
    x_cross = None          # where the ball crosses the rim plane on the way down
    t = 0.0

    hoop_x = P.hoop_x
    board_top = P.hoop_height + 0.90
    board_bot = P.hoop_height - 0.15

    while t < t_max:
        prev = state.copy()
        state = rk4_step(state, dt, wind_x, P)
        t += dt

        # ---- defender clearance (first crossing of the defender line)
        # ---- defender contact / clearance
        if clears_defender is None and state[0] >= defender_x > prev[0]:
            frac = (defender_x - prev[0]) / (state[0] - prev[0] + 1e-12)
            y_at = prev[1] + frac * (state[1] - prev[1])
            defender_clearance = y_at - P.R - defender_reach
            clears_defender = defender_clearance > 0.0

            if not clears_defender:
                # ============================================
                # BLOCKED — the ball physically hits the defender
                # and deflects backwards / downwards.
                # ============================================
                # Place the ball exactly at the contact point
                state[0] = defender_x - P.R - 0.005   # touching the defender
                state[1] = y_at

                # Realistic block deflection:
                #   - horizontal velocity reverses (defender's hand pushes back)
                #   - vertical velocity is killed / slightly downward
                #   - restitution ~0.35 (a hand is not a springy wall)
                e_block = 0.35
                state[2] = -e_block * state[2]        # reverse & dampen vx
                state[3] = -0.5 * abs(state[3])       # push down

                # Record this point in the trajectory and stop integrating
                xs.append(state[0]); ys.append(state[1]); ts.append(t)
                break   # ← exit the while loop; the shot is over

        # ---- contact with the iron (front and back rim edges)
        for rx in ([] if not collide else (hoop_x - P.hoop_radius, hoop_x + P.hoop_radius)):
            dx, dy = state[0] - rx, state[1] - P.hoop_height
            d = np.hypot(dx, dy)
            if d < P.R:
                hit_rim = True
                nx, ny = dx / (d + 1e-12), dy / (d + 1e-12)
                vn = state[2] * nx + state[3] * ny
                if vn < 0:                       # moving into the rim
                    e = 0.45                     # restitution of the iron
                    state[2] -= (1 + e) * vn * nx
                    state[3] -= (1 + e) * vn * ny
                state[0] = rx + nx * P.R * 1.001
                state[1] = P.hoop_height + ny * P.R * 1.001

        # ---- rim plane crossing (ball falling through hoop height)
        if not made and prev[1] >= P.hoop_height > state[1]:
            frac = (P.hoop_height - prev[1]) / (state[1] - prev[1] + 1e-12)
            x_at = prev[0] + frac * (state[0] - prev[0])
            if x_cross is None:
                x_cross = x_at
            if abs(x_at - hoop_x) < (P.hoop_radius - P.R):
                made = True
                entry_angle = np.degrees(np.arctan2(-state[3], abs(state[2])))
                state[2] *= 0.12          # the net kills the forward speed

        # ---- backboard collision
        if (collide and not made and board_bot <= state[1] <= board_top
                and state[0] + P.R >= P.backboard_x and state[2] > 0):
            hit_backboard = True
            state[2] = -0.55 * state[2]
            state[0] = P.backboard_x - P.R - 0.005

        xs.append(state[0]); ys.append(state[1]); ts.append(t)

        # ---- the ball has dropped clear of the net
        if made and state[1] < P.hoop_height - 0.75:
            break

        # ---- ground
        if state[1] <= P.R:
            hit_ground = True
            break

        # ---- ball has clearly left the scene
        if state[0] > P.backboard_x + 0.8 and state[3] < 0:
            break

    if clears_defender is None:          # never reached the defender
        clears_defender = False
        defender_clearance = -defender_reach

    xs = np.array(xs); ys = np.array(ys)
    blocked = not clears_defender

    if blocked:
        status = "BLOCKED"
        symbol = "\u2718"
    elif made:
        if hit_rim:
            status = "SCORE \u2014 rim in"
        elif hit_backboard:
            status = "SCORE \u2014 bank"
        else:
            status = "SCORE \u2014 swish"
        symbol = "\u2714"
    elif hit_rim:
        status = "MISS \u2014 rim out"
        symbol = "\u2718"
    elif hit_backboard:
        status = "MISS \u2014 off the board"
        symbol = "\u2718"
    else:
        status = "MISS \u2014 short / long"
        symbol = "\u2718"

    return {
        't': np.array(ts), 'x': xs, 'y': ys,
        'v0': v0, 'theta': theta_deg, 'wind': wind_x,
        'made': made, 'clears_defender': clears_defender,
        'defender_clearance': defender_clearance,
        'hit_backboard': hit_backboard, 'hit_rim': hit_rim,
        'hit_ground': hit_ground,
        'success': made and clears_defender,
        'status': status, 'symbol': symbol,
        'apex': float(ys.max()),
        'x_cross': x_cross,
        'aim_error': None if x_cross is None else x_cross - hoop_x,
        'flight_time': float(ts[-1]),
        'entry_angle': entry_angle,
    }


# ------------------------------------------------------------
# Quick self-test: run this file directly to simulate one
# nominal free-throw-style shot and print the outcome.
# ------------------------------------------------------------
if __name__ == "__main__":
    P = Params()
    result = simulate(v0=6.85, theta_deg=52.0, wind_x=0.0,
                       defender_x=2.8, defender_reach=2.55,
                       release_height=2.30, P=P)
    print("Shot status :", result['status'])
    print("Apex        : %.2f m" % result['apex'])
    print("Flight time : %.2f s" % result['flight_time'])
    print("Entry angle :", result['entry_angle'])
