import numpy as np

from physics_engine import Params, rk4_step


def simulate(v0, theta_deg, wind_x, defender_x, defender_reach,
             release_height, P: Params, dt: float = 0.010,
             collide: bool = True):
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
    x_cross = None
    t = 0.0

    hoop_x = P.hoop_x
    board_top = P.hoop_height + 0.90
    board_bot = P.hoop_height - 0.15

    while t < t_max:
        prev = state.copy()
        state = rk4_step(state, dt, wind_x, P)
        t += dt

        if clears_defender is None and state[0] >= defender_x > prev[0]:
            frac = (defender_x - prev[0]) / (state[0] - prev[0] + 1e-12)
            y_at = prev[1] + frac * (state[1] - prev[1])
            defender_clearance = y_at - P.R - defender_reach
            clears_defender = defender_clearance > 0.0

            if not clears_defender:

                state[0] = defender_x - P.R - 0.005
                state[1] = y_at

                e_block = 0.35
                state[2] = -e_block * state[2]
                state[3] = -0.5 * abs(state[3])

                xs.append(state[0]); ys.append(state[1]); ts.append(t)
                break

        for rx in ([] if not collide else (hoop_x - P.hoop_radius, hoop_x + P.hoop_radius)):
            dx, dy = state[0] - rx, state[1] - P.hoop_height
            d = np.hypot(dx, dy)
            if d < P.R:
                hit_rim = True
                nx, ny = dx / (d + 1e-12), dy / (d + 1e-12)
                vn = state[2] * nx + state[3] * ny
                if vn < 0:
                    e = 0.45
                    state[2] -= (1 + e) * vn * nx
                    state[3] -= (1 + e) * vn * ny
                state[0] = rx + nx * P.R * 1.001
                state[1] = P.hoop_height + ny * P.R * 1.001

        if not made and prev[1] >= P.hoop_height > state[1]:
            frac = (P.hoop_height - prev[1]) / (state[1] - prev[1] + 1e-12)
            x_at = prev[0] + frac * (state[0] - prev[0])
            if x_cross is None:
                x_cross = x_at
            if abs(x_at - hoop_x) < (P.hoop_radius - P.R):
                made = True
                entry_angle = np.degrees(np.arctan2(-state[3], abs(state[2])))
                state[2] *= 0.12

        if (collide and not made and board_bot <= state[1] <= board_top
                and state[0] + P.R >= P.backboard_x and state[2] > 0):
            hit_backboard = True
            state[2] = -0.55 * state[2]
            state[0] = P.backboard_x - P.R - 0.005

        xs.append(state[0]); ys.append(state[1]); ts.append(t)

        if made and state[1] < P.hoop_height - 0.75:
            break

        if state[1] <= P.R:
            hit_ground = True
            break

        if state[0] > P.backboard_x + 0.8 and state[3] < 0:
            break

    if clears_defender is None:
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

if __name__ == "__main__":
    P = Params()
    result = simulate(v0=6.85, theta_deg=52.0, wind_x=0.0,
                       defender_x=2.8, defender_reach=2.55,
                       release_height=2.30, P=P)
    print("Shot status :", result['status'])
    print("Apex        : %.2f m" % result['apex'])
    print("Flight time : %.2f s" % result['flight_time'])
    print("Entry angle :", result['entry_angle'])
