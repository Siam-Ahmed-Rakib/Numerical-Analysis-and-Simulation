import numpy as np

from physics_engine import Params, rk4_step

RIM_RESTITUTION = 0.45
BOARD_RESTITUTION = 0.55
BLOCK_RESTITUTION = 0.35
NET_DAMPING = 0.12


def launch_velocity(v0, pitch_deg, yaw_deg):
    a = np.radians(pitch_deg)
    b = np.radians(yaw_deg)
    return np.array([v0 * np.cos(a) * np.cos(b),
                     v0 * np.cos(a) * np.sin(b),
                     v0 * np.sin(a)])


def aim_point(v0, pitch_deg, yaw_deg, wind_x, wind_y, release_height,
              P: Params, dt: float = 0.008):
    wind = np.array([wind_x, wind_y, 0.0])
    state = np.concatenate((
        np.array([P.release_x, P.release_y, release_height]),
        launch_velocity(v0, pitch_deg, yaw_deg)))

    apex = state[2]
    t = 0.0
    while t < 5.0:
        px, py, pz = state[0], state[1], state[2]
        state = rk4_step(state, dt, wind, P)
        t += dt
        if state[2] > apex:
            apex = state[2]
        if pz >= P.hoop_height > state[2]:
            frac = (P.hoop_height - pz) / (state[2] - pz - 1e-12)
            return (px + frac * (state[0] - px) - P.hoop_x,
                    py + frac * (state[1] - py) - P.hoop_y,
                    float(apex))
        if state[2] <= 0.0:
            break
    return None, None, float(apex)


def simulate(v0, pitch_deg, yaw_deg, wind_x, wind_y, defender_x, defender_y,
             defender_reach, release_height, P: Params, dt: float = 0.010,
             collide: bool = True):
    wind = np.array([wind_x, wind_y, 0.0])
    state = np.concatenate((
        np.array([P.release_x, P.release_y, release_height]),
        launch_velocity(v0, pitch_deg, yaw_deg)))

    xs = [state[0]]
    ys = [state[1]]
    zs = [state[2]]
    ts = [0.0]

    clears_defender = None
    vertical_gap = None
    lateral_gap = None
    hit_backboard = False
    hit_rim = False
    hit_ground = False
    made = False
    entry_angle = None
    cross = None
    t = 0.0
    t_max = 5.0

    hoop_x = P.hoop_x
    hoop_y = P.hoop_y
    usable = P.hoop_radius - P.R

    while t < t_max:
        prev = state.copy()
        state = rk4_step(state, dt, wind, P)
        t += dt

        if collide and clears_defender is None and state[0] >= defender_x > prev[0]:
            frac = (defender_x - prev[0]) / (state[0] - prev[0] + 1e-12)
            y_at = prev[1] + frac * (state[1] - prev[1])
            z_at = prev[2] + frac * (state[2] - prev[2])
            lateral_gap = abs(y_at - defender_y) - (P.defender_radius + P.R)
            vertical_gap = z_at - P.R - defender_reach
            clears_defender = lateral_gap > 0.0 or vertical_gap > 0.0

            if not clears_defender:
                state[0] = defender_x - P.R - 0.005
                state[1] = y_at
                state[2] = z_at
                side = 1.0 if y_at >= defender_y else -1.0
                state[3] = -BLOCK_RESTITUTION * state[3]
                state[4] = -BLOCK_RESTITUTION * state[4] + 0.45 * side
                state[5] = -0.5 * abs(state[5])
                xs.append(state[0])
                ys.append(state[1])
                zs.append(state[2])
                ts.append(t)
                break

        if collide:
            ox = state[0] - hoop_x
            oy = state[1] - hoop_y
            radial = np.hypot(ox, oy)
            if radial > 1e-9:
                near = np.array([hoop_x + P.hoop_radius * ox / radial,
                                 hoop_y + P.hoop_radius * oy / radial,
                                 P.hoop_height])
                offset = state[:3] - near
                dist = float(np.linalg.norm(offset))
                if dist < P.R:
                    hit_rim = True
                    n = offset / (dist + 1e-12)
                    vn = float(np.dot(state[3:], n))
                    if vn < 0:
                        state[3:] -= (1.0 + RIM_RESTITUTION) * vn * n
                    state[:3] = near + n * P.R * 1.001

        if not made and prev[2] >= P.hoop_height > state[2]:
            frac = (P.hoop_height - prev[2]) / (state[2] - prev[2] + 1e-12)
            x_at = prev[0] + frac * (state[0] - prev[0])
            y_at = prev[1] + frac * (state[1] - prev[1])
            if cross is None:
                cross = (x_at, y_at)
            if np.hypot(x_at - hoop_x, y_at - hoop_y) < usable:
                made = True
                entry_angle = float(np.degrees(np.arctan2(
                    -state[5], np.hypot(state[3], state[4]))))
                state[3] *= NET_DAMPING
                state[4] *= NET_DAMPING

        if (collide and not made
                and P.board_bottom <= state[2] <= P.board_top
                and abs(state[1] - P.backboard_y) <= P.board_half_width
                and state[0] + P.R >= P.backboard_x and state[3] > 0):
            hit_backboard = True
            state[3] = -BOARD_RESTITUTION * state[3]
            state[0] = P.backboard_x - P.R - 0.005

        xs.append(state[0])
        ys.append(state[1])
        zs.append(state[2])
        ts.append(t)

        if made and state[2] < P.hoop_height - 0.75:
            break
        if state[2] <= P.R:
            hit_ground = True
            break
        if state[0] > P.backboard_x + 0.8 and state[5] < 0:
            break
        if abs(state[1]) > 3.2:
            break

    if clears_defender is None:
        if collide:
            clears_defender = False
            vertical_gap = -defender_reach
            lateral_gap = -P.defender_radius
        else:
            clears_defender = True

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    if not clears_defender:
        status = "BLOCKED"
        symbol = "✘"
    elif made:
        if hit_rim:
            status = "SCORE — rim in"
        elif hit_backboard:
            status = "SCORE — bank"
        else:
            status = "SCORE — swish"
        symbol = "✔"
    elif hit_rim:
        status = "MISS — rim out"
        symbol = "✘"
    elif hit_backboard:
        status = "MISS — off the board"
        symbol = "✘"
    else:
        status = "MISS — short / long / wide"
        symbol = "✘"

    aim_error = None if cross is None else cross[0] - hoop_x
    lateral_error = None if cross is None else cross[1] - hoop_y
    radial_error = (None if cross is None
                    else float(np.hypot(aim_error, lateral_error)))

    return {
        't': np.array(ts), 'x': xs, 'y': ys, 'z': zs,
        'v0': v0, 'pitch': pitch_deg, 'yaw': yaw_deg,
        'wind_x': wind_x, 'wind_y': wind_y,
        'made': made, 'clears_defender': clears_defender,
        'defender_clearance': vertical_gap,
        'defender_lateral': lateral_gap,
        'hit_backboard': hit_backboard, 'hit_rim': hit_rim,
        'hit_ground': hit_ground,
        'success': made and clears_defender,
        'status': status, 'symbol': symbol,
        'apex': float(zs.max()),
        'max_drift': float(np.abs(ys).max()),
        'cross': cross,
        'aim_error': aim_error,
        'lateral_error': lateral_error,
        'radial_error': radial_error,
        'flight_time': float(ts[-1]),
        'entry_angle': entry_angle,
    }


if __name__ == "__main__":
    P = Params()
    shots = [
        ("Aimed straight, still air", 0.0, 0.0, 0.0),
        ("Aimed 6 degrees off line", 6.0, 0.0, 0.0),
        ("Aimed straight into a 3 m/s crosswind", 0.0, 0.0, 3.0),
    ]

    print()
    print("Three shots from the free-throw line, all released at 6.85 m/s and")
    print("52 degrees. Only the aim and the wind change.")
    print()

    for name, yaw, wx, wy in shots:
        r = simulate(v0=6.85, pitch_deg=52.0, yaw_deg=yaw,
                     wind_x=wx, wind_y=wy, defender_x=2.8, defender_y=0.0,
                     defender_reach=2.55, release_height=2.30, P=P)
        print(name)
        print("  %s" % r['status'])
        print("  reached %.2f m at the top and wandered %.2f m sideways"
              % (r['apex'], r['max_drift']))
        if r['lateral_error'] is not None:
            side = abs(r['lateral_error'] * 100.0)
            if side < 1.0:
                print("  came down through the middle of the rim")
            else:
                print("  crossed the rim %.0f cm to the side of centre" % side)
        print()
