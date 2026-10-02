import os
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from physics_engine import Params
from trajectory_simulation import simulate, aim_point

DV_GRID = (-0.2, 0.0, 0.2)
DPITCH_GRID = (-2.0, 0.0, 2.0)
DYAW_GRID = (-2.0, 0.0, 2.0)
MISS_LONG = 3.0
MISS_LAT = 3.0


class OptimizationMixin:

    def _errors(self, v, pitch, yaw):
        el, et, apex = aim_point(v, pitch, yaw, self.wind_x, self.wind_y,
                                 self.release_height, self.P)
        if el is not None:
            return el, et
        short = -MISS_LONG if apex < self.P.hoop_height else MISS_LONG
        return short, MISS_LAT

    def _solve_speed(self, pitch, yaw, lo=5.0, hi=12.0, iters=12):
        f_lo = self._errors(lo, pitch, yaw)[0]
        f_hi = self._errors(hi, pitch, yaw)[0]
        if f_lo * f_hi > 0:
            return None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self._errors(mid, pitch, yaw)[0] * f_lo > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _solve_yaw(self, v, pitch, lo=-20.0, hi=20.0, iters=10):
        f_lo = self._errors(v, pitch, lo)[1]
        f_hi = self._errors(v, pitch, hi)[1]
        if f_lo * f_hi > 0:
            return None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self._errors(v, pitch, mid)[1] * f_lo > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _solve_shot(self, pitch, rounds=1):
        yaw = 0.0
        v = self._solve_speed(pitch, yaw)
        if v is None:
            return None
        for _ in range(rounds):
            new_yaw = self._solve_yaw(v, pitch)
            if new_yaw is None:
                break
            yaw = new_yaw
            new_v = self._solve_speed(pitch, yaw)
            if new_v is None:
                break
            v = new_v
        return v, yaw

    def _auto_optimize(self):
        self.txt_note.set_text(textwrap.fill(
            "Solving speed and yaw by bisection and scoring the error "
            "margin at every launch angle...", 34))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        tol = self.P.hoop_radius - self.P.R
        best = None

        for pitch in np.arange(34.0, 68.1, 3.0):
            solved = self._solve_shot(pitch)
            if solved is None:
                continue
            v_mid, yaw_mid = solved
            r = simulate(v_mid, pitch, yaw_mid, self.wind_x, self.wind_y,
                         self.defender_x, self.defender_y,
                         self.defender_reach, self.release_height, self.P)
            if not r['success']:
                continue

            hits = 0
            total = 0
            for dv in DV_GRID:
                for dp in DPITCH_GRID:
                    for dy in DYAW_GRID:
                        el, et = self._errors(v_mid + dv, pitch + dp,
                                              yaw_mid + dy)
                        total += 1
                        if np.hypot(el, et) < tol:
                            hits += 1
            score = hits / total
            key = (score, -v_mid)
            if best is None or key > best[0]:
                best = (key, pitch, v_mid, yaw_mid, score, r['entry_angle'])

        if best is None:
            self.txt_note.set_text(textwrap.fill(
                "No scoring shot exists for this defender and wind. Move the "
                "defender back, sideways, or lower the reach.", 34))
            self.fig.canvas.draw_idle()
            return

        _, pitch, v_mid, yaw_mid, score, entry = best
        self._muted = True
        try:
            self.sl_a.set_val(round(pitch, 1))
            self.sl_b.set_val(round(yaw_mid / 0.25) * 0.25)
            self.sl_v.set_val(round(v_mid / 0.05) * 0.05)
        finally:
            self._muted = False
        self._on_change(None)

        self.txt_note.set_text(textwrap.fill(
            f"Best margin at pitch {pitch:.0f} deg, yaw {yaw_mid:+.2f} deg, "
            f"{v_mid:.2f} m/s: {score * 100:.0f}% of shots still score with "
            f"+/-0.2 m/s, +/-2 deg pitch and +/-2 deg yaw of execution "
            f"error. Entry angle {entry:.0f} deg.", 34))
        self.fig.canvas.draw_idle()


def aim_errors(P, v, pitch, yaw, wind_x=0.0, wind_y=0.0, release_height=2.30,
               dt=0.008):
    el, et, apex = aim_point(v, pitch, yaw, wind_x, wind_y, release_height,
                             P, dt=dt)
    if el is not None:
        return el, et
    short = -MISS_LONG if apex < P.hoop_height else MISS_LONG
    return short, MISS_LAT


def solve_speed_for_angle(P, pitch, yaw, wind_x, wind_y, release_height,
                          lo=5.0, hi=12.0, iters=12):
    f_lo = aim_errors(P, lo, pitch, yaw, wind_x, wind_y, release_height)[0]
    f_hi = aim_errors(P, hi, pitch, yaw, wind_x, wind_y, release_height)[0]
    if f_lo * f_hi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if aim_errors(P, mid, pitch, yaw, wind_x, wind_y,
                      release_height)[0] * f_lo > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_yaw_for_angle(P, v, pitch, wind_x, wind_y, release_height,
                        lo=-20.0, hi=20.0, iters=10):
    f_lo = aim_errors(P, v, pitch, lo, wind_x, wind_y, release_height)[1]
    f_hi = aim_errors(P, v, pitch, hi, wind_x, wind_y, release_height)[1]
    if f_lo * f_hi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if aim_errors(P, v, pitch, mid, wind_x, wind_y,
                      release_height)[1] * f_lo > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_shot(P, pitch, wind_x=0.0, wind_y=0.0, release_height=2.30,
               rounds=1):
    yaw = 0.0
    v = solve_speed_for_angle(P, pitch, yaw, wind_x, wind_y, release_height)
    if v is None:
        return None
    for _ in range(rounds):
        new_yaw = solve_yaw_for_angle(P, v, pitch, wind_x, wind_y,
                                      release_height)
        if new_yaw is None:
            break
        yaw = new_yaw
        new_v = solve_speed_for_angle(P, pitch, yaw, wind_x, wind_y,
                                      release_height)
        if new_v is None:
            break
        v = new_v
    return v, yaw


def best_shot_for_defender(P, defender_x, defender_y, defender_reach,
                           wind_x=0.0, wind_y=0.0, release_height=2.30,
                           angle_grid=None):
    if angle_grid is None:
        angle_grid = np.arange(34.0, 68.1, 3.0)
    tol = P.hoop_radius - P.R
    best = None

    for pitch in angle_grid:
        solved = solve_shot(P, pitch, wind_x, wind_y, release_height)
        if solved is None:
            continue
        v_mid, yaw_mid = solved
        r = simulate(v_mid, pitch, yaw_mid, wind_x, wind_y, defender_x,
                     defender_y, defender_reach, release_height, P)
        if not r['success']:
            continue

        hits = 0
        total = 0
        for dv in DV_GRID:
            for dp in DPITCH_GRID:
                for dy in DYAW_GRID:
                    el, et = aim_errors(P, v_mid + dv, pitch + dp,
                                        yaw_mid + dy, wind_x, wind_y,
                                        release_height)
                    total += 1
                    if np.hypot(el, et) < tol:
                        hits += 1
        score = hits / total
        key = (score, -v_mid)
        if best is None or key > best[0]:
            best = (key, pitch, v_mid, yaw_mid, score)

    if best is None:
        return None
    _, pitch, v_mid, yaw_mid, score = best
    return {'pitch': pitch, 'v0': v_mid, 'yaw': yaw_mid, 'score': score}


def success_region(P, wind_x=0.0, wind_y=0.0, release_height=2.30,
                   defender_x=None, defender_y=0.0, defender_reach=None,
                   yaw=0.0, v_vals=None, a_vals=None, dt=0.010):
    if v_vals is None:
        v_vals = np.linspace(5.0, 10.0, 45)
    if a_vals is None:
        a_vals = np.linspace(25.0, 75.0, 45)

    has_defender = defender_x is not None and defender_reach is not None
    dx = defender_x if has_defender else 1.0e6
    dy = defender_y if has_defender else 0.0
    dr = defender_reach if has_defender else 0.0

    grid = np.zeros((len(a_vals), len(v_vals)), dtype=bool)
    for i, a in enumerate(a_vals):
        for j, v in enumerate(v_vals):
            r = simulate(v, a, yaw, wind_x, wind_y, dx, dy, dr,
                         release_height, P, dt=dt)
            grid[i, j] = r['success'] if has_defender else r['made']
    return v_vals, a_vals, grid


def aim_window(P, v0, wind_x=0.0, wind_y=0.0, release_height=2.30,
               yaw_vals=None, a_vals=None, dt=0.010):
    if yaw_vals is None:
        yaw_vals = np.linspace(-8.0, 8.0, 45)
    if a_vals is None:
        a_vals = np.linspace(35.0, 70.0, 45)
    grid = np.zeros((len(a_vals), len(yaw_vals)), dtype=bool)
    for i, a in enumerate(a_vals):
        for j, b in enumerate(yaw_vals):
            r = simulate(v0, a, b, wind_x, wind_y, 1.0e6, 0.0, 0.0,
                         release_height, P, dt=dt)
            grid[i, j] = r['made']
    return yaw_vals, a_vals, grid


def defender_lateral_sweep(P, offsets, defender_x=1.0, defender_reach=3.20,
                           wind_x=0.0, wind_y=0.0, release_height=2.30):
    pitches, yaws, scores = [], [], []
    for dy in offsets:
        best = best_shot_for_defender(P, defender_x, dy, defender_reach,
                                      wind_x, wind_y, release_height)
        if best is None:
            pitches.append(np.nan)
            yaws.append(np.nan)
            scores.append(0.0)
        else:
            pitches.append(best['pitch'])
            yaws.append(best['yaw'])
            scores.append(best['score'])
    return np.array(pitches), np.array(yaws), np.array(scores)


def defender_reach_sweep(P, reach_vals, defender_x=0.8, defender_y=0.0,
                         wind_x=0.0, wind_y=0.0, release_height=2.30):
    pitches, speeds, scores = [], [], []
    for dr in reach_vals:
        best = best_shot_for_defender(P, defender_x, defender_y, dr,
                                      wind_x, wind_y, release_height)
        if best is None:
            pitches.append(np.nan)
            speeds.append(np.nan)
            scores.append(0.0)
        else:
            pitches.append(best['pitch'])
            speeds.append(best['v0'])
            scores.append(best['score'])
    return np.array(pitches), np.array(speeds), np.array(scores)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    print("  finished %s" % name)


def make_sample_trajectories_figure(P, out_dir):
    v_fix, yaw_fix = solve_shot(P, 52.0, 0.0, 3.0)
    shots = [
        dict(v0=6.85, pitch=52.0, yaw=0.0, wind_x=0.0, wind_y=0.0,
             desc='Baseline, no wind', defender_x=2.8, defender_y=0.0,
             defender_reach=2.55, color='#16a34a'),
        dict(v0=6.85, pitch=52.0, yaw=0.0, wind_x=0.0, wind_y=3.0,
             desc='3 m/s crosswind, same aim', defender_x=2.8,
             defender_y=0.0, defender_reach=1.9, color='#0ea5e9'),
        dict(v0=v_fix, pitch=52.0, yaw=yaw_fix, wind_x=0.0, wind_y=3.0,
             desc=f'Same wind, solved aim (β={yaw_fix:+.2f}°)',
             defender_x=2.8, defender_y=0.0, defender_reach=1.9,
             color='#7c3aed'),
        dict(v0=7.20, pitch=48.0, yaw=8.0, wind_x=0.0, wind_y=0.0,
             desc='Aiming around a close defender', defender_x=1.0,
             defender_y=0.55, defender_reach=3.15, color='#dc2626'),
    ]
    fig = plt.figure(figsize=(8.2, 6.0))
    ax = fig.add_subplot(111, projection='3d')
    for s in shots:
        r = simulate(s['v0'], s['pitch'], s['yaw'], s['wind_x'], s['wind_y'],
                     s['defender_x'], s['defender_y'], s['defender_reach'],
                     2.30, P)
        ax.plot(r['x'], r['y'], r['z'], color=s['color'], lw=2,
                label=f"{s['desc']}: {r['status']}")

    th = np.linspace(0.0, 2.0 * np.pi, 80)
    ax.plot(P.hoop_x + P.hoop_radius * np.cos(th),
            P.hoop_y + P.hoop_radius * np.sin(th),
            [P.hoop_height] * len(th), color='#ea580c', lw=2.5)
    ax.plot([P.backboard_x] * 2, [-P.board_half_width, P.board_half_width],
            [P.hoop_height, P.hoop_height], color='#1e3a8a', lw=2)

    ax.set_xlabel('toward basket (m)')
    ax.set_ylabel('lateral (m)')
    ax.set_zlabel('height (m)')
    ax.set_zlim(0, 4.6)
    ax.set_box_aspect((5.2, 3.0, 4.0))
    ax.view_init(elev=16, azim=-62)
    ax.set_title('Representative 3D shots', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    fig.subplots_adjust(left=0.00, right=0.86, top=0.94, bottom=0.04)
    _save(fig, out_dir, 'fig1_sample_trajectories_3d.png')
    plt.close(fig)


def make_success_region_figure(P, out_dir):
    v_vals, a_vals, grid_no_def = success_region(P)
    _, _, grid_def = success_region(P, defender_x=1.0, defender_y=0.0,
                                    defender_reach=3.15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titles = ("No defender (base paper case)",
              "Defender 1.0 m out, 3.15 m reach, centred")
    for ax, grid, title in zip(axes, (grid_no_def, grid_def), titles):
        ax.contourf(v_vals, a_vals, grid, levels=[-0.5, 0.5, 1.5],
                    colors=['#fee2e2', '#86efac'])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('Release speed (m/s)')
    axes[0].set_ylabel('Pitch angle α (deg)')
    fig.suptitle('Scoring region in (speed, pitch) space at zero yaw',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, 'fig2_success_region.png')
    plt.close(fig)


def make_aim_window_figure(P, out_dir):
    yaw_vals, a_vals, calm = aim_window(P, 6.85)
    _, _, windy = aim_window(P, 6.85, wind_y=3.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titles = ("No wind", "3 m/s crosswind (+y)")
    for ax, grid, title in zip(axes, (calm, windy), titles):
        ax.contourf(yaw_vals, a_vals, grid, levels=[-0.5, 0.5, 1.5],
                    colors=['#fee2e2', '#86efac'])
        ax.axvline(0.0, color='#475569', lw=1, ls=':')
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('Yaw angle β (deg)')
    axes[0].set_ylabel('Pitch angle α (deg)')
    fig.suptitle('Lateral aim window at 6.85 m/s — the new 3D degree of '
                 'freedom', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, 'fig3_aim_window.png')
    plt.close(fig)


def make_defender_figures(P, out_dir):
    offsets = np.linspace(-0.9, 0.9, 13)
    pitches, yaws, lat_scores = defender_lateral_sweep(P, offsets)
    threshold = P.defender_radius + P.R

    reach_vals = np.linspace(1.80, 3.19, 10)
    r_pitches, r_speeds, r_scores = defender_reach_sweep(P, reach_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(offsets, pitches, 'o-', color='#2563eb')
    for sign in (-1.0, 1.0):
        axes[0].axvline(sign * threshold, color='#dc2626', lw=1.2,
                        ls=(0, (4, 3)))
    axes[0].text(threshold, axes[0].get_ylim()[1],
                 f'  contact limit\n  |y| = R$_{{def}}$+R = {threshold:.2f} m',
                 fontsize=7.5, color='#dc2626', va='top', ha='left')
    axes[0].set_xlabel('Defender lateral offset y (m), 1.0 m out, 3.20 m reach')
    axes[0].set_ylabel('Optimal pitch α (deg)', color='#2563eb')
    axes[0].set_title('Defender lateral offset vs. required arc', fontsize=10)

    axes[1].plot(reach_vals, r_pitches, 'o-', color='#2563eb',
                 label='pitch angle')
    ax3 = axes[1].twinx()
    ax3.plot(reach_vals, r_scores * 100, 's--', color='#16a34a')
    axes[1].set_xlabel('Defender reach at 0.8 m contest distance (m)')
    axes[1].set_ylabel('Optimal pitch α (deg)', color='#2563eb')
    ax3.set_ylabel('Robustness (%)', color='#16a34a')
    ax3.set_ylim(0, 105)
    axes[1].set_title('Margin for error vs. contest height', fontsize=10)

    fig.tight_layout()
    _save(fig, out_dir, 'fig4_defender_sweeps.png')
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "figures")
    P = Params()

    print()
    print("Building the report figures. They go in")
    print(out_dir)
    print()
    print("Two of them sweep a grid of shots and two run the optimizer many")
    print("times over, so give this a couple of minutes.")
    print()

    make_sample_trajectories_figure(P, out_dir)
    make_success_region_figure(P, out_dir)
    make_aim_window_figure(P, out_dir)
    make_defender_figures(P, out_dir)

    print()
    print("All four figures are up to date.")
    print()
