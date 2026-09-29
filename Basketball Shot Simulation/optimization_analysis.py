"""
================================================================
 MODULE 5 / 5  --  OPTIMIZATION & COMPARATIVE ANALYSIS
================================================================
 Contributor : Nur e Razin (2105141)
 Part of     : Defender-Aware Basketball Shot Simulation
               (extension of Silverberg, Tran & Adcock, 2003)

 Responsibility
   Two things, both about turning the simulator into results:

   (a) OptimizationMixin -- the interactive "Auto-Optimize"
       feature. For a chosen launch angle, the release speed
       that sends the ball through the rim centre is found by
       BISECTION root-finding on the horizontal aim error at the
       rim plane. That nominal shot is then perturbed by
       realistic execution errors (+/-0.2 m/s, +/-2 deg) and the
       fraction that still scores is used as a robustness score
       -- the "margin for error" criterion Silverberg, Tran &
       Adcock (2003) use in their own probability formulation
       (Sec. III), applied here under our defender constraint.

   (b) Stand-alone analysis / figure-generation functions (used
       for the written report, independent of the Matplotlib
       GUI) that reproduce, in spirit, the base paper's Fig.
       12-15 "rings" studies (speed vs. pitch angle scoring
       regions) -- once without a defender (reproducing the
       base paper's own case) and once with one, so the two can
       be compared directly, plus a sweep of the optimizer
       across defender positions.

   Run this file directly to regenerate every report figure
   into ./figures/.

 Depends on : numpy, matplotlib, physics_engine_mezba.py (Params),
              trajectory_simulation_siam.py (simulate)
 Used by    : main_simulation.py (OptimizationMixin, as a mixin of
              BasketballSimulator); figures/ (standalone analysis)
================================================================
"""

import os
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from physics_engine_mezba import Params
from trajectory_simulation_siam import simulate


# ============================================================
# (a) OptimizationMixin -- contributed to BasketballSimulator
# ============================================================

class OptimizationMixin:
    """
    Root-finding + robustness-scoring "Auto-Optimize" button.
    Combined with SceneMixin and UIPanelsMixin into the full
    BasketballSimulator class in main_simulation.py.
    """

    # --------------------------------------------------------
    def _aim_error(self, v, th):
        """
        Signed horizontal aim error at the rim plane (metres).
        Negative = the ball falls short of the rim centre.
        Collisions are switched off so the function stays smooth.
        """
        r = simulate(v, th, self.wind_x, self.defender_x, self.defender_reach,
                     self.release_height, self.P, dt=0.005, collide=False)
        if r['aim_error'] is not None:
            return r['aim_error']
        return -3.0 if r['apex'] < self.P.hoop_height else 3.0

    def _solve_speed(self, th, target, lo=5.0, hi=12.0, iters=22):
        """Bisect for the release speed whose aim error equals `target`."""
        f_lo = self._aim_error(lo, th) - target
        f_hi = self._aim_error(hi, th) - target
        if f_lo * f_hi > 0:
            return None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if (self._aim_error(mid, th) - target) * f_lo > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _auto_optimize(self):
        """
        Find the most forgiving shot.

        For each launch angle the release speed that sends the ball through
        the centre of the rim is found by bisection. That nominal shot is
        then disturbed by realistic execution errors (+/- 0.2 m/s in speed,
        +/- 2 deg in angle) and the fraction of disturbed shots that still
        drop through the hoop is used as a robustness score. The angle with
        the highest score wins; ties go to the softer shot. This is the
        margin-for-error criterion of Silverberg, Tran & Adcock (2003).
        """
        self.txt_note.set_text(textwrap.fill(
            "Solving the aim equation and scoring the error margin for "
            "every launch angle...", 32))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        tol = self.P.hoop_radius - self.P.R        # usable half-opening
        dv_grid = (-0.2, -0.1, 0.0, 0.1, 0.2)
        dth_grid = (-2.0, 0.0, 2.0)
        best = None                                # (score, -v, angle, speed)

        for th in np.arange(32, 70.1, 2.0):
            v_mid = self._solve_speed(th, 0.0, iters=16)
            if v_mid is None:
                continue
            r = simulate(v_mid, th, self.wind_x, self.defender_x,
                         self.defender_reach, self.release_height, self.P)
            if not r['success']:
                continue                            # blocked or unusable

            hits = 0
            for dv in dv_grid:
                for dth in dth_grid:
                    if abs(self._aim_error(v_mid + dv, th + dth)) < tol:
                        hits += 1
            score = hits / (len(dv_grid) * len(dth_grid))
            key = (score, -v_mid)
            if best is None or key > best[0]:
                best = (key, th, v_mid, score, r['entry_angle'])

        if best is None:
            self.txt_note.set_text(textwrap.fill(
                "No scoring shot exists for this defender and wind. Move the "
                "defender back or lower the reach.", 32))
            self.fig.canvas.draw_idle()
            return

        _, th, v_mid, score, entry = best
        self.sl_a.set_val(round(th, 1))
        self.sl_v.set_val(round(v_mid / 0.05) * 0.05)   # triggers _on_change
        self.txt_note.set_text(textwrap.fill(
            f"Best margin at {th:.0f} deg, {v_mid:.2f} m/s: {score * 100:.0f}% "
            f"of shots still score with +/-0.2 m/s and +/-2 deg of execution "
            f"error. Entry angle {entry:.0f} deg.", 32))
        self.fig.canvas.draw_idle()


# ============================================================
# (b) Stand-alone comparative analysis (no GUI needed)
# ============================================================

def success_region(P, wind_x=0.0, release_height=2.30,
                    defender_x=None, defender_reach=None,
                    v_vals=None, a_vals=None, dt=0.010):
    """
    Grid-search (v0, launch angle) -> made/blocked, the same style of
    study as the base paper's Fig. 12-15 "rings", extended so it can
    optionally honour the defender constraint.

    With defender_x/defender_reach = None the defender is effectively
    disabled (placed far downstream), which reproduces the base
    paper's own no-defender case for a direct comparison.
    """
    if v_vals is None:
        v_vals = np.linspace(5.0, 10.0, 55)
    if a_vals is None:
        a_vals = np.linspace(25, 75, 55)

    has_defender = defender_x is not None and defender_reach is not None
    dx = defender_x if has_defender else 1.0e6
    dr = defender_reach if has_defender else 0.0

    grid = np.zeros((len(a_vals), len(v_vals)), dtype=bool)
    for i, a in enumerate(a_vals):
        for j, v in enumerate(v_vals):
            r = simulate(v, a, wind_x, dx, dr, release_height, P, dt=dt)
            grid[i, j] = r['success'] if has_defender else r['made']
    return v_vals, a_vals, grid


def solve_speed_for_angle(P, theta, wind_x, defender_x, defender_reach,
                           release_height, target=0.0, lo=5.0, hi=12.0,
                           iters=18):
    """Stand-alone bisection identical in spirit to OptimizationMixin,
    used here for batch analysis instead of a live GUI slider."""

    def aim_error(v):
        r = simulate(v, theta, wind_x, defender_x, defender_reach,
                      release_height, P, dt=0.006, collide=False)
        if r['aim_error'] is not None:
            return r['aim_error']
        return -3.0 if r['apex'] < P.hoop_height else 3.0

    f_lo = aim_error(lo) - target
    f_hi = aim_error(hi) - target
    if f_lo * f_hi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (aim_error(mid) - target) * f_lo > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def best_shot_for_defender(P, defender_x, defender_reach, wind_x=0.0,
                            release_height=2.30, angle_grid=None):
    """Best (most robust) launch angle / speed pair for one defender
    position, using the same margin-for-error scoring as Auto-Optimize."""
    if angle_grid is None:
        angle_grid = np.arange(32, 68.1, 3.0)
    tol = P.hoop_radius - P.R
    dv_grid = (-0.2, -0.1, 0.0, 0.1, 0.2)
    dth_grid = (-2.0, 0.0, 2.0)
    best = None

    for th in angle_grid:
        v_mid = solve_speed_for_angle(P, th, wind_x, defender_x,
                                       defender_reach, release_height,
                                       iters=14)
        if v_mid is None:
            continue
        r = simulate(v_mid, th, wind_x, defender_x, defender_reach,
                     release_height, P)
        if not r['success']:
            continue

        hits = 0
        for dv in dv_grid:
            for dth in dth_grid:
                rr = simulate(v_mid + dv, th + dth, wind_x, defender_x,
                              defender_reach, release_height, P,
                              dt=0.006, collide=False)
                err = rr['aim_error'] if rr['aim_error'] is not None else 99.0
                if abs(err) < tol:
                    hits += 1
        score = hits / (len(dv_grid) * len(dth_grid))
        key = (score, -v_mid)
        if best is None or key > best[0]:
            best = (key, th, v_mid, score)

    if best is None:
        return None
    _, th, v_mid, score = best
    return {'theta': th, 'v0': v_mid, 'score': score}


def defender_reach_sensitivity_sweep(P, defender_reach_vals, defender_x=0.8,
                                      wind_x=0.0, release_height=2.30):
    """Sweep how high the defender contests the shot (a close-range
    defender at defender_x) and record the optimizer's best angle /
    speed / robustness score at each contest height. Close range is
    used because at the slider's normal (>1.5 m) defender positions a
    reach inside [1.8, 3.2] m essentially never threatens a well-timed
    high-arcing shot released at ~2.3 m -- the defender only forces a
    real trade-off when the shot is still low soon after release."""
    thetas, speeds, scores = [], [], []
    for dr in defender_reach_vals:
        best = best_shot_for_defender(P, defender_x, dr, wind_x,
                                       release_height)
        if best is None:
            thetas.append(np.nan); speeds.append(np.nan); scores.append(0.0)
        else:
            thetas.append(best['theta'])
            speeds.append(best['v0'])
            scores.append(best['score'])
    return np.array(thetas), np.array(speeds), np.array(scores)


# ------------------------------------------------------------
# Figure generation for the written report
# ------------------------------------------------------------

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print("saved", path)


def make_success_region_figure(P, out_dir):
    v_vals, a_vals, grid_no_def = success_region(P)
    _, _, grid_def = success_region(P, defender_x=1.0, defender_reach=3.15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, grid, title in zip(
            axes, (grid_no_def, grid_def),
            ("No defender (reproduces base paper, Fig. 12 style)",
             "Close, high-reaching defender: 1.0 m out, 3.15 m reach")):
        ax.contourf(v_vals, a_vals, grid, levels=[-0.5, 0.5, 1.5],
                    colors=['#fee2e2', '#86efac'])
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('Release speed (m/s)')
    axes[0].set_ylabel('Launch angle (deg)')
    fig.suptitle('Scoring region in (speed, angle) space', fontsize=12,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir, 'fig1_success_region.png')
    plt.close(fig)


def make_defender_sweep_figure(P, out_dir):
    reach_vals = np.linspace(1.80, 3.19, 12)
    thetas, speeds, scores = defender_reach_sensitivity_sweep(
        P, reach_vals, defender_x=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(reach_vals, thetas, 'o-', color='#2563eb', label='launch angle')
    ax2 = axes[0].twinx()
    ax2.plot(reach_vals, speeds, 's-', color='#f97316', label='release speed')
    axes[0].set_xlabel('Defender reach at 0.8 m contest distance (m)')
    axes[0].set_ylabel('Optimal launch angle (deg)', color='#2563eb')
    ax2.set_ylabel('Optimal release speed (m/s)', color='#f97316')
    axes[0].set_title('Optimal shot vs. contest height', fontsize=10)

    axes[1].plot(reach_vals, scores * 100, 'o-', color='#16a34a')
    axes[1].set_xlabel('Defender reach at 0.8 m contest distance (m)')
    axes[1].set_ylabel('Robustness score (%)')
    axes[1].set_title('Margin for error vs. contest height', fontsize=10)
    axes[1].set_ylim(0, 105)

    fig.tight_layout()
    _save(fig, out_dir, 'fig2_defender_sweep.png')
    plt.close(fig)


def make_sample_trajectories_figure(P, out_dir):
    # Each entry is a starting condition together with a short plain-
    # English *description* of the scenario; the actual outcome label
    # shown in the legend is always read back from the simulator's own
    # r['status'] rather than assumed, so the figure can never claim an
    # outcome the physics engine did not actually produce.
    shots = [
        dict(v0=6.85, theta_deg=52.0, wind_x=0.0, desc='Baseline shot',
             defender_x=2.8, defender_reach=2.55, color='#16a34a'),
        dict(v0=8.50, theta_deg=45.0, wind_x=0.0, desc='Too hard / flat',
             defender_x=2.8, defender_reach=1.9, color='#0ea5e9'),
        dict(v0=7.50, theta_deg=40.0, wind_x=0.0, desc='Close, high contest',
             defender_x=1.0, defender_reach=3.15, color='#dc2626'),
        dict(v0=6.30, theta_deg=45.0, wind_x=-4.0, desc='Strong headwind',
             defender_x=2.8, defender_reach=1.9, color='#a16207'),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in shots:
        r = simulate(s['v0'], s['theta_deg'], s['wind_x'], s['defender_x'],
                     s['defender_reach'], 2.30, P)
        ax.plot(r['x'], r['y'], color=s['color'], lw=2,
                label=f"{s['desc']}: {r['status']}")
    ax.axhline(P.hoop_height, color='#9ca3af', lw=1, ls=':')
    ax.axvline(P.hoop_x, color='#9ca3af', lw=1, ls=':')
    ax.set_xlabel('Horizontal distance from shooter (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Representative simulated shots', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(alpha=0.25, ls=':')
    fig.tight_layout()
    _save(fig, out_dir, 'fig3_sample_trajectories.png')
    plt.close(fig)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "figures")
    P = Params()
    print("Generating report figures in:", out_dir)
    make_sample_trajectories_figure(P, out_dir)
    make_success_region_figure(P, out_dir)
    make_defender_sweep_figure(P, out_dir)
    print("Done.")
