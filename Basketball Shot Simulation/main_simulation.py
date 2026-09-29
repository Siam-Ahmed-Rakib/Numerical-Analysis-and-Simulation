"""
================================================================
 COMBINED FILE  --  DEFENDER-AWARE BASKETBALL SHOT SIMULATION
================================================================
 Extension of Silverberg, Tran & Adcock (2003),
 "Numerical Analysis of the Basketball Shot",
 J. Dynamic Systems, Measurement, and Control, Vol. 125, pp. 531-540.



 What this file does
   It does NOT reimplement any physics, geometry or UI logic --
   every equation and every widget lives in exactly one of the
   five module files above, each written by the member named
   next to it. This file is the integration harness: it imports
   the five modules, assembles the BasketballSimulator class out
   of their three mixins (SceneMixin, UIPanelsMixin,
   OptimizationMixin), wires up the parts that only make sense
   once everything is together (the constructor, the per-frame
   animation callback, and the re-simulate-on-slider-change
   glue), and launches the live, animated, six-slider simulator.

 Physics summary
   - RK4 integration of projectile motion (Mezba)
   - Quadratic aerodynamic drag with horizontal wind -- our
     reinstatement of the effect the base paper explicitly
     neglects (Mezba)
   - Rim / backboard collision, swish vs. rim-hit detection,
     and a defender modelled as a vertical reach barrier -- our
     main extension beyond the base paper (Siam)
   - Animated ball with a fading motion trail; shooter and
     defender drawn to scale, both slider-controlled (Alif)
   - Six-slider / three-button control surface with live
     read-outs (Imdadul)
   - Auto-Optimize: bisection root-finding + a margin-for-error
     robustness score, in the spirit of the base paper's own
     shot-probability formulation (Razin)

 Run:  python main_simulation.py
 Deps: numpy, matplotlib
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from physics_engine import Params
from trajectory_simulation import simulate
from scene_visualization import SceneMixin
from ui_interaction import UIPanelsMixin
from optimization_analysis import OptimizationMixin


# ============================================================
# INTERACTIVE ANIMATED SIMULATOR
# (assembled from the three mixins contributed above)
# ============================================================

class BasketballSimulator(SceneMixin, UIPanelsMixin, OptimizationMixin):

    ARM_REACH = 0.45        # release point above the shooter's head
    REACH_RATIO = 1.32      # defender reach / standing height

    def __init__(self, show=True):
        self.P = Params()

        # ---- adjustable state
        self.v0 = 6.85
        self.theta = 52.0
        self.wind_x = 0.0
        self.shooter_h = 1.85
        self.defender_x = 2.8
        self.defender_reach = 2.55

        self.result = self._run()

        # ---- figure
        self.fig = plt.figure(figsize=(15.0, 9.0))
        self.fig.patch.set_facecolor('#f7f7fa')
        try:
            self.fig.canvas.manager.set_window_title(
                "Basketball Shot Simulator  —  Silverberg et al. Extension")
        except Exception:
            pass

        self.fig.text(0.5, 0.970,
                      "Defender-Aware Basketball Shot — Animated Simulation",
                      ha='center', va='top', fontsize=16, fontweight='bold',
                      color='#111827')
        self.fig.text(0.5, 0.940,
                      "RK4 projectile integration with aerodynamic drag, "
                      "wind and a defender reach constraint",
                      ha='center', va='top', fontsize=10, color='#4b5563')

        # Court on the left, read-out panel on the right: nothing overlaps.
        self.ax = self.fig.add_axes([0.045, 0.265, 0.515, 0.655])
        self.ax_info = self.fig.add_axes([0.580, 0.265, 0.195, 0.655])
        self.ax_model = self.fig.add_axes([0.795, 0.265, 0.185, 0.655])

        self._setup_court()            # SceneMixin        (Alif)
        self._setup_info_panel()       # UIPanelsMixin      (Imdadul)
        self._setup_model_panel()      # UIPanelsMixin      (Imdadul)

        # ---- animated artists
        self.ball = Circle((self.P.release_x, self.release_height), self.P.R,
                           facecolor='#f97316', edgecolor='#7c2d12',
                           lw=1.8, zorder=12)
        self.ax.add_patch(self.ball)
        self.ball_seam, = self.ax.plot([], [], color='#7c2d12', lw=1.2,
                                       zorder=13)
        self.trail, = self.ax.plot([], [], color='#ef4444', lw=2.6,
                                   alpha=0.95, zorder=7)
        self.ghost, = self.ax.plot(self.result['x'], self.result['y'],
                                   color='#94a3b8', lw=1.2, ls=(0, (5, 4)),
                                   alpha=0.85, zorder=6)

        self._create_sliders()         # UIPanelsMixin      (Imdadul)
        self._create_buttons()         # UIPanelsMixin      (Imdadul)

        self.anim = None
        self.frame = 0
        self._refresh_static()         # SceneMixin         (Alif)
        self.start_animation()

        if show:
            plt.show()

    # --------------------------------------------------------
    @property
    def release_height(self):
        return self.shooter_h + self.ARM_REACH

    def _run(self):
        return simulate(self.v0, self.theta, self.wind_x, self.defender_x,
                        self.defender_reach, self.release_height, self.P)

    # --------------------------------------------------------
    def _resimulate(self):
        self.result = self._run()
        self._refresh_static()         # SceneMixin (Alif) -> _update_panel (Imdadul)
        self.start_animation()

    # --------------------------------------------------------
    def start_animation(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        n = len(self.result['x'])
        self.step = max(1, n // 50)
        self.n_frames = len(range(0, n, self.step)) + 12   # hold at the end

        self.anim = animation.FuncAnimation(
            self.fig, self._update, frames=self.n_frames,
            interval=22, blit=False, repeat=False)
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    def _update(self, i):
        n = len(self.result['x'])
        idx = min(i * self.step, n - 1)
        bx = self.result['x'][idx]
        by = self.result['y'][idx]

        self.ball.center = (bx, by)
        # spinning seam: an arc across the ball whose curvature cycles
        u = np.linspace(-1, 1, 25)
        bulge = np.cos(idx * 0.12)
        self.ball_seam.set_data(bx + self.P.R * 0.92 * u,
                                by + self.P.R * 0.75 * bulge *
                                np.sqrt(np.clip(1 - u ** 2, 0, 1)))
        self.trail.set_data(self.result['x'][:idx + 1],
                            self.result['y'][:idx + 1])
        return self.ball, self.trail, self.ball_seam


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print(" Defender-Aware Basketball Shot Simulator  (v2.0)")
    print(" Group project -- extension of Silverberg, Tran & Adcock (2003)")
    print("=" * 64)
    print(" Sliders : release speed, launch angle, wind,")
    print("           shooter height, defender position, defender reach")
    print(" Replay  : re-run the current shot animation")
    print(" Optimize: softest shot that scores and clears the defender")
    print(" Reset   : restore the default configuration")
    print("=" * 64)
    BasketballSimulator()
