import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from physics_engine import Params
from trajectory_simulation import simulate
from scene_visualization import SceneMixin
from ui_interaction import UIPanelsMixin
from optimization_analysis import OptimizationMixin


class BasketballSimulator(SceneMixin, UIPanelsMixin, OptimizationMixin):

    ARM_REACH = 0.45
    REACH_RATIO = 1.32

    def __init__(self, show=True):
        self.P = Params()
        self._muted = False

        self.v0 = 6.85
        self.pitch = 52.0
        self.yaw = 0.0
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.shooter_h = 1.85
        self.defender_x = 2.8
        self.defender_y = 0.0
        self.defender_reach = 2.55

        self.result = self._run()

        self.fig = plt.figure(figsize=(15.0, 9.0))
        self.fig.patch.set_facecolor('#f7f7fa')
        try:
            self.fig.canvas.manager.set_window_title(
                "3D Basketball Shot Simulator  —  Silverberg et al. Extension")
        except Exception:
            pass

        self.fig.text(0.5, 0.975,
                      "Defender-Aware Basketball Shot — 3D Animated Simulation",
                      ha='center', va='top', fontsize=15, fontweight='bold',
                      color='#111827')
        self.fig.text(0.5, 0.948,
                      "RK4 integration of the full three-dimensional "
                      "trajectory with aerodynamic drag, crosswind and a "
                      "defender modelled as a reach cylinder",
                      ha='center', va='top', fontsize=9.5, color='#4b5563')

        self.ax = self.fig.add_axes([-0.015, 0.190, 0.530, 0.760],
                                    projection='3d')
        self.ax_plan = self.fig.add_axes([0.540, 0.605, 0.185, 0.310])
        self.ax_model = self.fig.add_axes([0.760, 0.605, 0.225, 0.310])
        self.ax_info = self.fig.add_axes([0.505, 0.215, 0.480, 0.340])

        self._setup_court()
        self._setup_plan()
        self._setup_info_panel()
        self._setup_model_panel()

        self.ghost, = self.ax.plot([], [], [], color='#94a3b8', lw=1.2,
                                   ls=(0, (5, 4)), alpha=0.9)
        self.trail, = self.ax.plot([], [], [], color='#ef4444', lw=2.4,
                                   alpha=0.95)
        self.ball, = self.ax.plot([], [], [], ls='none', marker='o', ms=13,
                                  color='#f97316', mec='#7c2d12', mew=1.5)

        self._create_sliders()
        self._create_buttons()

        self.anim = None
        self.frame = 0
        self.step = 1
        self.n_frames = 1
        self._refresh_static()
        self.start_animation()

        if show:
            plt.show()

    @property
    def release_height(self):
        return self.shooter_h + self.ARM_REACH

    def _run(self):
        return simulate(self.v0, self.pitch, self.yaw, self.wind_x,
                        self.wind_y, self.defender_x, self.defender_y,
                        self.defender_reach, self.release_height, self.P)

    def _resimulate(self):
        self.result = self._run()
        self._refresh_static()
        self.start_animation()

    def start_animation(self):
        n = len(self.result['x'])
        self.step = max(1, n // 50)
        self.n_frames = len(range(0, n, self.step)) + 10
        self.frame = 0

        if self.anim is None:
            self.anim = animation.FuncAnimation(
                self.fig, self._update, frames=itertools.count,
                interval=32, blit=False, repeat=False,
                cache_frame_data=False)
        else:
            try:
                self.anim.event_source.start()
            except Exception:
                pass
        self.fig.canvas.draw_idle()

    def _update(self, _i):
        r = self.result
        n = len(r['x'])
        idx = min(self.frame * self.step, n - 1)
        bx, by, bz = r['x'][idx], r['y'][idx], r['z'][idx]

        self.ball.set_data([bx], [by])
        self.ball.set_3d_properties([bz])
        self.trail.set_data(r['x'][:idx + 1], r['y'][:idx + 1])
        self.trail.set_3d_properties(r['z'][:idx + 1])

        self.plan_ball.set_data([bx], [by])
        self.plan_trail.set_data(r['x'][:idx + 1], r['y'][:idx + 1])

        if self.frame >= self.n_frames - 1:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        else:
            self.frame += 1
        return self.ball, self.trail, self.plan_ball, self.plan_trail


if __name__ == "__main__":
    print()
    print("Defender-aware basketball shot simulator, in three dimensions.")
    print("An extension of Silverberg, Tran and Adcock (2003).")
    print()
    print("The ball is carried forward with RK4 over the state")
    print("[x, y, z, vx, vy, vz], so a shot can drift sideways as well as rise")
    print("and fall. Drag and wind act in all three directions, and the")
    print("defender is a cylinder you can go over or around.")
    print()
    print("Nine sliders set up the shot: release speed, pitch and yaw, the")
    print("headwind and crosswind, the shooter's height, and where the")
    print("defender stands and how high they reach.")
    print()
    print("Replay runs the current shot again. Auto-Optimize searches for the")
    print("most forgiving shot that still scores, and takes a few seconds.")
    print("Reset puts everything back.")
    print()
    print("Drag the 3D view with the mouse to rotate the court.")
    print()
    BasketballSimulator()
