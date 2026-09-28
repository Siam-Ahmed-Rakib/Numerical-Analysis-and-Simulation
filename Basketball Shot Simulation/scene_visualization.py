"""
================================================================
 MODULE 3 / 5  --  SCENE & STICK-FIGURE VISUALIZATION
================================================================
 Contributor : Shahriar Alif (2105158)
 Part of     : Defender-Aware Basketball Shot Simulation
               (extension of Silverberg, Tran & Adcock, 2003)

 Responsibility
   This module draws the static parts of the scene: the court
   floor, backboard, rim, net, and the two people involved in
   the extended problem -- the shooter and the defender -- as
   simple, height-parametrised stick figures. Both figures are
   redrawn live whenever the corresponding height slider moves,
   which is what lets the defender's "reach" (our new geometric
   constraint, absent from the base paper) be explored
   interactively.

   Two things live here:
     1. StickFigure  -- a small reusable class: give it an (x,
        height) and it draws a head / body / legs / arms and
        keeps a live height label. `arms_up=True` draws a
        contesting defender; `arms_up=False` draws a shooter.
     2. SceneMixin    -- a mixin contributed to the main
        BasketballSimulator class (assembled in
        main_simulation.py) providing:
           _setup_court()    : draws the one-time static scene
           _refresh_static() : updates defender/shooter geometry
                                and the "ghost" (full predicted
                                path) whenever a slider changes

 Depends on : numpy, matplotlib.patches (Circle, Rectangle, Ellipse)
 Used by    : main_simulation.py (as a mixin of BasketballSimulator)
================================================================
"""

import numpy as np
from matplotlib.patches import Circle, Rectangle, Ellipse


# ============================================================
# 4. STICK FIGURE (redrawn live when heights change)
# ============================================================

class StickFigure:
    """A stick figure whose height can be updated on the fly."""

    def __init__(self, ax, x, height, color, arms_up=False, label=''):
        self.ax = ax
        self.color = color
        self.arms_up = arms_up

        self.head = Circle((0, 0), 0.01, color=color, zorder=9)
        ax.add_patch(self.head)
        self.body,  = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.leg_l, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.leg_r, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.arm_l, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.arm_r, = ax.plot([], [], color=color, lw=4, zorder=9,
                              solid_capstyle='round')
        self.name = label
        self.label = ax.text(0, 0, label, ha='center', va='bottom',
                             fontsize=9, fontweight='bold', color=color,
                             linespacing=1.3, zorder=9)
        self.update(x, height)

    def update(self, x, height, top=None):
        """
        height : standing height (top of the head)
        top    : height the raised hands reach (defender reach / release
                 point). Defaults to a little above the head.
        """
        h = height
        top = (h + 0.40) if top is None else top
        head_r = 0.105
        head_y = h - head_r
        shoulder = h - 2 * head_r
        hip = h * 0.48

        self.head.center = (x, head_y)
        self.head.set_radius(head_r)
        self.body.set_data([x, x], [shoulder, hip])
        self.leg_l.set_data([x, x - 0.16], [hip, 0])
        self.leg_r.set_data([x, x + 0.16], [hip, 0])

        if self.arms_up:                       # contesting the shot
            self.arm_l.set_data([x, x - 0.15], [shoulder, top])
            self.arm_r.set_data([x, x + 0.15], [shoulder, top])
        else:                                  # shooting motion
            self.arm_l.set_data([x, x + 0.04], [shoulder, top - 0.10])
            self.arm_r.set_data([x, x + 0.17], [shoulder, top])

        # height read-out travels with the figure, above the hands
        self.label.set_position((x, top + 0.14))
        self.label.set_text(f"{self.name}\n{h:.2f} m")


# ============================================================
# SceneMixin -- contributed to BasketballSimulator
# ============================================================

class SceneMixin:
    """
    Static scene set-up and geometry refresh. Combined with
    UIPanelsMixin and OptimizationMixin into the full
    BasketballSimulator class in main_simulation.py.
    """

    # --------------------------------------------------------
    def _setup_court(self):
        ax, P = self.ax, self.P
        ax.set_facecolor('#ffffff')
        ax.set_xlim(-0.95, P.backboard_x + 1.05)
        ax.set_ylim(0, 5.4)
        ax.set_xlabel('Horizontal distance from shooter (m)',
                      fontsize=11, labelpad=8)
        ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.25, ls=':', zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

        # ---- floor
        ax.axhspan(-0.2, 0, color='#c8a27a', zorder=1)
        ax.axhline(0, color='#7c4a1e', lw=3, zorder=2)

        hoop_x = P.hoop_x
        h = P.hoop_height

        # ---- support pole and arm
        pole_x = P.backboard_x + 0.55
        ax.plot([pole_x, pole_x], [0, h + 0.55],
                color='#64748b', lw=7, solid_capstyle='round', zorder=3)
        ax.plot([P.backboard_x, pole_x], [h + 0.55, h + 0.55],
                color='#64748b', lw=6, solid_capstyle='round', zorder=3)

        # ---- backboard (panel + shooter's square)
        ax.add_patch(Rectangle((P.backboard_x, h - 0.15), 0.06, 1.05,
                               facecolor='#dbeafe', edgecolor='#1e3a8a',
                               lw=2.5, zorder=4))
        ax.plot([P.backboard_x + 0.005, P.backboard_x + 0.005],
                [h + 0.05, h + 0.50], color='#1e3a8a', lw=3, zorder=5)

        # ---- rim: an open ellipse gives a visible "hole"
        ax.plot([P.backboard_x, hoop_x + P.hoop_radius], [h, h],
                color='#9a3412', lw=3, zorder=5)                 # rim mount
        ax.add_patch(Ellipse((hoop_x, h), 2 * P.hoop_radius, 0.11,
                             facecolor='none', edgecolor='#ea580c',
                             lw=4, zorder=8))
        ax.add_patch(Ellipse((hoop_x, h), 2 * P.hoop_radius, 0.11,
                             facecolor='#ffffff', edgecolor='none',
                             alpha=0.9, zorder=6))               # the hole

        # ---- net
        net_bottom = h - 0.38
        for f in np.linspace(-1, 1, 9):
            x_top = hoop_x + f * P.hoop_radius
            x_bot = hoop_x + f * P.hoop_radius * 0.55
            ax.plot([x_top, x_bot], [h - 0.02, net_bottom],
                    color='#9ca3af', lw=1.0, zorder=5)
        for frac in (0.33, 0.66, 1.0):
            y = h - 0.02 - frac * 0.36
            w = P.hoop_radius * (1 - 0.45 * frac)
            ax.plot([hoop_x - w, hoop_x + w], [y, y],
                    color='#9ca3af', lw=1.0, zorder=5)

        # ---- defender reach zone  (our new geometric constraint)
        self.defender_patch = Rectangle(
            (self.defender_x - 0.22, 0), 0.44, self.defender_reach,
            facecolor='#ef4444', alpha=0.13, edgecolor='none', zorder=3)
        ax.add_patch(self.defender_patch)
        self.defender_line, = ax.plot(
            [self.defender_x - 0.34, self.defender_x + 0.34],
            [self.defender_reach] * 2,
            color='#b91c1c', lw=2, ls=(0, (4, 3)), zorder=6)
        self.reach_tag = ax.text(
            self.defender_x + 0.40, self.defender_reach, '', fontsize=9,
            color='#b91c1c', va='center', ha='left', zorder=6)

        # ---- people (labels sit above the heads, clear of the tick labels)
        self.shooter_fig = StickFigure(ax, 0.0, self.shooter_h, '#1f2937',
                                       arms_up=False, label='Shooter')
        self.shooter_fig.update(0.0, self.shooter_h, self.release_height)
        self.defender_fig = StickFigure(
            ax, self.defender_x, self.defender_reach / self.REACH_RATIO,
            '#b91c1c', arms_up=True, label='Defender')
        self.defender_fig.update(self.defender_x,
                                 self.defender_reach / self.REACH_RATIO,
                                 self.defender_reach)

    # --------------------------------------------------------
    def _refresh_static(self):
        """Redraw everything that depends on the slider values."""
        # defender geometry
        self.defender_patch.set_x(self.defender_x - 0.22)
        self.defender_patch.set_height(self.defender_reach)
        self.defender_line.set_data(
            [self.defender_x - 0.34, self.defender_x + 0.34],
            [self.defender_reach] * 2)
        self.reach_tag.set_position((self.defender_x + 0.40,
                                     self.defender_reach))
        self.reach_tag.set_text(f"reach {self.defender_reach:.2f} m")
        self.defender_fig.update(self.defender_x,
                                 self.defender_reach / self.REACH_RATIO,
                                 self.defender_reach)

        # shooter geometry (release point follows the shooter's height)
        self.shooter_fig.update(0.0, self.shooter_h, self.release_height)

        self.ghost.set_data(self.result['x'], self.result['y'])
        self._update_panel()
