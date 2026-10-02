import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

POSTS = 8


def _quad(ax, corners, **kw):
    poly = Poly3DCollection([corners], **kw)
    ax.add_collection3d(poly)
    return poly


class StickFigure3D:

    def __init__(self, ax, x, y, height, color, arms_up=False, label=''):
        self.ax = ax
        self.color = color
        self.arms_up = arms_up
        self.name = label

        self.head, = ax.plot([], [], [], ls='none', marker='o', ms=8,
                             color=color)
        self.body, = ax.plot([], [], [], color=color, lw=3,
                             solid_capstyle='round')
        self.leg_l, = ax.plot([], [], [], color=color, lw=3,
                              solid_capstyle='round')
        self.leg_r, = ax.plot([], [], [], color=color, lw=3,
                              solid_capstyle='round')
        self.arm_l, = ax.plot([], [], [], color=color, lw=3,
                              solid_capstyle='round')
        self.arm_r, = ax.plot([], [], [], color=color, lw=3,
                              solid_capstyle='round')
        self.label = ax.text(x, y, height, '', color=color, fontsize=8,
                             fontweight='bold', ha='center', va='bottom')
        self.update(x, y, height)

    @staticmethod
    def _segment(line, p, q):
        line.set_data([p[0], q[0]], [p[1], q[1]])
        line.set_3d_properties([p[2], q[2]])

    def update(self, x, y, height, top=None):
        h = height
        top = (h + 0.40) if top is None else top
        head_r = 0.105
        shoulder = h - 2 * head_r
        hip = h * 0.48

        self.head.set_data([x], [y])
        self.head.set_3d_properties([h - head_r])
        self._segment(self.body, (x, y, shoulder), (x, y, hip))
        self._segment(self.leg_l, (x, y, hip), (x, y - 0.17, 0.0))
        self._segment(self.leg_r, (x, y, hip), (x, y + 0.17, 0.0))

        if self.arms_up:
            self._segment(self.arm_l, (x, y, shoulder), (x, y - 0.24, top))
            self._segment(self.arm_r, (x, y, shoulder), (x, y + 0.24, top))
        else:
            self._segment(self.arm_l, (x, y, shoulder),
                          (x + 0.06, y - 0.13, top - 0.10))
            self._segment(self.arm_r, (x, y, shoulder),
                          (x + 0.16, y + 0.06, top))

        self.label.set_position_3d((x, y, top + 0.16))
        self.label.set_text(f"{self.name} {h:.2f} m")


class SceneMixin:

    def _setup_court(self):
        ax, P = self.ax, self.P
        x_hi = P.backboard_x + 0.85
        ax.set_xlim(-1.0, x_hi)
        ax.set_ylim(-2.3, 2.3)
        ax.set_zlim(0.0, 5.2)
        ax.set_box_aspect((x_hi + 1.0, 4.6, 5.2))
        ax.view_init(elev=14, azim=-64)
        ax.set_xlabel('toward basket (m)', fontsize=9, labelpad=2)
        ax.set_ylabel('lateral (m)', fontsize=9, labelpad=2)
        ax.set_zlabel('height (m)', fontsize=9, labelpad=2)
        ax.tick_params(labelsize=7.5, pad=1)
        ax.set_facecolor('#ffffff')
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor('#f8fafc')
            pane.pane.set_edgecolor('#e2e8f0')
            pane._axinfo['grid'].update(color='#e2e8f0', linewidth=0.7)

        lane = P.lane_half_width
        _quad(ax, [(-1.0, -lane, 0.0), (x_hi, -lane, 0.0),
                   (x_hi, lane, 0.0), (-1.0, lane, 0.0)],
              facecolor='#d8b48a', edgecolor='none', alpha=0.55, zsort='min')
        for sign in (-1.0, 1.0):
            ax.plot([0.0, x_hi], [sign * lane, sign * lane], [0.0, 0.0],
                    color='#a1662f', lw=1.4)
        ax.plot([0.0, 0.0], [-lane, lane], [0.0, 0.0],
                color='#a1662f', lw=1.4)

        h = P.hoop_height
        bw = P.board_half_width
        _quad(ax, [(P.backboard_x, -bw, P.board_bottom),
                   (P.backboard_x, bw, P.board_bottom),
                   (P.backboard_x, bw, P.board_top),
                   (P.backboard_x, -bw, P.board_top)],
              facecolor='#dbeafe', edgecolor='#1e3a8a', lw=1.8, alpha=0.75)
        ax.plot([P.backboard_x] * 5,
                [-0.30, 0.30, 0.30, -0.30, -0.30],
                [h + 0.05, h + 0.05, h + 0.50, h + 0.50, h + 0.05],
                color='#1e3a8a', lw=2.0)

        pole_x = P.backboard_x + 0.55
        ax.plot([pole_x, pole_x], [0.0, 0.0], [0.0, h + 0.55],
                color='#64748b', lw=5, solid_capstyle='round')
        ax.plot([P.backboard_x, pole_x], [0.0, 0.0], [h + 0.55, h + 0.55],
                color='#64748b', lw=4, solid_capstyle='round')

        th = np.linspace(0.0, 2.0 * np.pi, 80)
        rim_x = P.hoop_x + P.hoop_radius * np.cos(th)
        rim_y = P.hoop_y + P.hoop_radius * np.sin(th)
        ax.plot(rim_x, rim_y, [h] * len(th), color='#ea580c', lw=3.2)
        ax.plot([P.backboard_x, P.hoop_x + P.hoop_radius], [0.0, 0.0],
                [h, h], color='#9a3412', lw=2.5)

        net_bottom = h - 0.38
        for k in range(12):
            a = 2.0 * np.pi * k / 12.0
            ax.plot([P.hoop_x + P.hoop_radius * np.cos(a),
                     P.hoop_x + 0.55 * P.hoop_radius * np.cos(a)],
                    [P.hoop_y + P.hoop_radius * np.sin(a),
                     P.hoop_y + 0.55 * P.hoop_radius * np.sin(a)],
                    [h, net_bottom], color='#9ca3af', lw=0.8)
        for frac in (0.4, 0.75, 1.0):
            rr = P.hoop_radius * (1.0 - 0.45 * frac)
            ax.plot(P.hoop_x + rr * np.cos(th), P.hoop_y + rr * np.sin(th),
                    [h - frac * 0.36] * len(th), color='#9ca3af', lw=0.8)

        self.def_top, = ax.plot([], [], [], color='#b91c1c', lw=2.0,
                                ls=(0, (4, 3)))
        self.def_base, = ax.plot([], [], [], color='#b91c1c', lw=1.2,
                                 alpha=0.6)
        self.def_posts = [ax.plot([], [], [], color='#ef4444', lw=1.0,
                                  alpha=0.65)[0] for _ in range(POSTS)]
        self.reach_tag = ax.text(0.0, 0.0, 0.0, '', color='#b91c1c',
                                 fontsize=8, ha='left', va='center')

        self.shooter_fig = StickFigure3D(ax, P.release_x, P.release_y,
                                         self.shooter_h, '#1f2937',
                                         arms_up=False, label='Shooter')
        self.defender_fig = StickFigure3D(
            ax, self.defender_x, self.defender_y,
            self.defender_reach / self.REACH_RATIO, '#b91c1c',
            arms_up=True, label='Defender')

    def _setup_plan(self):
        ax, P = self.ax_plan, self.P
        x_hi = P.backboard_x + 0.7
        lane = P.lane_half_width
        ax.set_facecolor('#ffffff')
        ax.set_xlim(-1.0, x_hi)
        ax.set_ylim(-2.35, 2.35)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, ls=':')
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#cbd5e1')
        ax.set_title('TOP VIEW  (lateral aim)', fontsize=9,
                     fontweight='bold', color='#6b7280', pad=4)
        ax.set_xlabel('x (m)', fontsize=8, labelpad=1)
        ax.set_ylabel('y (m)', fontsize=8, labelpad=1)

        for sign in (-1.0, 1.0):
            ax.axhline(sign * lane, color='#e2e8f0', lw=1.2)
        ax.plot([P.backboard_x] * 2, [-P.board_half_width,
                                      P.board_half_width],
                color='#1e3a8a', lw=3)
        th = np.linspace(0.0, 2.0 * np.pi, 80)
        ax.plot(P.hoop_x + P.hoop_radius * np.cos(th),
                P.hoop_y + P.hoop_radius * np.sin(th),
                color='#ea580c', lw=2.2)
        ax.plot(P.hoop_x + (P.hoop_radius - P.R) * np.cos(th),
                P.hoop_y + (P.hoop_radius - P.R) * np.sin(th),
                color='#fdba74', lw=1.0, ls=(0, (3, 2)))

        self.plan_ghost, = ax.plot([], [], color='#94a3b8', lw=1.2,
                                   ls=(0, (5, 4)))
        self.plan_trail, = ax.plot([], [], color='#ef4444', lw=2.0)
        self.plan_def = Circle((self.defender_x, self.defender_y),
                               P.defender_radius, facecolor='#ef4444',
                               alpha=0.25, edgecolor='#b91c1c', lw=1.4)
        ax.add_patch(self.plan_def)
        ax.plot([P.release_x], [P.release_y], ls='none', marker='o', ms=6,
                color='#1f2937')
        self.plan_ball, = ax.plot([], [], ls='none', marker='o', ms=7,
                                  color='#f97316', mec='#7c2d12', mew=1.2)

    def _refresh_static(self):
        P = self.P
        th = np.linspace(0.0, 2.0 * np.pi, 60)
        cx = self.defender_x + P.defender_radius * np.cos(th)
        cy = self.defender_y + P.defender_radius * np.sin(th)
        reach = self.defender_reach

        self.def_top.set_data(cx, cy)
        self.def_top.set_3d_properties([reach] * len(th))
        self.def_base.set_data(cx, cy)
        self.def_base.set_3d_properties([0.0] * len(th))
        for k, post in enumerate(self.def_posts):
            a = 2.0 * np.pi * k / POSTS
            px = self.defender_x + P.defender_radius * np.cos(a)
            py = self.defender_y + P.defender_radius * np.sin(a)
            post.set_data([px, px], [py, py])
            post.set_3d_properties([0.0, reach])

        self.reach_tag.set_position_3d(
            (self.defender_x, self.defender_y + P.defender_radius + 0.22,
             reach - 0.16))
        self.reach_tag.set_text(f"reach {reach:.2f} m")

        self.defender_fig.update(self.defender_x, self.defender_y,
                                 reach / self.REACH_RATIO, reach)
        self.shooter_fig.update(P.release_x, P.release_y, self.shooter_h,
                                self.release_height)

        self.ghost.set_data(self.result['x'], self.result['y'])
        self.ghost.set_3d_properties(self.result['z'])
        self.plan_ghost.set_data(self.result['x'], self.result['y'])
        self.plan_def.center = (self.defender_x, self.defender_y)
        self.plan_def.set_radius(P.defender_radius)

        self._update_panel()
