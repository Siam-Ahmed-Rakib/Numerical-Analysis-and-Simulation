import textwrap
from matplotlib.widgets import Slider, Button


class UIPanelsMixin:


    def _setup_info_panel(self):
        ax = self.ax_info
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_facecolor('#ffffff')
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

        self.banner = ax.text(0.5, 0.965, '', ha='center', va='top',
                              fontsize=13, fontweight='bold', color='white',
                              bbox=dict(boxstyle='round,pad=0.45',
                                        facecolor='#16a34a', edgecolor='none'))
        ax.text(0.07, 0.855, 'SHOT INPUTS', fontsize=10, fontweight='bold',
                color='#6b7280')
        self.txt_inputs = ax.text(0.07, 0.815, '', fontsize=10.5, va='top',
                                  family='monospace', color='#111827',
                                  linespacing=1.6)
        ax.text(0.07, 0.500, 'FLIGHT DIAGNOSTICS', fontsize=10,
                fontweight='bold', color='#6b7280')
        self.txt_stats = ax.text(0.07, 0.460, '', fontsize=10.5, va='top',
                                 family='monospace', color='#111827',
                                 linespacing=1.6)
        self.txt_note = ax.text(0.07, 0.04, '', fontsize=9, va='bottom',
                                color='#4b5563', linespacing=1.5)

    def _setup_model_panel(self):
        ax = self.ax_model
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)
        ax.set_facecolor('#ffffff')
        for sp in ax.spines.values():
            sp.set_color('#cbd5e1')

        ax.text(0.5, 0.965, 'MODEL', ha='center', va='top', fontsize=10,
                fontweight='bold', color='#6b7280')
        ax.text(0.5, 0.905,
                r'$m\frac{d\vec{v}}{dt}=-mg\hat{\jmath}'
                r'-\frac{1}{2}\rho C_d A |\vec{v}_r|\vec{v}_r$',
                ha='center', va='top', fontsize=11, color='#111827')
        ax.text(0.5, 0.815, r'$\vec{v}_r=\vec{v}-\vec{v}_{wind}$',
                ha='center', va='top', fontsize=10, color='#111827')

        P = self.P
        ax.text(0.07, 0.735, 'CONSTANTS', fontsize=10, fontweight='bold',
                color='#6b7280')
        ax.text(0.07, 0.695,
                f"m   = {P.m:.3f} kg\n"
                f"R   = {P.R:.4f} m\n"
                f"Cd  = {P.Cd:.2f}\n"
                f"rho = {P.rho:.3f} kg/m3\n"
                f"g   = {P.g:.2f} m/s2\n"
                f"rim = {P.hoop_height:.3f} m\n"
                f"d   = {P.backboard_x:.3f} m",
                fontsize=10, va='top', family='monospace', color='#111827',
                linespacing=1.6)

        ax.text(0.07, 0.385, 'LEGEND', fontsize=10, fontweight='bold',
                color='#6b7280')
        items = [('#ef4444', 'solid',  'flown path'),
                 ('#94a3b8', 'dashed', 'full predicted path'),
                 ('#b91c1c', 'dashed', 'defender reach line'),
                 ('#f97316', 'solid',  'ball (size 7)')]
        y = 0.320
        for colour, style, text in items:
            ls = '-' if style == 'solid' else (0, (4, 3))
            ax.plot([0.09, 0.24], [y, y], color=colour, lw=2.6, ls=ls,
                    transform=ax.transAxes, clip_on=False)
            ax.text(0.28, y, text, fontsize=9, va='center', color='#374151')
            y -= 0.055

        ax.text(0.07, 0.020,
                textwrap.fill("Success = the ball clears the reach line and "
                              "drops through the rim.", 32),
                fontsize=9, va='bottom', color='#4b5563', linespacing=1.5)

    def _create_sliders(self):
        sl_kw = dict(track_color='#e5e7eb')
        L, W, H = 0.115, 0.250, 0.022
        R = 0.620

        ax_v = self.fig.add_axes([L, 0.160, W, H])
        self.sl_v = Slider(ax_v, 'Release speed\n(m/s)', 5.0, 12.0,
                           valinit=self.v0, valstep=0.05, color='#f97316',
                           valfmt='%.2f', **sl_kw)

        ax_a = self.fig.add_axes([L, 0.116, W, H])
        self.sl_a = Slider(ax_a, 'Launch angle\n(deg)', 25, 75,
                           valinit=self.theta, valstep=0.5, color='#f97316',
                           valfmt='%.1f', **sl_kw)

        ax_w = self.fig.add_axes([L, 0.072, W, H])
        self.sl_w = Slider(ax_w, 'Wind\n(m/s)', -6, 6, valinit=self.wind_x,
                           valstep=0.2, color='#0ea5e9', valfmt='%+.1f',
                           **sl_kw)

        ax_sh = self.fig.add_axes([L, 0.028, W, H])
        self.sl_sh = Slider(ax_sh, 'Shooter height\n(m)', 1.50, 2.20,
                            valinit=self.shooter_h, valstep=0.01,
                            color='#334155', valfmt='%.2f', **sl_kw)

        ax_dx = self.fig.add_axes([R, 0.160, W, H])
        self.sl_dx = Slider(ax_dx, 'Defender X\n(m)', 0.8, 4.0,
                            valinit=self.defender_x, valstep=0.1,
                            color='#dc2626', valfmt='%.1f', **sl_kw)

        ax_dh = self.fig.add_axes([R, 0.116, W, H])
        self.sl_dh = Slider(ax_dh, 'Defender reach\n(m)', 1.80, 3.20,
                            valinit=self.defender_reach, valstep=0.05,
                            color='#dc2626', valfmt='%.2f', **sl_kw)

        for s in (self.sl_v, self.sl_a, self.sl_w, self.sl_sh,
                  self.sl_dx, self.sl_dh):
            s.label.set_fontsize(9.5)
            s.label.set_ha('right')
            s.valtext.set_fontsize(10)
            s.valtext.set_fontweight('bold')
            s.on_changed(self._on_change)

    def _create_buttons(self):
        R0 = 0.600
        self.btn_replay = Button(self.fig.add_axes([R0, 0.045, 0.105, 0.042]),
                                 '\u25b6  Replay', color='#22c55e',
                                 hovercolor='#16a34a')
        self.btn_replay.on_clicked(lambda e: self.start_animation())

        self.btn_opt = Button(self.fig.add_axes([R0 + 0.120, 0.045,
                                                 0.140, 0.042]),
                              '\u26a1 Auto-Optimize', color='#3b82f6',
                              hovercolor='#2563eb')
        self.btn_opt.on_clicked(lambda e: self._auto_optimize())

        self.btn_reset = Button(self.fig.add_axes([R0 + 0.275, 0.045,
                                                   0.100, 0.042]),
                                '\u21ba  Reset', color='#e5e7eb',
                                hovercolor='#cbd5e1')
        self.btn_reset.on_clicked(lambda e: self._reset())

        for b in (self.btn_replay, self.btn_opt, self.btn_reset):
            b.label.set_fontsize(11)
            b.label.set_fontweight('bold')
        self.btn_replay.label.set_color('white')
        self.btn_opt.label.set_color('white')

    def _on_change(self, _val):
        self.v0 = self.sl_v.val
        self.theta = self.sl_a.val
        self.wind_x = self.sl_w.val
        self.shooter_h = self.sl_sh.val
        self.defender_x = self.sl_dx.val
        self.defender_reach = self.sl_dh.val
        self._resimulate()

    def _reset(self):
        self.sl_w.reset(); self.sl_sh.reset(); self.sl_dx.reset()
        self.sl_dh.reset(); self.sl_a.reset(); self.sl_v.reset()

    def _update_panel(self):
        r = self.result
        ok = r['success']
        self.banner.set_text(f"{r['symbol']}  {r['status']}")
        self.banner.get_bbox_patch().set_facecolor(
            '#16a34a' if ok else '#dc2626')

        self.txt_inputs.set_text(
            f"angle        {self.theta:6.1f} deg\n"
            f"speed        {self.v0:6.2f} m/s\n"
            f"wind         {self.wind_x:+6.1f} m/s\n"
            f"shooter h    {self.shooter_h:6.2f} m\n"
            f"release h    {self.release_height:6.2f} m\n"
            f"defender X   {self.defender_x:6.1f} m\n"
            f"defender rch {self.defender_reach:6.2f} m")

        clr = r['defender_clearance']
        entry = ('  n/a' if r['entry_angle'] is None
                 else f"{r['entry_angle']:6.1f} deg")
        self.txt_stats.set_text(
            f"apex         {r['apex']:6.2f} m\n"
            f"flight time  {r['flight_time']:6.2f} s\n"
            f"entry angle {entry}\n"
            f"clearance    {clr:+6.2f} m\n"
            f"over defender {'YES' if r['clears_defender'] else 'NO'}")

        if not r['clears_defender']:
            note = ("The ball passes the defender below the reach line — "
                    "raise the launch angle or the release speed.")
        elif not r['made']:
            note = ("Clears the defender but misses the rim. Try "
                    "Auto-Optimize for the softest scoring shot.")
        else:
            note = ("Scoring shot. Entry angles near 45-50 deg give the "
                    "largest margin for error (Silverberg et al., 2003).")
        self.txt_note.set_text(textwrap.fill(note, 30))
