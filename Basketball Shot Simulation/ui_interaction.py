import textwrap
from matplotlib.widgets import Slider, Button

COL_X = (0.085, 0.395, 0.705)
ROW_Y = (0.158, 0.110, 0.062)
SL_W = 0.145
SL_H = 0.026


class UIPanelsMixin:

    def _setup_info_panel(self):
        ax = self.ax_info
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('#ffffff')
        for s in ax.spines.values():
            s.set_color('#cbd5e1')

        self.banner = ax.text(0.5, 0.975, '', ha='center', va='top',
                              fontsize=12.5, fontweight='bold', color='white',
                              bbox=dict(boxstyle='round,pad=0.45',
                                        facecolor='#16a34a',
                                        edgecolor='none'))

        ax.text(0.030, 0.760, 'SHOT INPUTS', fontsize=9, fontweight='bold',
                color='#6b7280')
        self.txt_inputs = ax.text(0.030, 0.700, '', fontsize=9.5, va='top',
                                  family='monospace', color='#111827',
                                  linespacing=1.55)

        ax.text(0.320, 0.760, 'FLIGHT DIAGNOSTICS', fontsize=9,
                fontweight='bold', color='#6b7280')
        self.txt_stats = ax.text(0.320, 0.700, '', fontsize=9.5, va='top',
                                 family='monospace', color='#111827',
                                 linespacing=1.55)

        ax.text(0.640, 0.760, 'LEGEND', fontsize=9, fontweight='bold',
                color='#6b7280')
        items = [('#ef4444', 'solid', 'flown path'),
                 ('#94a3b8', 'dashed', 'full predicted path'),
                 ('#b91c1c', 'dashed', 'defender reach cylinder'),
                 ('#f97316', 'solid', 'ball (size 7)')]
        y = 0.690
        for colour, style, text in items:
            ls = '-' if style == 'solid' else (0, (4, 3))
            ax.plot([0.650, 0.710], [y, y], color=colour, lw=2.4, ls=ls,
                    transform=ax.transAxes, clip_on=False)
            ax.text(0.725, y, text, fontsize=8.5, va='center',
                    color='#374151')
            y -= 0.075

        self.txt_note = ax.text(0.640, 0.330, '', fontsize=8.5, va='top',
                                color='#4b5563', linespacing=1.45)

    def _setup_model_panel(self):
        ax = self.ax_model
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_autoscale_on(False)
        ax.set_facecolor('#ffffff')
        for sp in ax.spines.values():
            sp.set_color('#cbd5e1')

        ax.text(0.5, 0.975, 'MODEL  (3D)', ha='center', va='top',
                fontsize=9.5, fontweight='bold', color='#6b7280')
        ax.text(0.5, 0.855,
                r'$m\ddot{\vec{r}}=-mg\hat{k}'
                r'-\frac{1}{2}\rho C_d A\,|\vec{v}_r|\,\vec{v}_r$',
                ha='center', va='center', fontsize=11, color='#111827')
        ax.text(0.5, 0.730,
                r'$\vec{v}_r=\dot{\vec{r}}-\vec{v}_{wind}$',
                ha='center', va='center', fontsize=10, color='#111827')
        ax.text(0.5, 0.625,
                r'$\vec{s}=[\,x\;y\;z\;\dot{x}\;\dot{y}\;\dot{z}\,]^{T}$',
                ha='center', va='center', fontsize=10, color='#111827')
        ax.text(0.5, 0.520,
                r'$\vec{v}_0=v_0\,(\cos\alpha\cos\beta,\;'
                r'\cos\alpha\sin\beta,\;\sin\alpha)$',
                ha='center', va='center', fontsize=8.5, color='#374151')

        P = self.P
        ax.text(0.06, 0.445, 'CONSTANTS', fontsize=9, fontweight='bold',
                color='#6b7280')
        ax.text(0.06, 0.375,
                f"m    = {P.m:.3f} kg\n"
                f"R    = {P.R:.4f} m\n"
                f"Cd   = {P.Cd:.2f}\n"
                f"rho  = {P.rho:.3f}\n"
                f"g    = {P.g:.2f}",
                fontsize=9, va='top', family='monospace', color='#111827',
                linespacing=1.55)
        ax.text(0.54, 0.375,
                f"rim  = {P.hoop_height:.3f} m\n"
                f"Rhoop= {P.hoop_radius:.4f} m\n"
                f"board= {P.backboard_x:.3f} m\n"
                f"width= {P.backboard_width:.3f} m\n"
                f"Rdef = {P.defender_radius:.2f} m",
                fontsize=9, va='top', family='monospace', color='#111827',
                linespacing=1.55)

    def _create_sliders(self):
        kw = dict(track_color='#e5e7eb')

        def axes_at(col, row):
            return self.fig.add_axes([COL_X[col], ROW_Y[row], SL_W, SL_H])

        self.sl_v = Slider(axes_at(0, 0), 'speed (m/s)', 5.0, 12.0,
                           valinit=self.v0, valstep=0.05, color='#f97316',
                           valfmt='%.2f', **kw)
        self.sl_a = Slider(axes_at(0, 1), 'pitch α (deg)', 25, 75,
                           valinit=self.pitch, valstep=0.5, color='#f97316',
                           valfmt='%.1f', **kw)
        self.sl_b = Slider(axes_at(0, 2), 'yaw β (deg)', -20, 20,
                           valinit=self.yaw, valstep=0.25, color='#a855f7',
                           valfmt='%+.2f', **kw)

        self.sl_wx = Slider(axes_at(1, 0), 'wind x (m/s)', -6, 6,
                            valinit=self.wind_x, valstep=0.2, color='#0ea5e9',
                            valfmt='%+.1f', **kw)
        self.sl_wy = Slider(axes_at(1, 1), 'wind y (m/s)', -6, 6,
                            valinit=self.wind_y, valstep=0.2, color='#0ea5e9',
                            valfmt='%+.1f', **kw)
        self.sl_sh = Slider(axes_at(1, 2), 'shooter h (m)', 1.50, 2.20,
                            valinit=self.shooter_h, valstep=0.01,
                            color='#334155', valfmt='%.2f', **kw)

        self.sl_dx = Slider(axes_at(2, 0), 'defender X (m)', 0.8, 4.0,
                            valinit=self.defender_x, valstep=0.1,
                            color='#dc2626', valfmt='%.1f', **kw)
        self.sl_dy = Slider(axes_at(2, 1), 'defender Y (m)', -1.5, 1.5,
                            valinit=self.defender_y, valstep=0.05,
                            color='#dc2626', valfmt='%+.2f', **kw)
        self.sl_dh = Slider(axes_at(2, 2), 'defender reach (m)', 1.80, 3.20,
                            valinit=self.defender_reach, valstep=0.05,
                            color='#dc2626', valfmt='%.2f', **kw)

        for s in self.sliders:
            s.label.set_fontsize(8.5)
            s.label.set_ha('right')
            s.valtext.set_fontsize(8.5)
            s.valtext.set_fontweight('bold')
            s.on_changed(self._on_change)

    @property
    def sliders(self):
        return (self.sl_v, self.sl_a, self.sl_b, self.sl_wx, self.sl_wy,
                self.sl_sh, self.sl_dx, self.sl_dy, self.sl_dh)

    def _create_buttons(self):
        self.btn_replay = Button(
            self.fig.add_axes([0.300, 0.008, 0.100, 0.038]),
            '▶  Replay', color='#22c55e', hovercolor='#16a34a')
        self.btn_replay.on_clicked(lambda e: self.start_animation())

        self.btn_opt = Button(
            self.fig.add_axes([0.420, 0.008, 0.140, 0.038]),
            '⚡ Auto-Optimize', color='#3b82f6', hovercolor='#2563eb')
        self.btn_opt.on_clicked(lambda e: self._auto_optimize())

        self.btn_reset = Button(
            self.fig.add_axes([0.580, 0.008, 0.100, 0.038]),
            '↺  Reset', color='#e5e7eb', hovercolor='#cbd5e1')
        self.btn_reset.on_clicked(lambda e: self._reset())

        for b in (self.btn_replay, self.btn_opt, self.btn_reset):
            b.label.set_fontsize(9.5)
            b.label.set_fontweight('bold')
        self.btn_replay.label.set_color('white')
        self.btn_opt.label.set_color('white')

    def _on_change(self, _val):
        if self._muted:
            return
        self.v0 = self.sl_v.val
        self.pitch = self.sl_a.val
        self.yaw = self.sl_b.val
        self.wind_x = self.sl_wx.val
        self.wind_y = self.sl_wy.val
        self.shooter_h = self.sl_sh.val
        self.defender_x = self.sl_dx.val
        self.defender_y = self.sl_dy.val
        self.defender_reach = self.sl_dh.val
        self._resimulate()

    def _reset(self):
        self._muted = True
        try:
            for s in self.sliders:
                s.reset()
        finally:
            self._muted = False
        self._on_change(None)

    def _update_panel(self):
        r = self.result
        ok = r['success']
        self.banner.set_text(f"{r['symbol']}  {r['status']}")
        self.banner.get_bbox_patch().set_facecolor(
            '#16a34a' if ok else '#dc2626')

        self.txt_inputs.set_text(
            f"speed      {self.v0:6.2f} m/s\n"
            f"pitch a    {self.pitch:6.1f} deg\n"
            f"yaw b      {self.yaw:+6.2f} deg\n"
            f"wind x     {self.wind_x:+6.1f} m/s\n"
            f"wind y     {self.wind_y:+6.1f} m/s\n"
            f"shooter h  {self.shooter_h:6.2f} m\n"
            f"release h  {self.release_height:6.2f} m\n"
            f"defender X {self.defender_x:6.1f} m\n"
            f"defender Y {self.defender_y:+6.2f} m\n"
            f"def reach  {self.defender_reach:6.2f} m")

        entry = ('   n/a' if r['entry_angle'] is None
                 else f"{r['entry_angle']:6.1f} deg")
        lat = ('   n/a' if r['lateral_error'] is None
               else f"{r['lateral_error']:+6.3f} m")
        lon = ('   n/a' if r['aim_error'] is None
               else f"{r['aim_error']:+6.3f} m")
        vgap = ('   n/a' if r['defender_clearance'] is None
                else f"{r['defender_clearance']:+6.2f} m")
        hgap = ('   n/a' if r['defender_lateral'] is None
                else f"{r['defender_lateral']:+6.2f} m")

        self.txt_stats.set_text(
            f"apex       {r['apex']:6.2f} m\n"
            f"flight     {r['flight_time']:6.2f} s\n"
            f"entry     {entry}\n"
            f"long miss {lon}\n"
            f"side miss {lat}\n"
            f"drift      {r['max_drift']:6.2f} m\n"
            f"over def  {vgap}\n"
            f"past def  {hgap}\n"
            f"cleared    {'YES' if r['clears_defender'] else 'NO':>6}")

        if not r['clears_defender']:
            note = ("Blocked: the ball passes the defender both below the "
                    "reach line and inside their lateral radius. Raise the "
                    "arc, or aim around them with yaw.")
        elif not r['made']:
            if r['lateral_error'] is not None and abs(r['lateral_error']) > 0.05:
                note = ("Clears the defender but misses sideways. Trim the "
                        "yaw angle — a crosswind needs aim into the wind.")
            else:
                note = ("Clears the defender but misses long or short. Try "
                        "Auto-Optimize for the most forgiving shot.")
        else:
            note = ("Scoring shot. Entry angles near 45-50 deg give the "
                    "largest margin for error (Silverberg et al., 2003).")
        self.txt_note.set_text(textwrap.fill(note, 38))
