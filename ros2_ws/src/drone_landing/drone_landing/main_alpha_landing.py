"""
main_crazysim.py  —  LOS + FSM (percorso a L) + logging CSV + plot diagnostico.
"""

import time
import math
import csv
from datetime import datetime
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

import rclpy
from rclpy.node import Node

try:
    from drone_landing.controller_deploy import HybridController   # dentro package ROS
except ImportError:
    from controller_deploy import HybridController                 # standalone

URI = uri_helper.uri_from_env(default='udp://127.0.0.1:19850')
DT = 0.02
G = 9.81

# ---- calibrazione spinta -----------------------------------------------------
#HOVER_CMD = 32000
HOVER_CMD = 32748
MASS = 0.0379
#MASS = 0.029                      
HOVER_FORCE = MASS * G


def force_to_cmd(force_N):
    return int(np.clip(HOVER_CMD * force_N / HOVER_FORCE, 10001, 60000))


def rad2deg(x):
    return x * 180.0 / math.pi


# ---- missione (dinamica, percorso a L con piattaforma mobile) ----------------
TARGET_XY = np.array([3.5, 3.5])
TARGET_VEL = np.array([0.0, 0.0, 0.0])  # Velocità del quadrupede/piattaforma
Z_CRUISE = 1.8
Z_LAND = 0.1
Z_HOLD = 1.8
ALPHA_CONE = 1.0
LOS_DELTA = 0.3
A_XY = 0.17
R_BASE_CONE = 0.3   
IDLE, RISING, NAV, HOLD, LANDING = 0, 1, 2, 3, 4


def trapz_profile(tau, L, V, t_acc):
    """Profilo posizione/velocita' trapezoidale (rampa ad accelerazione limitata,
    invece dello scalino di velocita' di s_lin = min(L, V*tau)).
    tau: tempo dall'ingresso nel segmento. L: lunghezza segmento. V: velocita' di
    crociera. t_acc: tempo di accelerazione/decelerazione (self.RAMP_T).
    Ritorna (s, v) ascissa curvilinea e velocita' scalare lungo u = seg/L."""
    if L < 1e-9:
        return 0.0, 0.0
    if tau <= 0.0:
        return 0.0, 0.0
    a = V / t_acc if t_acc > 1e-9 else float("inf")
    d_acc = 0.5 * V * t_acc
    if 2 * d_acc >= L or a == float("inf"):
        # segmento troppo corto per raggiungere V: profilo triangolare
        t_pk = math.sqrt(L / a) if a > 0 else 0.0
        T = 2 * t_pk
        if tau < t_pk:
            return 0.5 * a * tau**2, a * tau
        elif tau < T:
            td = T - tau
            return L - 0.5 * a * td**2, a * td
        else:
            return L, 0.0
    else:
        t_cruise = (L - 2 * d_acc) / V
        T = 2 * t_acc + t_cruise
        if tau < t_acc:
            return 0.5 * a * tau**2, a * tau
        elif tau < t_acc + t_cruise:
            return d_acc + V * (tau - t_acc), V
        elif tau < T:
            td = T - tau
            return L - 0.5 * a * td**2, a * td
        else:
            return L, 0.0


def LOS_wp(p_actual, p_start, p_end, delta, stop_delta):
    p_actual = np.array(p_actual); p_start = np.array(p_start); p_end = np.array(p_end)
    path_vector = p_end - p_start
    path_length = np.linalg.norm(path_vector)
    if path_length < 1e-6:
        return p_end, True
    u = path_vector / path_length
    s = np.dot(p_actual - p_start, u)
    reached_end = False
    if (s + delta) <= 0:
        p_LOS = p_start
    elif (s + delta) >= path_length:
        p_LOS = p_end; reached_end = True
    else:
        p_LOS = p_start + (s + delta) * u
    return p_LOS, reached_end


def build_logconf():
    lg = LogConfig(name="state", period_in_ms=int(DT * 1000))
    for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
              "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
        lg.add_variable(v, "float")
    return lg


def reset_estimator(cf):
    cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.5)


def save_and_plot(rows):
    if not rows:
        print("nessun dato da plottare"); return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"last_run_plots/flight_{stamp}.csv"
    cols = ["t", "state", "mode", "x", "y", "z", "vx", "vy", "vz",
            "carrot_x", "carrot_y", "carrot_z", "force", "cmd", "az", "roll", "pitch", "solve_ms",
            "ref_vx", "ref_vy", "ref_vz",
            "target_x", "target_y", "target_z", "target_vx", "target_vy", "target_vz"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"CSV salvato: {csv_path}  ({len(rows)} righe)")

    a = {c: np.array([r[c] for r in rows], dtype=float) for c in cols if c != "state"}
    mode = a["mode"]; t = a["t"]
    state_str = np.array([r["state"] for r in rows])

    dt_wall = np.diff(t)
    if len(dt_wall):
        print(f"loop: dt medio {np.mean(dt_wall)*1000:.1f} ms "
              f"(target {DT*1000:.0f} ms), max {np.max(dt_wall)*1000:.1f} ms")
    solve = a["solve_ms"][a["solve_ms"] > 0]
    if len(solve):
        print(f"solve MPC: medio {np.mean(solve):.1f} ms, max {np.max(solve):.1f} ms, "
              f"95pct {np.percentile(solve,95):.1f} ms")

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib non disponibile:", e); return

    # ax_est/ay_est: accelerazione orizzontale "fisica", coerente con az = F/m - G,
    # stimata dagli angoli comandati (roll/pitch in radianti). Convenzione coerente
    # con send_setpoint(rad2deg(roll), rad2deg(pitch), 0, cmd): pitch positivo ->
    # accelerazione in +x, roll positivo -> accelerazione in -y. Verifica il segno
    # confrontando l'andamento con vx/vy in un tratto di moto orizzontale netto.
    ax_est = G * np.tan(a["pitch"])
    ay_est = -G * np.tan(a["roll"])

    fig, ax = plt.subplots(8, 1, sharex=True, figsize=(11, 13))

    def shade_mpc(axis):
        on = mode > 0.5
        if not on.any():
            return
        start = None
        for i in range(len(on)):
            if on[i] and start is None:
                start = t[i]
            if (not on[i] or i == len(on) - 1) and start is not None:
                axis.axvspan(start, t[i], color="orange", alpha=0.12); start = None

    ax[0].plot(t, a["z"], label="z", lw=1.5)
    ax[0].plot(t, a["carrot_z"], label="carrot_z", lw=1, ls="--")
    ax[0].axhline(Z_LAND, color="k", lw=0.8, ls=":", label="target land")
    ax[0].set_ylabel("z [m]"); ax[0].legend(loc="upper right"); shade_mpc(ax[0])

    ax[1].plot(t, a["vz"], color="tab:green"); ax[1].axhline(0, color="k", lw=0.6)
    ax[1].plot(t, a["ref_vz"], color="tab:green", lw=1, ls="--", label="ref vz")
    ax[1].set_ylabel("vz [m/s]"); ax[1].legend(loc="upper right"); shade_mpc(ax[1])

    ax[2].plot(t, a["cmd"], color="tab:red")
    ax[2].axhline(HOVER_CMD, color="k", lw=0.8, ls=":", label="hover cmd")
    ax[2].axhline(60000, color="gray", lw=0.6, ls=":")
    ax[2].set_ylabel("thrust cmd"); ax[2].legend(loc="upper right"); shade_mpc(ax[2])

    ax[3].plot(t, a["az"], color="tab:purple"); ax[3].axhline(0, color="k", lw=0.6)
    ax[3].set_ylabel("az [m/s^2]"); shade_mpc(ax[3])

    ax[4].plot(t, a["x"], label="x", lw=1.5)
    ax[4].plot(t, a["carrot_x"], label="carrot_x", lw=1, ls="--")
    ax[4].plot(t, a["target_x"], color="k", lw=0.8, ls=":", label="target platform")
    ax[4].set_ylabel("x [m]"); ax[4].legend(loc="upper right"); shade_mpc(ax[4])

    ax[5].plot(t, a["y"], label="y", lw=1.5)
    ax[5].plot(t, a["carrot_y"], label="carrot_y", lw=1, ls="--")
    ax[5].plot(t, a["target_y"], color="k", lw=0.8, ls=":", label="target platform")
    ax[5].set_ylabel("y [m]"); ax[5].legend(loc="upper right"); shade_mpc(ax[5])

    ax[6].plot(t, a["vx"], label="vx", color="tab:blue", lw=1.3)
    ax[6].plot(t, a["vy"], label="vy", color="tab:orange", lw=1.3)
    ax[6].plot(t, a["ref_vx"], color="tab:blue", lw=1, ls="--", label="ref vx")
    ax[6].plot(t, a["ref_vy"], color="tab:orange", lw=1, ls="--", label="ref vy")
    ax[6].axhline(0, color="k", lw=0.6)
    ax[6].set_ylabel("v_xy [m/s]"); ax[6].legend(loc="upper right", ncol=2, fontsize=8); shade_mpc(ax[6])

    ax[7].plot(t, ax_est, label="ax (da pitch)", color="tab:blue", lw=1.3)
    ax[7].plot(t, ay_est, label="ay (da roll)", color="tab:orange", lw=1.3)
    ax[7].axhline(0, color="k", lw=0.6)
    ax[7].set_ylabel("a_xy [m/s^2]"); ax[7].set_xlabel("t [s]")
    ax[7].legend(loc="upper right"); shade_mpc(ax[7])

    trans_idx = np.where(state_str[:-1] != state_str[1:])[0]
    if len(t) > 0:
        ax[0].text(t[0], 1.05, f" {state_str[0].upper()}", transform=ax[0].get_xaxis_transform(),
                   fontsize=9, color="black", fontweight="bold", alpha=0.7)

    for idx in trans_idx:
        t_trans = t[idx + 1]
        new_state = state_str[idx + 1]
        for axi in ax:
            axi.axvline(t_trans, color="black", linestyle="--", lw=1.2, alpha=0.6)
        ax[0].text(t_trans, 1.05, f" {new_state.upper()}", transform=ax[0].get_xaxis_transform(),
                   fontsize=9, color="black", fontweight="bold", alpha=0.7)

    fig.suptitle("Volo CrazySim — Inseguimento Piattaforma Mobile", y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    
    png = f"last_run_plots/flight_{stamp}.png"; fig.savefig(png, dpi=110)
    print(f"plot salvato: {png}")
    plt.show()

    return stamp

def plot_advanced_diagnostics(rows, stamp=None, alpha_cone=1.0, z_cut=1.0, r_base=0.1, polytope_path="reachable_polytope.npz"):
    import numpy as np
    import os
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError:
        print("Matplotlib non disponibile per i plot avanzati.")
        return
 
    if not rows: return
    if stamp is None: stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("last_run_plots", exist_ok=True)
 
    t = np.array([r["t"] for r in rows])
    pos = np.array([[r["x"], r["y"], r["z"]] for r in rows]).T
    vel = np.array([[r["vx"], r["vy"], r["vz"]] for r in rows]).T
    los = np.array([[r["carrot_x"], r["carrot_y"], r["carrot_z"]] for r in rows]).T
    state_str = np.array([r["state"] for r in rows])
    
    # NOVITA: Errori dinamici tracciati lungo la piattaforma in movimento
    target_pos = np.array([[r["target_x"], r["target_y"], r["target_z"]] for r in rows]).T
    target_vel = np.array([[r["target_vx"], r["target_vy"], r["target_vz"]] for r in rows]).T

    landing_mask = (state_str == "landing")
    fp = target_pos[:, landing_mask]
    fv = target_vel[:, landing_mask]
 
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    fig2d = plt.figure(figsize=(10, 8))
    ax2d = fig2d.add_subplot(111)
 
    ax3d.plot(pos[0], pos[1], pos[2], label="Trajectory", color='b', linewidth=2)
    ax3d.plot(los[0], los[1], los[2], label="LOS Target", color='g', linestyle='--', linewidth=1.5)
    ax3d.plot(target_pos[0], target_pos[1], target_pos[2], label="Moving Platform", color='k', linestyle=':', linewidth=2)
    
    step_size = max(1, int(pos.shape[1] / 50))
    ax3d.quiver(pos[0, ::step_size], pos[1, ::step_size], pos[2, ::step_size],
                los[0, ::step_size] - pos[0, ::step_size],
                los[1, ::step_size] - pos[1, ::step_size],
                los[2, ::step_size] - pos[2, ::step_size],
                color='r', alpha=0.6, arrow_length_ratio=0.15, linewidth=1.5, label="Error Vector")
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
    ax3d.set_title('3D Tracking: Actual Position vs LOS Target')
    ax3d.legend()
    fig3d.savefig(f"last_run_plots/flight_{stamp}_3d_track.png", dpi=110)
 
    ax2d.plot(pos[0], pos[1], label="Trajectory", color='b', linewidth=2)
    ax2d.plot(los[0], los[1], label="LOS Target", color='g', linestyle='--', linewidth=1.5)
    ax2d.plot(target_pos[0], target_pos[1], label="Moving Platform", color='k', linestyle=':', linewidth=2)
    ax2d.quiver(pos[0, ::step_size], pos[1, ::step_size],
                los[0, ::step_size] - pos[0, ::step_size],
                los[1, ::step_size] - pos[1, ::step_size],
                angles='xy', scale_units='xy', scale=1, color='r', alpha=0.6, width=0.003, label="Error Vector")
    ax2d.set_xlabel('X [m]'); ax2d.set_ylabel('Y [m]')
    ax2d.set_title('2D Top-Down View: XY Tracking')
    ax2d.axis('equal'); ax2d.grid(True); ax2d.legend()
    fig2d.savefig(f"last_run_plots/flight_{stamp}_2d_track.png", dpi=110)
 
    try:
        d = np.load(polytope_path)
        H_ax, h_ax, H_vz, h_vz = d["H_ax"], d["h_ax"], d["H_vz"], d["h_vz"]
        V_ax, V_vz = d["V_ax"], d["V_vz"]
        
        # Errore relativo alla piattaforma mobile
        ep = pos[:, landing_mask] - fp
        ev = vel[:, landing_mask] - fv
        tt = t[landing_mask]
        Np = ep.shape[1]
 
        def inside(H, h, e, v): return np.all(H @ np.array([e, v]) <= h + 1e-6)
        in_all = np.array([inside(H_ax, h_ax, ep[0, k], ev[0, k]) and 
                           inside(H_ax, h_ax, ep[1, k], ev[1, k]) and 
                           inside(H_vz, h_vz, ep[2, k], ev[2, k]) for k in range(Np)])
        entry = int(np.argmax(in_all)) if in_all.any() else None
 
        panels = [("Asse X", ep[0], ev[0], V_ax, "e_x [m]", "e_vx [m/s]"),
                  ("Asse Y", ep[1], ev[1], V_ax, "e_y [m]", "e_vy [m/s]"),
                  ("Asse Z", ep[2], ev[2], V_vz, "e_z [m]", "e_vz [m/s]")]
        
        fig_poly, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        for ax, (name, e, v, V, xl, yl) in zip(axes, panels):
            Vc = np.vstack([V, V[0]])
            ax.fill(Vc[:, 0], Vc[:, 1], color="tab:green", alpha=0.10)
            ax.plot(Vc[:, 0], Vc[:, 1], color="tab:green", lw=1.6)
            pts = np.array([e, v]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="plasma", zorder=2)
            lc.set_array(tt[:-1]); lc.set_linewidth(2.0); ax.add_collection(lc)
            ax.scatter(e[0], v[0], c="k", s=55, zorder=4)
            if entry is not None:
                ax.scatter(e[entry], v[entry], marker="*", s=280, c="red", edgecolor="k", zorder=5)
            xs, ys = np.r_[Vc[:, 0], e], np.r_[Vc[:, 1], v]
            mx, my = 0.1*np.ptp(xs), 0.1*np.ptp(ys)
            ax.set_xlim(xs.min()-mx, xs.max()+mx); ax.set_ylim(ys.min()-my, ys.max()+my)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(name)
            ax.grid(alpha=.3); ax.axhline(0, color='k', lw=.4); ax.axvline(0, color='k', lw=.4)
        
        fig_poly.colorbar(lc, ax=axes, fraction=0.025, pad=0.02).set_label("tempo [s] (landing)")
        msg = f"Gate a t={tt[entry]:.2f}s" if entry is not None else "Gate mai attivo"
        fig_poly.suptitle(f"Stato-errore sul set controllabile (Mobile Frame) — {msg}")
        fig_poly.savefig(f"last_run_plots/flight_{stamp}_polytope.png", dpi=110, bbox_inches="tight")
        
    except FileNotFoundError:
        pass
 
    if np.sum(landing_mask) > 1:
        ex, ey, ez = ep[0], ep[1], ep[2]
        d_xy = np.sqrt(ex**2 + ey**2)
        margin = ez - alpha_cone * np.maximum(0, d_xy - r_base)
        n_out = int(np.sum((margin < -1e-6) & (ez < z_cut)))
 
        fig_cone = plt.figure(figsize=(11, 8))
        ax_cone = fig_cone.add_subplot(111, projection="3d")
        r_max = float(np.nanmax(d_xy)) * 1.05 + 1e-6
        rr, th = np.linspace(0, r_max, 30), np.linspace(0, 2*np.pi, 40)
        R, TH = np.meshgrid(rr, th)
        Xc, Yc = R*np.cos(TH), R*np.sin(TH)
        Zc = np.minimum(alpha_cone * np.maximum(0, R - r_base), z_cut)
        
        ax_cone.plot_surface(Xc, Yc, Zc, alpha=0.15, color="tab:green", linewidth=0, antialiased=True)
        ax_cone.plot_wireframe(Xc, Yc, Zc, color="tab:green", linewidth=0.4, rstride=4, cstride=4, alpha=0.5)
 
        pts3d = np.array([ex, ey, ez]).T.reshape(-1, 1, 3)
        segs3d = np.concatenate([pts3d[:-1], pts3d[1:]], axis=1)
        lc3d = Line3DCollection(segs3d, cmap="plasma", linewidth=2.5)
        lc3d.set_array(tt[:-1])
        ax_cone.add_collection3d(lc3d)
 
        ax_cone.scatter(ex[0], ey[0], ez[0], c="k", s=60, label="inizio landing")
        ax_cone.scatter(ex[-1], ey[-1], ez[-1], marker="*", s=260, c="red", edgecolor="k", label="touchdown")
        fig_cone.colorbar(lc3d, ax=ax_cone, fraction=0.03, pad=0.08).set_label("tempo [s] (landing)")
        ax_cone.set_xlabel("e_x [m]"); ax_cone.set_ylabel("e_y [m]"); ax_cone.set_zlabel("e_z [m]")
 
        xall = np.concatenate([Xc.ravel(), ex]); yall = np.concatenate([Yc.ravel(), ey])
        zall = np.concatenate([Zc.ravel(), ez])
        xr = (xall.min(), xall.max()); yr = (yall.min(), yall.max()); zr = (zall.min(), zall.max())
        max_range = max(xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]) / 2.0
        xm = 0.5*(xr[0]+xr[1]); ym = 0.5*(yr[0]+yr[1]); zm = 0.5*(zr[0]+zr[1])
        ax_cone.set_xlim(xm-max_range, xm+max_range)
        ax_cone.set_ylim(ym-max_range, ym+max_range)
        ax_cone.set_zlim(zm-max_range, zm+max_range)
        ax_cone.set_box_aspect((1, 1, 1))
        
        status = "DENTRO il cono" if n_out == 0 else f"{n_out}/{Np} campioni FUORI"
        ax_cone.set_title(f"Traiettoria di landing nel cono Mobile (alpha={alpha_cone}) — {status}")
        ax_cone.legend(loc="upper left")
        ax_cone.view_init(elev=18, azim=-60)
        fig_cone.savefig(f"last_run_plots/flight_{stamp}_cone.png", dpi=110, bbox_inches="tight")
 
    plt.show()


class LandingNode(Node):
    """
    Il main di volo E' questo nodo ROS. Il loop di controllo e' un timer ROS a
    1/DT Hz (single-threaded). Lo stato del drone arriva in modo ASINCRONO dal
    logger cflib (thread interno di cflib) e viene salvato in self.latest_state;
    il timer legge sempre l'ultimo stato disponibile.

    La logica di controllo dentro tick() e' IDENTICA al vecchio for-loop:
    stessa FSM, stessa guida (LOS / generatrice), stesso ctrl.compute,
    stesso send_setpoint. Cambia solo l'involucro (timer + callback al posto
    del for su SyncLogger).
    """

    def __init__(self):
        super().__init__("drone_landing_node")

        cflib.crtp.init_drivers()
        self.ctrl = HybridController(dt=DT)
        self.rows = []
        self.t_start = time.perf_counter()
        self.RAMP_T = 1.0
        
        self.V_LAND = 0.3
        self.V_NAV  = 0.3     # velocita' carrot in rising/nav [m/s]
        self.V_HOLD = 0.3     # velocita' carrot in hold [m/s]

        # stato del drone (aggiornato dal callback asincrono del logger)
        self.latest_state = None

        # --- connessione e setup sequenziale (arming, reset, warmup) ---
        # NB: teniamo il SyncCrazyflie aperto per tutta la vita del nodo.
        self.get_logger().info(f"Connessione a {URI} ...")
        self._scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache"))
        self._scf.open_link()
        self.cf = self._scf.cf

        self.cf.supervisor.send_arming_request(True); time.sleep(1.0)
        reset_estimator(self.cf)
        for _ in range(10):
            self.cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)

        # --- logger ASINCRONO: aggiorna self.latest_state ad ogni pacchetto ---
        self._logconf = build_logconf()
        self.cf.log.add_config(self._logconf)
        self._logconf.data_received_cb.add_callback(self._on_state)
        self._logconf.start()

        # --- stato FSM (erano variabili locali del for, ora attributi) ---
        self.state = "rising"
        self.wp_counter = RISING
        self.old_wp_id = IDLE
        self.stop_delta = 0.1
        self.WP = None
        self.prev_wp = self.wp_counter
        self.seg_t0 = time.perf_counter()
        self.dynamic_p_start = None
        self.land_t0 = None
        self.seg_p_start = None
        self.last_p_LOS = None   # ultima posizione reale della carota (per continuita' tra segmenti)
        self._finished = False

        # --- timer di controllo a 1/DT Hz: E' il loop ---
        self.timer = self.create_timer(DT, self.tick)
        self.get_logger().info("Nodo avviato: loop di controllo attivo.")

    def _on_state(self, timestamp, data, logconf):
        """Callback asincrono del logger cflib. Gira nel thread di cflib."""
        self.latest_state = data

    def tick(self):
        """Un ciclo di controllo. Identico al corpo del vecchio for-loop."""
        global TARGET_XY

        if self._finished:
            return
        data = self.latest_state
        if data is None:
            return  # nessuno stato ancora ricevuto

        pos = np.array([data["stateEstimate.x"], data["stateEstimate.y"],
                        data["stateEstimate.z"]])
        vel = np.array([data["stateEstimate.vx"], data["stateEstimate.vy"],
                        data["stateEstimate.vz"]])

        if self.WP is None:
            hx, hy = pos[0], pos[1]
            self.WP = np.array([
                [hx, hy, pos[2]],
                [hx, hy, Z_CRUISE],
                [TARGET_XY[0] - 3.5, TARGET_XY[1], Z_CRUISE],
                [TARGET_XY[0], TARGET_XY[1], Z_HOLD],
                [TARGET_XY[0], TARGET_XY[1], Z_LAND],
            ])

        if self.state == "idle":
            self._shutdown_flight()
            return

        landing = (self.state == "landing")
        
        if self.seg_p_start is None:
            self.seg_p_start = self.WP[self.old_wp_id].copy()

        if self.wp_counter != self.prev_wp:
            if self.wp_counter != LANDING:
                self.seg_t0 = time.perf_counter()
                # punto d'ingresso segmento: da dove la carota era REALMENTE rimasta
                # (non il waypoint nominale) per continuita' di posizione, dato che
                # col profilo trapezoidale la carota puo' non aver ancora raggiunto
                # p_end quando scatta la transizione (basata sulla tolleranza sul drone)
                self.seg_p_start = (self.last_p_LOS.copy() if self.last_p_LOS is not None
                                     else self.WP[self.old_wp_id].copy())
            self.prev_wp = self.wp_counter
        elapsed = time.perf_counter() - self.seg_t0

        step_disp = TARGET_VEL * DT

        TARGET_XY += step_disp[0:2]
        self.WP[HOLD] += step_disp
        self.WP[LANDING] += step_disp

        if self.state in ["hold", "landing"]:
            current_target_vel = TARGET_VEL
            if self.dynamic_p_start is not None:
                self.dynamic_p_start += step_disp
        else:
            current_target_vel = np.zeros(3)

        # --- GUIDA UNIFORME: riferimento rettilineo parametrizzato nel tempo ---
        # Carrot che scorre da p_start (ingresso segmento) a p_end a velocita' V
        # costante; il feed-forward di velocita' v_ff = V*u viene passato al PID/MPC.
        # Stessa formulazione in tutte le fasi (rising/nav/hold/landing).
        if self.state == "landing" and self.dynamic_p_start is not None:
            p_start = self.dynamic_p_start
            p_end = self.WP[LANDING]
            V = self.V_LAND
            tau = time.perf_counter() - self.land_t0
        else:
            p_start = self.seg_p_start
            p_end = self.WP[self.wp_counter]
            V = self.V_NAV if self.state in ["rising", "nav_to_wp"] else self.V_HOLD
            tau = elapsed

        seg = p_end - p_start
        L = np.linalg.norm(seg)
        if L < 1e-6:
            p_LOS = p_end.copy()
            v_ff = np.zeros(3)
        else:
            u = seg / L
            s_lin, v_scalar = trapz_profile(tau, L, V, self.RAMP_T)
            p_LOS = p_start + s_lin * u
            v_ff = v_scalar * u

        # NB: v_ff (rampa) NON viene piu' azzerato in landing. Il controllore lo
        # somma alla velocita' della piattaforma (target_vel) e lo instrada al
        # SOLO PID (target_vel + ramp_ref_vel in _pid_reach); l'MPC riceve solo
        # target_vel. Cosi' durante la fase PID del landing (prima del gate
        # reachable-set) il feed-forward di velocita' della discesa e' presente,
        # invece di essere perso. In hold e landing il riferimento totale e'
        # quindi rampa + feed-forward piattaforma, come voluto.

        self.last_p_LOS = p_LOS.copy()   # per la continuita' della carota al prossimo cambio segmento

        t0 = time.perf_counter()
        force, roll, pitch, yaw, mode = self.ctrl.compute(
            pos, vel, p_LOS, target_yaw=0.0, target_vel=current_target_vel, ramp_ref_vel=v_ff,
            a_xy_lim=A_XY, final_pos=self.WP[LANDING], landing=landing)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        cmd = force_to_cmd(force)
        self.cf.commander.send_setpoint(rad2deg(roll), rad2deg(pitch), 0.0, cmd)

        az = force / MASS - G
        self.rows.append(dict(
            t=time.perf_counter() - self.t_start, state=self.state, mode=mode,
            x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2],
            carrot_x=p_LOS[0], carrot_y=p_LOS[1], carrot_z=p_LOS[2], force=force, cmd=cmd, az=az,
            roll=roll, pitch=pitch, solve_ms=solve_ms,
            ref_vx=v_ff[0], ref_vy=v_ff[1], ref_vz=v_ff[2],
            target_x=self.WP[LANDING][0], target_y=self.WP[LANDING][1], target_z=self.WP[LANDING][2],
            target_vx=current_target_vel[0], target_vy=current_target_vel[1], target_vz=current_target_vel[2]))

        print(f"[{self.state:9s} mode={mode}] x={pos[0]:5.2f} y={pos[1]:5.2f} z={pos[2]:5.2f} "
              f"cmd={cmd:5d} az={az:+5.2f} carrot_z={p_LOS[2]:.2f}")

        pos_e = self.WP[self.wp_counter] - pos
        distance = np.linalg.norm(pos_e)
        # la transizione scatta solo quando il drone e' vicino ABBASTANZA E la
        # carota ha finito la sua rampa (s_lin ha raggiunto L, quindi v_scalar=0
        # e la fase di decelerazione e' gia' avvenuta) - altrimenti lo
        # state-machine "batte sul tempo" il trapezio e la decelerazione non
        # scatta mai (vedi analisi: d_acc=0.5*V*RAMP_T era < stop_delta).
        carrot_arrived = (s_lin >= L - 1e-3) if L > 1e-6 else True

        # --- terminazione del landing: condizione dedicata, puramente verticale ---
        # Il landing e' concluso appena il drone raggiunge (o scende sotto) la
        # quota di terra Z_LAND. NON usa il gate xy+carrot delle transizioni di
        # navigazione: in coda il drone e' gia' a terra (z<=Z_LAND) ma dxy puo'
        # rioscillare sopra stop_delta, lasciando il drone "appeso" nello stato
        # landing per secondi (l'MPC continua a tenere z al riferimento). Questo
        # produceva l'hovering finale e i campioni fuori-cono al vertice.
        if self.state == "landing" and pos[2] <= Z_LAND+1e-2:
            self.state = "idle"; self.old_wp_id, self.wp_counter = LANDING, IDLE
            return

        if distance <= self.stop_delta and carrot_arrived:
            if self.state == "rising":
                self.state = "nav_to_wp"; self.old_wp_id, self.wp_counter, self.stop_delta = RISING, NAV, 0.3
            elif self.state == "hold":
                self.state = "landing"; self.old_wp_id, self.wp_counter, self.stop_delta = HOLD, LANDING, 0.05
                self.dynamic_p_start = (self.last_p_LOS.copy() if self.last_p_LOS is not None
                                         else self.WP[HOLD].copy())
                self.land_t0 = time.perf_counter()
            elif self.state == "nav_to_wp":
                self.state = "hold"; self.old_wp_id, self.wp_counter, self.stop_delta = NAV, HOLD, 0.3
                dist_xy_start = (Z_HOLD - Z_LAND) / ALPHA_CONE
                seg_nav = self.WP[HOLD][0:2] - self.WP[NAV][0:2]
                Lnav = np.linalg.norm(seg_nav)
                u_nav = seg_nav / Lnav if Lnav > 1e-6 else np.zeros(2)
                P_start_xy = TARGET_XY - u_nav * dist_xy_start
                self.WP[HOLD] = np.array([P_start_xy[0], P_start_xy[1], Z_HOLD])

    def _shutdown_flight(self):
        """Fine missione: ferma motori, salva CSV/plot, chiude il nodo."""
        if self._finished:
            return
        self._finished = True
        self.timer.cancel()
        try:
            for _ in range(20):
                self.cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)
            self.cf.commander.send_stop_setpoint()
        finally:
            try:
                self._logconf.stop()
            except Exception:
                pass
            try:
                self._scf.close_link()
            except Exception:
                pass

        self.get_logger().info("Volo terminato: salvo CSV e plot...")
        stamp = save_and_plot(self.rows)
        plot_advanced_diagnostics(rows=self.rows, stamp=stamp,
                                  alpha_cone=ALPHA_CONE, z_cut=1.0, r_base=0.3)
        self.get_logger().info("Fatto. Puoi chiudere con Ctrl+C.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # se interrotto a meta', prova a fermare i motori in sicurezza
        try:
            node._shutdown_flight()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()