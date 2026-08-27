"""
main_crazysim.py  —  LOS + FSM (percorso a L) + logging CSV + plot diagnostico.

Versione con i TUOI valori: HOVER_CMD=32000, massa 0.0379, Z_LAND=0.1,
TARGET_XY=[3.5,3.5], NAV a (0,3.5). Registra tutto su CSV e a fine volo apre
un grafico a 4 pannelli (z/carrot, vz, cmd, az) con la zona MPC ombreggiata e
una riga verticale allo switch PID->MPC. In console: statistiche loop e solve.
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

from controller_deploy import HybridController

URI = uri_helper.uri_from_env(default='udp://127.0.0.1:19850')
DT = 0.02
G = 9.81

# ---- calibrazione spinta -----------------------------------------------------
HOVER_CMD = 32000
MASS = 0.0379                      # deve combaciare con self.M nel controllore
HOVER_FORCE = MASS * G


def force_to_cmd(force_N):
    return int(np.clip(HOVER_CMD * force_N / HOVER_FORCE, 10001, 60000))


def rad2deg(x):
    return x * 180.0 / math.pi


# ---- missione (statica, percorso a L) ----------------------------------------
TARGET_XY = np.array([3.5, 3.5])
Z_CRUISE = 1.8
Z_LAND = 0.1
Z_HOLD = 1.8
#ALPHA_CONE = 5.6
ALPHA_CONE = 1.0
LOS_DELTA = 0.3
A_XY = 0.17
R_BASE_CONE = 0.1   # deve combaciare con self.r_base del controllore
IDLE, RISING, NAV, HOLD, LANDING = 0, 1, 2, 3, 4


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
            "carrot_x", "carrot_y", "carrot_z", "force", "cmd", "az", "solve_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"CSV salvato: {csv_path}  ({len(rows)} righe)")

    # Estraiamo i dati numerici
    a = {c: np.array([r[c] for r in rows], dtype=float) for c in cols if c != "state"}
    mode = a["mode"]; t = a["t"]
    
    # NOVITÀ: Estraiamo la sequenza degli stati (stringhe)
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

    # Aumentato leggermente figsize per fare spazio alle etichette di stato
    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(11, 10))

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
    ax[1].set_ylabel("vz [m/s]"); shade_mpc(ax[1])

    ax[2].plot(t, a["cmd"], color="tab:red")
    ax[2].axhline(HOVER_CMD, color="k", lw=0.8, ls=":", label="hover cmd")
    ax[2].axhline(60000, color="gray", lw=0.6, ls=":")
    ax[2].set_ylabel("thrust cmd"); ax[2].legend(loc="upper right"); shade_mpc(ax[2])

    ax[3].plot(t, a["az"], color="tab:purple"); ax[3].axhline(0, color="k", lw=0.6)
    ax[3].set_ylabel("az [m/s^2]"); shade_mpc(ax[3])

    ax[4].plot(t, a["x"], label="x", lw=1.5)
    ax[4].plot(t, a["carrot_x"], label="carrot_x", lw=1, ls="--")
    ax[4].axhline(3.5, color="k", lw=0.8, ls=":", label="target land")
    ax[4].set_ylabel("x [m]"); ax[4].legend(loc="upper right"); shade_mpc(ax[4])

    # Corretto label='y' e spostato set_xlabel in fondo
    ax[5].plot(t, a["y"], label="y", lw=1.5)
    ax[5].plot(t, a["carrot_y"], label="carrot_y", lw=1, ls="--")
    ax[5].axhline(3.5, color="k", lw=0.8, ls=":", label="target land")
    ax[5].set_ylabel("y [m]"); ax[5].set_xlabel("t [s]"); ax[5].legend(loc="upper right"); shade_mpc(ax[5])


    # =====================================================================
    # NOVITÀ: LINEE VERTICALI PER OGNI TRANSIZIONE DI STATO (FSM)
    # =====================================================================
    trans_idx = np.where(state_str[:-1] != state_str[1:])[0]
    
    # Etichetta dello stato iniziale all'istante t=0
    if len(t) > 0:
        ax[0].text(t[0], 1.05, f" {state_str[0].upper()}", transform=ax[0].get_xaxis_transform(),
                   fontsize=9, color="black", fontweight="bold", alpha=0.7)

    for idx in trans_idx:
        t_trans = t[idx + 1]
        new_state = state_str[idx + 1]
        
        # Disegna la linea tratteggiata su tutti i grafici
        for axi in ax:
            axi.axvline(t_trans, color="black", linestyle="--", lw=1.2, alpha=0.6)
        
        # Scrive il nome del nuovo stato sopra il primo grafico
        ax[0].text(t_trans, 1.05, f" {new_state.upper()}", transform=ax[0].get_xaxis_transform(),
                   fontsize=9, color="black", fontweight="bold", alpha=0.7)
    # =====================================================================

    fig.suptitle("Volo CrazySim — Transizioni FSM e MPC attivo (sfondo arancione)", y=0.99)
    fig.tight_layout()
    # Lasciamo spazio in alto per i nomi degli stati
    fig.subplots_adjust(top=0.94)
    
    png = f"last_run_plots/flight_{stamp}.png"; fig.savefig(png, dpi=110)
    print(f"plot salvato: {png}")
    plt.show()

    return stamp

def plot_advanced_diagnostics(rows, target_xy, z_land, stamp=None, alpha_cone=1.0, z_cut=1.0, r_base=0.1, polytope_path="reachable_polytope.npz"):
    import numpy as np
    import os
    from datetime import datetime
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError:
        print("Matplotlib non disponibile per i plot avanzati.")
        return
 
    if not rows:
        return
 
    # Genera un timestamp se non viene passato
    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Crea la cartella se non esiste
    save_dir = "last_run_plots"
    os.makedirs(save_dir, exist_ok=True)
 
    # 1. Estrazione dati dal dizionario
    t = np.array([r["t"] for r in rows])
    pos = np.array([[r["x"], r["y"], r["z"]] for r in rows]).T
    vel = np.array([[r["vx"], r["vy"], r["vz"]] for r in rows]).T
    los = np.array([[r["carrot_x"], r["carrot_y"], r["carrot_z"]] for r in rows]).T
    state_str = np.array([r["state"] for r in rows])
    
    landing_mask = (state_str == "landing")
    fp = np.array([target_xy[0], target_xy[1], z_land])
 
    # =====================================================================
    # PLOT 1: Traiettoria 3D e 2D (Reale vs LOS)
    # =====================================================================
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    fig2d = plt.figure(figsize=(10, 8))
    ax2d = fig2d.add_subplot(111)
 
    # 3D
    ax3d.plot(pos[0], pos[1], pos[2], label="Trajectory", color='b', linewidth=2)
    ax3d.plot(los[0], los[1], los[2], label="LOS Target", color='g', linestyle='--', linewidth=1.5)
    step_size = max(1, int(pos.shape[1] / 50))
    ax3d.quiver(pos[0, ::step_size], pos[1, ::step_size], pos[2, ::step_size],
                los[0, ::step_size] - pos[0, ::step_size],
                los[1, ::step_size] - pos[1, ::step_size],
                los[2, ::step_size] - pos[2, ::step_size],
                color='r', alpha=0.6, arrow_length_ratio=0.15, linewidth=1.5, label="Error Vector")
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
    ax3d.set_title('3D Tracking: Actual Position vs LOS Target')
    ax3d.legend()
    
    png_3d = f"{save_dir}/flight_{stamp}_3d_track.png"
    fig3d.savefig(png_3d, dpi=110)
    print(f"Plot 3D salvato: {png_3d}")
 
    # 2D
    ax2d.plot(pos[0], pos[1], label="Trajectory", color='b', linewidth=2)
    ax2d.plot(los[0], los[1], label="LOS Target", color='g', linestyle='--', linewidth=1.5)
    ax2d.quiver(pos[0, ::step_size], pos[1, ::step_size],
                los[0, ::step_size] - pos[0, ::step_size],
                los[1, ::step_size] - pos[1, ::step_size],
                angles='xy', scale_units='xy', scale=1, color='r', alpha=0.6, width=0.003, label="Error Vector")
    ax2d.set_xlabel('X [m]'); ax2d.set_ylabel('Y [m]')
    ax2d.set_title('2D Top-Down View: XY Tracking')
    ax2d.axis('equal'); ax2d.grid(True); ax2d.legend()
    
    png_2d = f"{save_dir}/flight_{stamp}_2d_track.png"
    fig2d.savefig(png_2d, dpi=110)
    print(f"Plot 2D salvato: {png_2d}")
 
    # =====================================================================
    # PLOT 2: Reachable Polytope Entry (Solo fase Landing)
    # =====================================================================
    try:
        d = np.load(polytope_path)
        H_ax, h_ax, H_vz, h_vz = d["H_ax"], d["h_ax"], d["H_vz"], d["h_vz"]
        V_ax, V_vz = d["V_ax"], d["V_vz"]
        
        ep, ev, tt = pos[:, landing_mask] - fp[:, None], vel[:, landing_mask], t[landing_mask]
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
        fig_poly.suptitle(f"Stato-errore sul set controllabile — {msg}")
        
        png_poly = f"{save_dir}/flight_{stamp}_polytope.png"
        fig_poly.savefig(png_poly, dpi=110, bbox_inches="tight")
        print(f"Plot Politopo salvato: {png_poly}")
        
    except FileNotFoundError:
        print(f"File {polytope_path} non trovato, plot politopo ignorato.")
 
    # =====================================================================
    # PLOT 3: Glideslope Cone (CBF)
    # =====================================================================
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
 
        # --- ASPECT RATIO FISICO: 1 metro = stessa lunghezza su x, y, z ---------
        # senza questo matplotlib rende il box cubico e il cono appare sempre
        # ugualmente svasato a prescindere da alpha. Imponiamo range uguali sui
        # tre assi (il piu' grande dei tre) centrati, cosi' la pendenza reale
        # del cono (governata da alpha) e' visibile.
        xall = np.concatenate([Xc.ravel(), ex]); yall = np.concatenate([Yc.ravel(), ey])
        zall = np.concatenate([Zc.ravel(), ez])
        xr = (xall.min(), xall.max()); yr = (yall.min(), yall.max()); zr = (zall.min(), zall.max())
        max_range = max(xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]) / 2.0
        xm = 0.5*(xr[0]+xr[1]); ym = 0.5*(yr[0]+yr[1]); zm = 0.5*(zr[0]+zr[1])
        ax_cone.set_xlim(xm-max_range, xm+max_range)
        ax_cone.set_ylim(ym-max_range, ym+max_range)
        ax_cone.set_zlim(zm-max_range, zm+max_range)
        ax_cone.set_box_aspect((1, 1, 1))   # box cubico + range uguali => scala metrica isotropa
        status = "DENTRO il cono" if n_out == 0 else f"{n_out}/{Np} campioni FUORI"
        ax_cone.set_title(f"Traiettoria di landing nel cono (alpha={alpha_cone}) — {status}")
        ax_cone.legend(loc="upper left")
        ax_cone.view_init(elev=18, azim=-60)
        
        png_cone = f"{save_dir}/flight_{stamp}_cone.png"
        fig_cone.savefig(png_cone, dpi=110, bbox_inches="tight")
        print(f"Plot Cono salvato: {png_cone}")
 
    plt.show()



def main():
    cflib.crtp.init_drivers()
    ctrl = HybridController(dt=DT)
    rows = []
    t_start = time.perf_counter()
    RAMP_T = 0.5
    V_LAND = 0.5   # m/s, velocita' fissa della diagonale di landing (rif. PID)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        cf.supervisor.send_arming_request(True); time.sleep(1.0)
        reset_estimator(cf)
        for _ in range(10):
            cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)

        state = "rising"; wp_counter = RISING; old_wp_id = IDLE
        stop_delta = 0.1; WP = None
        prev_wp = wp_counter
        seg_t0 = time.perf_counter()

        dynamic_p_start = None
        land_t0 = None

        try:
            with SyncLogger(scf, build_logconf()) as logger:
                for _ts, data, _ in logger:
                    pos = np.array([data["stateEstimate.x"], data["stateEstimate.y"],
                                    data["stateEstimate.z"]])
                    vel = np.array([data["stateEstimate.vx"], data["stateEstimate.vy"],
                                    data["stateEstimate.vz"]])

                    if WP is None:
                        hx, hy = pos[0], pos[1]
                        WP = np.array([
                            [hx, hy, pos[2]],                              # 0 IDLE
                            [hx, hy, Z_CRUISE],                            # 1 RISING
                            [TARGET_XY[0] - 3.5, TARGET_XY[1], Z_CRUISE],  # 2 NAV (0,3.5)
                            [TARGET_XY[0], TARGET_XY[1], Z_HOLD],          # 3 HOLD
                            [TARGET_XY[0], TARGET_XY[1], Z_LAND],          # 4 LANDING
                        ])

                    if state == "idle":
                        break

                    landing = (state == "landing")

                    # rampa TEMPORALE del look-ahead: al cambio segmento delta_eff
                    # riparte da 0 e sale a LOS_DELTA in RAMP_T secondi (indipendente da s)
                    if wp_counter != prev_wp:
                        if wp_counter != LANDING:
                            seg_t0 = time.perf_counter()
                        prev_wp = wp_counter
                    elapsed = time.perf_counter() - seg_t0
                    delta_eff = LOS_DELTA * min(1.0, elapsed / RAMP_T)
                    #delta_eff = LOS_DELTA
                    if state == "landing" and dynamic_p_start is not None:
                        # --- riferimento DIAGONALE a velocita' fissa, dipendente solo dal tempo ---
                        # retta da dynamic_p_start (punto d'ingresso landing) al target di touchdown,
                        # percorsa a V_LAND [m/s]. Feedforward puro: NON dipende da pos -> niente
                        # accoppiamento/ondulazione. Serve al PID finche' l'MPC non subentra.
                        p_end_land = WP[LANDING]                      # (target_x, target_y, Z_LAND)
                        seg = p_end_land - dynamic_p_start
                        L = np.linalg.norm(seg)
                        tau = time.perf_counter() - land_t0
                        if L < 1e-6:
                            p_LOS = p_end_land.copy()
                        else:
                            s_lin = min(L, V_LAND * tau)              # distanza percorsa lungo la retta
                            p_LOS = dynamic_p_start + (s_lin / L) * seg
                    else:
                        current_p_start = WP[old_wp_id]
                        p_LOS, _ = LOS_wp(pos, current_p_start, WP[wp_counter],
                                          delta=delta_eff, stop_delta=stop_delta)

                    t0 = time.perf_counter()
                    force, roll, pitch, yaw, mode = ctrl.compute(
                        pos, vel, p_LOS, target_yaw=0.0, target_vel=np.zeros(3),
                        a_xy_lim=A_XY, final_pos=WP[LANDING], landing=landing)
                    solve_ms = (time.perf_counter() - t0) * 1000.0

                    cmd = force_to_cmd(force)
                    cf.commander.send_setpoint(rad2deg(roll), rad2deg(pitch), 0.0, cmd)

                    az = force / MASS - G
                    rows.append(dict(
                        t=time.perf_counter() - t_start, state=state, mode=mode,
                        x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2],
                        carrot_x=p_LOS[0], carrot_y=p_LOS[1], carrot_z=p_LOS[2], force=force, cmd=cmd, az=az, solve_ms=solve_ms))

                    print(f"[{state:9s} mode={mode}] x={pos[0]:5.2f} y={pos[1]:5.2f} z={pos[2]:5.2f} "
                          f"cmd={cmd:5d} az={az:+5.2f} carrot_z={p_LOS[2]:.2f}")

                    pos_e = WP[wp_counter] - pos
                    distance = np.linalg.norm(pos_e)
                    if distance <= stop_delta:
                        if state == "rising":
                            state = "nav_to_wp"; old_wp_id, wp_counter, stop_delta = RISING, NAV, 0.3
                        elif state == "hold":
                            state = "landing"; old_wp_id, wp_counter, stop_delta = HOLD, LANDING, 0.05
                            dynamic_p_start = WP[HOLD].copy()   # = P_start, dove il drone e' ora
                            land_t0 = time.perf_counter()
                        elif state == "nav_to_wp":
                            state = "hold"; old_wp_id, wp_counter, stop_delta = NAV, HOLD, 0.3
                            # --- sposta il wp di HOLD sul punto d'aggancio del glideslope (P_start) ---
                            # P_start sta sul segmento nav->hold, a dist_xy=(Z_HOLD-Z_LAND)/alpha
                            # dal target: e' il punto da cui una retta di pendenza alpha, puntando
                            # all'asse del cono (target,Z_LAND), sale fino a quota Z_HOLD. Cosi'
                            # la retta di landing (P_start->target) sta DENTRO il cono con margine
                            # alpha*r_base (grazie al cilindro di base), e il landing scatta quando
                            # il drone RAGGIUNGE P_start -> nessun salto di riferimento.
                            dist_xy_start = (Z_HOLD - Z_LAND) / ALPHA_CONE 
                            seg_nav = WP[HOLD][0:2] - WP[NAV][0:2]
                            Lnav = np.linalg.norm(seg_nav)
                            u_nav = seg_nav / Lnav if Lnav > 1e-6 else np.zeros(2)
                            P_start_xy = TARGET_XY - u_nav * dist_xy_start
                            WP[HOLD] = np.array([P_start_xy[0], P_start_xy[1], Z_HOLD])
                        elif state == "landing":
                            state = "idle"; old_wp_id, wp_counter = LANDING, IDLE
        finally:
            for _ in range(20):
                cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)
            cf.commander.send_stop_setpoint()

    current_stamp = save_and_plot(rows)

    plot_advanced_diagnostics(
        rows=rows, 
        target_xy=TARGET_XY, 
        z_land=Z_LAND, 
        stamp=current_stamp,
        alpha_cone=ALPHA_CONE,
        z_cut=1.0, 
        r_base=0.3
    )


if __name__ == "__main__":
    main()