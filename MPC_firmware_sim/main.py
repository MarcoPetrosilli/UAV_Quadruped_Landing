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
ALPHA_CONE = 5.6
LOS_DELTA = 0.3
A_XY = 0.17
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
    csv_path = f"flight_{stamp}.csv"
    cols = ["t", "state", "mode", "x", "y", "z", "vx", "vy", "vz",
            "carrot_z", "force", "cmd", "az", "solve_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"CSV salvato: {csv_path}  ({len(rows)} righe)")

    a = {c: np.array([r[c] for r in rows], dtype=float) for c in cols if c != "state"}
    mode = a["mode"]; t = a["t"]

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

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(11, 9))

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
    ax[3].set_ylabel("az [m/s^2]"); ax[3].set_xlabel("t [s]"); shade_mpc(ax[3])

    sw = np.where(np.diff(mode) > 0.5)[0]
    for s in sw:
        for axi in ax:
            axi.axvline(t[s + 1], color="orange", lw=1.0)

    fig.suptitle("Volo CrazySim — zona arancione = MPC attivo")
    fig.tight_layout()
    png = f"flight_{stamp}.png"; fig.savefig(png, dpi=110)
    print(f"plot salvato: {png}")
    plt.show()


def main():
    cflib.crtp.init_drivers()
    ctrl = HybridController(dt=DT)
    rows = []
    t_start = time.perf_counter()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        cf.supervisor.send_arming_request(True); time.sleep(1.0)
        reset_estimator(cf)
        for _ in range(10):
            cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)

        state = "rising"; wp_counter = RISING; old_wp_id = IDLE
        stop_delta = 0.1; WP = None

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

                    p_LOS, _ = LOS_wp(pos, WP[old_wp_id], WP[wp_counter],
                                      delta=LOS_DELTA, stop_delta=stop_delta)
                    landing = (state == "landing")

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
                        carrot_z=p_LOS[2], force=force, cmd=cmd, az=az, solve_ms=solve_ms))

                    print(f"[{state:9s} mode={mode}] z={pos[2]:5.2f} "
                          f"cmd={cmd:5d} az={az:+5.2f} carrot_z={p_LOS[2]:.2f}")

                    pos_e = WP[wp_counter] - pos
                    distance = np.linalg.norm(pos_e)
                    if state == "hold":
                        d_xy = np.linalg.norm(TARGET_XY - pos[0:2])
                        if d_xy <= Z_HOLD / ALPHA_CONE:
                            state = "landing"; old_wp_id, wp_counter, stop_delta = HOLD, LANDING, 0.1
                    elif distance <= stop_delta:
                        if state == "rising":
                            state = "nav_to_wp"; old_wp_id, wp_counter, stop_delta = RISING, NAV, 0.3
                        elif state == "nav_to_wp":
                            state = "hold"; old_wp_id, wp_counter, stop_delta = NAV, HOLD, 0.3
                        elif state == "landing":
                            state = "idle"; old_wp_id, wp_counter = LANDING, IDLE
        finally:
            for _ in range(20):
                cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)
            cf.commander.send_stop_setpoint()

    save_and_plot(rows)


if __name__ == "__main__":
    main()