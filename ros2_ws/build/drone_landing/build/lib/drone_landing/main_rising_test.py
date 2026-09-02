"""
main_rising_test.py  —  PROVA MINIMALE su drone reale.

Solo due stati: rising (sale a Z_TEST) -> landing (torna giu' a Z_LAND),
entrambi gestiti dal PID puro (self.ctrl.compute con landing=False, quindi
l'MPC non si attiva MAI: e' una prova del solo anello PID di reach).

Riprende lo stile del main vero: nodo ROS + timer a 1/DT, logger cflib
asincrono, profilo di velocita' trapezoidale con feed-forward v_ff instradato
al PID, transizione di stato che controlla la distanza MA rispetta anche il
completamento del profilo (carrot_arrived), CSV + plot diagnostico.
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
from geometry_msgs.msg import PoseStamped   # topic mocap /cf_drone/pose

try:
    from drone_landing.controller_deploy import HybridController   # dentro package ROS
except ImportError:
    from controller_deploy import HybridController                 # standalone

URI = uri_helper.uri_from_env(default='radio://0/80/2M')
MOCAP_TOPIC = "/cf_drone/pose"
DT = 0.02
G = 9.81

# ---- calibrazione spinta -----------------------------------------------------
HOVER_CMD = 38000
#MASS = 0.0379
MASS = 0.029
HOVER_FORCE = MASS * G


def force_to_cmd(force_N):
    return int(np.clip(HOVER_CMD * force_N / HOVER_FORCE, 10001, 60000))


def rad2deg(x):
    return x * 180.0 / math.pi


# ---- missione minimale: solo salita e discesa verticale ----------------------
Z_TEST = 1.40          # quota di prova [m]
Z_LAND = 0.1          # quota di "atterraggio" [m]
A_XY = 0.17
IDLE, RISING, LANDING = 0, 1, 2


def trapz_profile(tau, L, V, t_acc):
    """Profilo posizione/velocita' trapezoidale (rampa ad accelerazione limitata).
    tau: tempo dall'ingresso nel segmento. L: lunghezza. V: velocita' di crociera.
    t_acc: tempo di accel/decel (RAMP_T). Ritorna (s, v) ascissa e velocita' scalare."""
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
            "ref_vx", "ref_vy", "ref_vz"]
    import os
    os.makedirs("last_run_plots", exist_ok=True)
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

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib non disponibile:", e); return

    # ax_est/ay_est: accelerazione orizzontale dagli angoli comandati (roll/pitch).
    # pitch positivo -> +x, roll positivo -> -y (convenzione send_setpoint).
    ax_est = G * np.tan(a["pitch"])
    ay_est = -G * np.tan(a["roll"])

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(11, 12))

    ax[0].plot(t, a["z"], label="z", lw=1.5)
    ax[0].plot(t, a["carrot_z"], label="carrot_z", lw=1, ls="--")
    ax[0].axhline(Z_TEST, color="gray", lw=0.8, ls=":", label="z test")
    ax[0].axhline(Z_LAND, color="k", lw=0.8, ls=":", label="target land")
    ax[0].set_ylabel("z [m]"); ax[0].legend(loc="upper right")

    ax[1].plot(t, a["vz"], color="tab:green", label="vz")
    ax[1].plot(t, a["ref_vz"], color="tab:green", lw=1, ls="--", label="ref vz")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("vz [m/s]"); ax[1].legend(loc="upper right")

    ax[2].plot(t, a["cmd"], color="tab:red")
    ax[2].axhline(HOVER_CMD, color="k", lw=0.8, ls=":", label="hover cmd")
    ax[2].set_ylabel("thrust cmd"); ax[2].legend(loc="upper right")

    ax[3].plot(t, a["az"], color="tab:purple"); ax[3].axhline(0, color="k", lw=0.6)
    ax[3].set_ylabel("az [m/s^2]")

    ax[4].plot(t, a["x"], label="x", lw=1.2)
    ax[4].plot(t, a["y"], label="y", lw=1.2)
    ax[4].axhline(0, color="k", lw=0.6)
    ax[4].set_ylabel("x,y [m]"); ax[4].legend(loc="upper right")

    ax[5].plot(t, a["vx"], label="vx", color="tab:blue", lw=1.3)
    ax[5].plot(t, a["vy"], label="vy", color="tab:orange", lw=1.3)
    ax[5].plot(t, a["ref_vx"], color="tab:blue", lw=1, ls="--", label="ref vx")
    ax[5].plot(t, a["ref_vy"], color="tab:orange", lw=1, ls="--", label="ref vy")
    ax[5].axhline(0, color="k", lw=0.6)
    ax[5].set_ylabel("v_xy [m/s]"); ax[5].legend(loc="upper right", ncol=2, fontsize=8)

    ax[6].plot(t, ax_est, label="ax (da pitch)", color="tab:blue", lw=1.3)
    ax[6].plot(t, ay_est, label="ay (da roll)", color="tab:orange", lw=1.3)
    ax[6].axhline(0, color="k", lw=0.6)
    ax[6].set_ylabel("a_xy [m/s^2]"); ax[6].set_xlabel("t [s]")
    ax[6].legend(loc="upper right")

    trans_idx = np.where(state_str[:-1] != state_str[1:])[0]
    for idx in trans_idx:
        for axi in ax:
            axi.axvline(t[idx + 1], color="black", linestyle="--", lw=1.0, alpha=0.5)

    fig.suptitle("Prova RISING + LANDING PID puro — drone reale", y=0.99)
    fig.tight_layout(); fig.subplots_adjust(top=0.95)
    png = f"last_run_plots/flight_{stamp}.png"; fig.savefig(png, dpi=110)
    print(f"plot salvato: {png}")
    plt.show()
    return stamp


class RisingTestNode(Node):
    def __init__(self):
        super().__init__("rising_test_node")

        cflib.crtp.init_drivers()
        self.ctrl = HybridController(dt=DT)
        self.rows = []
        self.t_start = time.perf_counter()
        self.RAMP_T = 0.8
        self.V_RISE = 0.3     # velocita' carrot in salita [m/s]
        self.V_LAND = 0.3     # velocita' carrot in discesa [m/s]

        self.latest_state = None          # ultimo pacchetto logger cflib (velocita' Kalman)
        self.latest_pose = None           # ultima posizione dal mocap [x,y,z]
        self.SETTLE_T = 3.0               # assestamento stima prima di partire [s]

        self.get_logger().info(f"Connessione a {URI} ...")
        self._scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache"))
        self._scf.open_link()
        self.cf = self._scf.cf

        # --- configura l'estimatore PRIMA di armare/resettare ---
        # Perche' la posizione esterna (mocap) venga fusa serve l'estimatore
        # Kalman attivo (estimator=2) e una deviazione standard della misura
        # esterna ragionevole (piccola = "mi fido molto del mocap"). Senza
        # questo, il filtro ignora/mal-usa la extpos e la stima di velocita'
        # diventa spazzatura (vz impazzita a drone fermo).
        try:
            self.cf.param.set_value('stabilizer.estimator', '2')      # 2 = Kalman
            time.sleep(0.1)
            self.cf.param.set_value('locSrv.extPosStdDev', '0.01')    # std pos esterna [m]
            time.sleep(0.1)
        except Exception as e:
            self.get_logger().warn(f"config estimatore fallita: {e}")

        self.cf.supervisor.send_arming_request(True); time.sleep(1.0)

        # --- sottoscrizione al mocap (posizione) ---
        # /cf_drone/pose e' un PoseStamped: uso solo pose.position (x,y,z);
        # l'orientamento in quaternioni non serve al controllore.
        # NB: se il topic fosse un geometry_msgs/Pose (non "Stamped"), cambia
        # l'import e usa msg.position invece di msg.pose.position nella callback.
        self._pose_sub = self.create_subscription(
            PoseStamped, MOCAP_TOPIC, self._on_pose, 10)

        # --- logger ASINCRONO: da qui prendo le VELOCITA' (filtro di Kalman) ---
        self._logconf = build_logconf()
        self.cf.log.add_config(self._logconf)
        self._logconf.data_received_cb.add_callback(self._on_state)
        self._logconf.start()

        # --- reset Kalman + assestamento prima di partire ---
        self._warmup()

        # --- FSM: solo rising -> landing ---
        self.state = "rising"
        self.stop_delta = 0.1
        self.WP = None                 # [rising_target, land_target], riempito al 1o tick
        self.seg_p_start = None        # ingresso segmento (per la rampa)
        self.last_p_LOS = None         # ultima pos reale della carota
        self.seg_t0 = time.perf_counter()
        self.prev_state = self.state
        self._finished = False

        self.timer = self.create_timer(DT, self.tick)
        self.get_logger().info("Nodo avviato: loop di controllo attivo.")

    def _on_state(self, timestamp, data, logconf):
        self.latest_state = data

    def _on_pose(self, msg):
        """Callback mocap.

        1) Rimanda la POSIZIONE al drone via send_extpos, cosi' il filtro di
           Kalman onboard la fonde (assetto stimato dalla IMU). Non si mandano i
           quaternioni: erano rumorosi e potevano destabilizzare la parte di
           assetto del filtro. Se in futuro servisse la posa completa, tornare a
           send_extpose(x,y,z, qx,qy,qz,qw).
        2) Salva la posizione (x,y,z) per il controllo.
        """
        p = msg.pose.position
        try:
            self.cf.extpos.send_extpos(p.x, p.y, p.z)
        except Exception as e:
            self.get_logger().warn(f"send_extpos fallito: {e}")

        self.latest_pose = np.array([p.x, p.y, p.z])

    def _warmup(self):
        """Reset del filtro di Kalman e attesa di assestamento prima di partire.

        Durante l'attesa serve pompare le callback ROS (rclpy.spin_once) per
        ricevere il primo pacchetto mocap: rclpy.spin() non e' ancora attivo in
        __init__, quindi senza spin_once la posizione non arriverebbe mai. Nel
        frattempo si inviano setpoint a zero per tenere vivo il commander.
        """
        self.get_logger().info("Reset del filtro di Kalman...")
        reset_estimator(self.cf)   # reset + settle interno (~1.6s)

        self.get_logger().info("Attendo primo pose mocap e primo stato...")
        t0 = time.perf_counter()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)          # pompa callback mocap
            self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
            if self.latest_pose is not None and self.latest_state is not None:
                break
            if time.perf_counter() - t0 > 10.0:
                self.get_logger().warn("Timeout: pose mocap o stato non ricevuti!")
                break
            time.sleep(DT)

        self.get_logger().info(f"Assestamento stima per {self.SETTLE_T:.1f}s...")
        t0 = time.perf_counter()
        while rclpy.ok() and (time.perf_counter() - t0) < self.SETTLE_T:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
            time.sleep(DT)
        self.get_logger().info("Assestamento completato: avvio controllo.")

    def tick(self):
        if self._finished:
            return
        data = self.latest_state
        pose_xyz = self.latest_pose
        if data is None or pose_xyz is None:
            return

        # POSIZIONE dal mocap (/cf_drone/pose); VELOCITA' dal filtro di Kalman
        # onboard (stateEstimate.vx/vy/vz del logger cflib).
        pos = pose_xyz.copy()
        vel = np.array([data["stateEstimate.vx"], data["stateEstimate.vy"],
                        data["stateEstimate.vz"]])

        # waypoint verticali costruiti sulla posizione di partenza (x,y fissi)
        if self.WP is None:
            hx, hy = pos[0], pos[1]
            self.WP = {
                "rising":  np.array([hx, hy, Z_TEST]),
                "nav":  np.array([hx, hy-1.5, Z_TEST]),
                "hold":  np.array([hx+1.5, hy-1.5, Z_TEST]),
                "landing": np.array([hx+1.5, hy-1.5, Z_LAND]),
            }
            self.seg_p_start = np.array([hx, hy, pos[2]])

        if self.state == "idle":
            self._shutdown_flight()
            return

        # --- reset del cronometro/segmento al cambio stato, ripartendo dalla
        #     posizione reale della carota (continuita') ---
        if self.state != self.prev_state:
            self.seg_t0 = time.perf_counter()
            self.seg_p_start = (self.last_p_LOS.copy() if self.last_p_LOS is not None
                                else self.WP[self.prev_state].copy())
            self.prev_state = self.state

        # --- guida a rampa trapezoidale (uguale in rising e landing) ---
        p_start = self.seg_p_start
        p_end = self.WP[self.state]
        V = self.V_LAND if self.state == "landing" else self.V_RISE
        tau = time.perf_counter() - self.seg_t0

        seg = p_end - p_start
        L = np.linalg.norm(seg)
        if L < 1e-6:
            p_LOS = p_end.copy(); v_ff = np.zeros(3)
        else:
            u = seg / L
            s_lin, v_scalar = trapz_profile(tau, L, V, self.RAMP_T)
            p_LOS = p_start + s_lin * u
            v_ff = v_scalar * u

        self.last_p_LOS = p_LOS.copy()

        # --- PID puro: landing=False => l'MPC non si attiva mai ---
        t0 = time.perf_counter()
        force, roll, pitch, yaw, mode = self.ctrl.compute(
            pos, vel, p_LOS, target_yaw=0.0, target_vel=np.zeros(3), ramp_ref_vel=v_ff,
            a_xy_lim=A_XY, final_pos=self.WP["landing"], landing=False)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        cmd = force_to_cmd(force)
        self.cf.commander.send_setpoint(rad2deg(roll), rad2deg(pitch), 0.0, cmd)

        az = force / MASS - G
        self.rows.append(dict(
            t=time.perf_counter() - self.t_start, state=self.state, mode=mode,
            x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2],
            carrot_x=p_LOS[0], carrot_y=p_LOS[1], carrot_z=p_LOS[2],
            force=force, cmd=cmd, az=az, roll=roll, pitch=pitch, solve_ms=solve_ms,
            ref_vx=v_ff[0], ref_vy=v_ff[1], ref_vz=v_ff[2]))

        print(f"[{self.state:8s}] z={pos[2]:5.2f} carrot_z={p_LOS[2]:5.2f} "
              f"vz={vel[2]:+5.2f} cmd={cmd:5d}")

        # --- transizione: distanza raggiunta E rampa completata ---
        distance = np.linalg.norm(self.WP[self.state] - pos)
        carrot_arrived = (s_lin >= L - 1e-3) if L > 1e-6 else True

        if self.state == "rising" and distance <= self.stop_delta and carrot_arrived:
            self.state = "nav"
        elif self.state == "nav" and distance <= self.stop_delta and carrot_arrived:
            self.state = "hold"
        elif self.state == "hold" and distance <= self.stop_delta and carrot_arrived:
            self.state = "landing"
        elif self.state == "landing" and pos[2] <= Z_LAND:
            self.state = "idle"
            return

    def _shutdown_flight(self):
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
        save_and_plot(self.rows)
        self.get_logger().info("Fatto. Puoi chiudere con Ctrl+C.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RisingTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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