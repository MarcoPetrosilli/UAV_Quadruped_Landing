"""
calibrate_hover_real.py — trova l'HOVER_CMD vero in closed-loop, su drone reale.

Nodo ROS 2 (a differenza della versione CrazySim, qui la posizione NON arriva
da nessun'altra parte se non dal mocap): sottoscrive /cf_drone/pose
(geometry_msgs/PoseStamped) e alimenta il Kalman filter con
cf.extpos.send_extpos(x, y, z) — SOLO posizione, mai orientazione
(send_extpose con quaternione destabilizza l'EKF, vedi note del progetto).
Senza questo il filtro deriva libero (visto nel test precedente: z=318m,
rumoroso e non monotono — nessun riferimento assoluto, solo IMU).

Logica di calibrazione (P+I con clamp anti-windup +D+slew in spazio cmd,
indipendente da force_to_cmd/MASS) identica alla versione sim gia' validata.

NOVITA' rispetto alla versione precedente:
  - Stima la massa reale (batteria+deck+marker compresi) SENZA bilancia:
    durante l'hover assestato legge motor.m1..m4 (PWM per motore, GIA'
    compensato per la tensione di batteria dal firmware), li converte in
    grammi di spinta con la curva pwm->thrust di Bitcraze e li somma. In
    hover spinta totale = peso reale, quindi quella somma e' la massa vera.
  - CMD_GUESS0 alzato (partiva troppo basso e ci metteva troppo ad arrivare
    all'hover vero, che oggi sappiamo essere ~49000-52000).
  - Guadagni PID alzati per convergere piu' in fretta (KP/KI erano tarati
    troppo prudenti per la prima calibrazione da zero).

Uso: prova PRIMA su tether/gabbia.
  ros2 run drone_landing calibrate_hover_real
(oppure "python3 calibrate_hover_real.py" standalone, vedi __main__)
"""

import time
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
MOCAP_TOPIC = '/cf_drone/pose'
DT = 0.02

# ---- parametri di calibrazione (reale: piu' cauti che in sim) -----------
TARGET_Z = 0.5             # quota di hover per il primo test [m]
DURATION = 50.0            # durata totale del volo [s]
SETTLE_AFTER = 6.0         # scarta i primi N secondi (transitorio di salita)
ERR_Z_OK = 0.03            # soglia |z - target| per considerare "assestato"
VZ_OK = 0.05               # soglia |vz| per considerare "assestato"
STATE_TIMEOUT = 0.3        # log cflib non fresco oltre questo -> STOP
MOCAP_TIMEOUT = 0.3        # mocap non fresco oltre questo -> STOP (niente
                            # riferimento assoluto, l'EKF ricomincia a derivare)
MOCAP_SETTLE_S = 2.0       # secondi di feed mocap prima del reset_estimator
EKF_CONVERGE_S = 2.0       # secondi di attesa dopo il reset perche' l'EKF agganci
EXTPOS_MIN_PERIOD = 0.02   # invia send_extpos al massimo ogni 20ms (50Hz): il
                            # mocap puo' pubblicare piu' veloce, ma il link radio
                            # e' condiviso con commander (50Hz) e log (50Hz) —
                            # mandare extpos alla piena frequenza del mocap
                            # (spesso 100-200+Hz) satura il link e fa cadere il log

# guadagni PID in SPAZIO CMD (non Newton): l'integrale trova l'hover da solo
CMD_GUESS0 = 47000.0       # prima 36000: troppo lontano dal vero hover (~49-52k
                            # misurato ripetutamente oggi), ci metteva troppo ad arrivarci
KP = 3200.0                # prima 2000: risposta piu' pronta
KI = 900.0                 # prima 300: l'integratore recupera il residuo molto piu' in fretta
KD = 2800.0                # smorzamento leggermente piu' alto per compensare guadagni piu' aggressivi
CMD_MIN, CMD_MAX = 10001, 60000
CMD_SLEW_MAX = 450.0        # prima 300: permette variazioni piu' rapide per ciclo
INTEGRAL_ERR_CLAMP = 0.356

# ---- position-hold orizzontale (mancava: roll/pitch erano sempre 0, quindi
# nessuna correzione alla deriva) — piccoli angoli, solo per tenerlo fermo
# durante la calibrazione, non e' un controllore di traiettoria
KP_XY = 1.2        # accelerazione desiderata [m/s^2] per metro di errore
KD_XY = 1.0         # accelerazione desiderata [m/s^2] per (m/s) di velocita'
MAX_TILT_DEG = 6.0  # angolo massimo di correzione, per sicurezza
G = 9.81

# ---- stima massa da motor.m1..m4 (curva pwm->thrust Bitcraze, no bilancia) ----
def pwm_to_thrust_g(pwm_16bit):
    """PWM 16 bit (gia' compensato batteria, come letto da motor.m1..m4) -> grammi
    di spinta per un singolo motore. Curva empirica Bitcraze (wiki:misc:investigations:thrust),
    valida per motori/eliche stock CF2.x — approssimazione, non sostituisce una bilancia
    ma non ne richiede una."""
    p8 = pwm_16bit / 257.0   # 16 bit -> 8 bit (0-255)
    return 0.409e-3 * p8**2 + 140.5e-3 * p8 - 0.099


def build_logconf():
    lg = LogConfig(name="state", period_in_ms=int(DT * 1000))
    for v in ("stateEstimate.z", "stateEstimate.vz",
              "stateEstimate.vx", "stateEstimate.vy",
              "motor.m1", "motor.m2", "motor.m3", "motor.m4"):
        lg.add_variable(v, "uint16_t" if v.startswith("motor.") else "float")
    return lg


def pid_step(err, vz, integral_cmd, last_cmd):
    err_i = float(np.clip(err, -INTEGRAL_ERR_CLAMP, INTEGRAL_ERR_CLAMP))
    integral_cmd += KI * err_i * DT
    integral_cmd = float(np.clip(integral_cmd, CMD_MIN, CMD_MAX))

    cmd_raw = integral_cmd + KP * err - KD * vz
    cmd_raw = float(np.clip(cmd_raw, CMD_MIN, CMD_MAX))

    delta = np.clip(cmd_raw - last_cmd, -CMD_SLEW_MAX, CMD_SLEW_MAX)
    cmd = int(np.clip(last_cmd + delta, CMD_MIN, CMD_MAX))
    return cmd, integral_cmd


def xy_hold_attitude(ex, ey, vx, vy):
    """Piccola correzione di assetto per tenere fermo x,y. Stessa convenzione
    di main_rising_test.py: pitch positivo -> +x, roll positivo -> -y."""
    ax_des = KP_XY * ex - KD_XY * vx
    ay_des = KP_XY * ey - KD_XY * vy
    pitch = np.degrees(np.arctan2(ax_des, G))
    roll = np.degrees(np.arctan2(-ay_des, G))
    pitch = float(np.clip(pitch, -MAX_TILT_DEG, MAX_TILT_DEG))
    roll = float(np.clip(roll, -MAX_TILT_DEG, MAX_TILT_DEG))
    return roll, pitch


class CalibrateHoverRealNode(Node):

    def __init__(self):
        super().__init__("calibrate_hover_real_node")

        cflib.crtp.init_drivers()
        self.get_logger().info(f"Connessione a {URI} ...")
        self._scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache"))
        self._scf.open_link()
        self.cf = self._scf.cf

        self.cf.supervisor.send_arming_request(True); time.sleep(1.0)

        # estimatore Kalman esplicito prima di volare (non lasciarlo al default)
        self.cf.param.set_value('stabilizer.estimator', '2')
        time.sleep(0.2)

        for _ in range(10):
            self.cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)

        # --- mocap: sottoscrizione + feed continuo (rate-limited) al Kalman ---
        self.last_mocap = None          # (x,y,z)
        self.last_mocap_wall_t = None
        self.last_extpos_sent_wall_t = None
        self.create_subscription(PoseStamped, MOCAP_TOPIC, self._on_mocap, 50)

        self.get_logger().info(f"In attesa di dati mocap su {MOCAP_TOPIC} ...")
        t_wait0 = time.perf_counter()
        while self.last_mocap is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.perf_counter() - t_wait0 > 5.0:
                self.get_logger().error("Nessun dato mocap ricevuto in 5s, abort.")
                self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
                self._scf.close_link()
                raise SystemExit(1)

        # dai tempo al Kalman di "vedere" un po' di mocap prima del reset
        self.get_logger().info("Alimento il Kalman con mocap prima del reset...")
        t_settle0 = time.perf_counter()
        while time.perf_counter() - t_settle0 < MOCAP_SETTLE_S:
            rclpy.spin_once(self, timeout_sec=DT)

        self.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        t_conv0 = time.perf_counter()
        while time.perf_counter() - t_conv0 < EKF_CONVERGE_S:
            rclpy.spin_once(self, timeout_sec=DT)

        # --- log cflib (stato fuso, usato per il controllo) ---
        self.latest_state = None
        self.last_state_wall_t = None
        self._logconf = build_logconf()
        self.cf.log.add_config(self._logconf)
        self._logconf.data_received_cb.add_callback(self._on_state)
        self._logconf.start()
        time.sleep(0.3)

        if self.latest_state is not None:
            self.get_logger().info(
                f"Sanity check: mocap z={self.last_mocap[2]:.3f}  "
                f"stateEstimate.z={self.latest_state['stateEstimate.z']:.3f}")

        # --- stato calibrazione ---
        self.integral_cmd = CMD_GUESS0
        self.last_cmd = CMD_GUESS0
        self.samples = []          # (t, z, vz, cmd, m1, m2, m3, m4)
        self.t_start = time.perf_counter()
        self.phase = "hover"
        self.land_t0 = None
        self._finished = False
        self._aborted = False
        self.home_xy = (self.last_mocap[0], self.last_mocap[1])
        self.get_logger().info(f"Posizione di riferimento xy: {self.home_xy}")

        self.timer = self.create_timer(DT, self.tick)
        self.get_logger().info("Calibrazione in corso...")

    def _on_mocap(self, msg: PoseStamped):
        p = msg.pose.position
        self.last_mocap = (p.x, p.y, p.z)
        self.last_mocap_wall_t = time.perf_counter()   # freschezza: aggiornata SEMPRE

        # invio a valle rate-limited, per non saturare il link radio
        now = self.last_mocap_wall_t
        if self.last_extpos_sent_wall_t is None or (now - self.last_extpos_sent_wall_t) >= EXTPOS_MIN_PERIOD:
            self.cf.extpos.send_extpos(p.x, p.y, p.z)   # SOLO posizione, mai quaternione
            self.last_extpos_sent_wall_t = now

    def _on_state(self, timestamp, data, logconf):
        self.latest_state = data
        self.last_state_wall_t = time.perf_counter()

    def _watchdog_ok(self):
        now = time.perf_counter()
        if self.last_mocap_wall_t is None or (now - self.last_mocap_wall_t) > MOCAP_TIMEOUT:
            self._abort("mocap non fresco")
            return False
        if self.last_state_wall_t is None or (now - self.last_state_wall_t) > STATE_TIMEOUT:
            self._abort("log cflib non fresco")
            return False
        return True

    def _abort(self, reason):
        self.get_logger().error(f"!!! STOP DI EMERGENZA: {reason} !!!")
        self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
        time.sleep(0.05)
        self.cf.commander.send_stop_setpoint()
        self._aborted = True
        self._finished = True
        self._logconf.stop()
        self._scf.close_link()
        self._report()
        self.timer.cancel()
        if rclpy.ok():
            rclpy.shutdown()

    def tick(self):
        if self._finished:
            return
        if not self._watchdog_ok():
            return
        if self.latest_state is None:
            return

        z = self.latest_state["stateEstimate.z"]
        vz = self.latest_state["stateEstimate.vz"]
        t = time.perf_counter() - self.t_start

        if self.phase == "hover":
            target = TARGET_Z
            if t > DURATION:
                self.phase = "land"
                self.land_t0 = time.perf_counter()
        elif self.phase == "land":
            t_land = time.perf_counter() - self.land_t0
            if t_land > 4.0:
                self._shutdown()
                return
            target = max(0.05, TARGET_Z * (1.0 - t_land / 4.0))
        else:
            return

        err = target - z
        cmd, self.integral_cmd = pid_step(err, vz, self.integral_cmd, self.last_cmd)
        self.last_cmd = cmd

        ex = self.home_xy[0] - self.last_mocap[0]
        ey = self.home_xy[1] - self.last_mocap[1]
        vx = self.latest_state.get("stateEstimate.vx", 0.0)
        vy = self.latest_state.get("stateEstimate.vy", 0.0)
        roll, pitch = xy_hold_attitude(ex, ey, vx, vy)

        self.cf.commander.send_setpoint(roll, pitch, 0.0, cmd)

        if self.phase == "hover":
            m1 = self.latest_state.get("motor.m1", 0)
            m2 = self.latest_state.get("motor.m2", 0)
            m3 = self.latest_state.get("motor.m3", 0)
            m4 = self.latest_state.get("motor.m4", 0)
            self.samples.append((t, z, vz, cmd, m1, m2, m3, m4))
            if int(t) != int(t - DT):
                self.get_logger().info(f"t={t:5.1f}s  z={z:5.2f}  vz={vz:+5.2f}  cmd={cmd}")

    def _shutdown(self):
        self._finished = True
        self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
        time.sleep(0.1)
        self.cf.commander.send_stop_setpoint()
        self._logconf.stop()
        self._scf.close_link()
        self._report()
        self.timer.cancel()
        if rclpy.ok():
            rclpy.shutdown()

    def _report(self):
        settled = [row for row in self.samples
                   if row[0] > SETTLE_AFTER and abs(row[1] - TARGET_Z) < ERR_Z_OK and abs(row[2]) < VZ_OK]
        self.get_logger().info("=" * 50)
        if self._aborted:
            self.get_logger().info("Volo abortito: nessun risultato di calibrazione affidabile.")
        elif len(settled) >= 20:
            cmds = [row[3] for row in settled]
            mean_cmd = float(np.mean(cmds))
            std_cmd = float(np.std(cmds))
            self.get_logger().info(
                f"HOVER_CMD calibrato ~= {mean_cmd:.0f}  (std={std_cmd:.1f}, n={len(settled)})")
            self.get_logger().info("Metti questo valore in HOVER_CMD dentro main_alpha_landing.py")

            # stima massa reale dai motori (senza bilancia): spinta totale in
            # hover = peso reale (batteria + deck + marker compresi)
            thrust_g_samples = [
                sum(pwm_to_thrust_g(pwm) for pwm in row[4:8]) for row in settled
            ]
            mean_g = float(np.mean(thrust_g_samples))
            std_g = float(np.std(thrust_g_samples))
            self.get_logger().info(
                f"Massa reale stimata (da motor.m1-m4, no bilancia) ~= {mean_g:.1f} g "
                f"(std={std_g:.1f} g, n={len(thrust_g_samples)})")
            self.get_logger().info(
                f"Metti questo valore in MASS dentro main_rising_test.py: {mean_g/1000.0:.4f}")
        else:
            self.get_logger().warn(
                f"Pochi campioni assestati ({len(settled)}) - aumenta DURATION o rivedi KP/KI.")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)
    node = CalibrateHoverRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node._abort("interrotto dall'utente (Ctrl+C)")
        except Exception as e:
            print(f"(shutdown gia' in corso, ignoro: {e})")


if __name__ == "__main__":
    main()
