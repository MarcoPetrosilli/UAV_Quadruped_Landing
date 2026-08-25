"""
main_crazysim.py  —  i blocchi "Main" + "SyncLogger" del tuo schema.

Loop:  SyncLogger legge (pos, vel) da CrazySim via cflib  ->  Controller  ->
       send_setpoint(roll, pitch, yawrate, thrust)  ->  CrazySim/MuJoCo.

Il SyncLogger fa da CLOCK del loop: ad ogni pacchetto di log (ogni DT) leggi lo
stato, calcoli, mandi il setpoint. E' esattamente la freccia "read-state" del
disegno.

Piattaforma mobile IGNORATA: target statico, target_vel = 0.

Prerequisiti:
  - env (drones) attivo, CrazySim su (una sola istanza), reachable_polytope.npz
    nella cartella da cui lanci questo script;
  - lancia con:  python main_crazysim.py
"""

import time
import math
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

from controller_deploy import HybridController

URI = uri_helper.uri_from_env(default='udp://127.0.0.1:19850')   # radio://... per HW
DT = 0.02                                                        # = dt del controllore

# ---- calibrazione spinta: forza [N] -> comando thrust uint16 --------------
# !!! DA TARARE !!! Hai visto che 38000 SALE, quindi l'hover e' sotto: parti da
# ~30000 e alza/abbassa finche' in RISING il drone tiene la quota HOVER.
HOVER_CMD = 38000
HOVER_FORCE = 0.0379 * 9.81           # forza che il controllore emette all'hover


def force_to_cmd(force_N):
    return int(np.clip(HOVER_CMD * force_N / HOVER_FORCE, 10001, 60000))


def rad2deg(x):
    return x * 180.0 / math.pi


# ---- missione minima (statica) --------------------------------------------
HOVER = np.array([0.0, 0.0, 3.0])    # punto di hover
LAND = np.array([2.0, 2.0, 3.0])     # target di atterraggio (statico)


def build_logconf():
    lg = LogConfig(name="state", period_in_ms=int(DT * 1000))
    for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
              "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"):
        lg.add_variable(v, "float")
    return lg


def reset_estimator(cf):
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(1.5)                  # lascia convergere la stima


def main():
    cflib.crtp.init_drivers()
    ctrl = HybridController(dt=DT)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        reset_estimator(cf)

        # sblocco anti-flyaway: primo setpoint a thrust 0
        for _ in range(10):
            cf.commander.send_setpoint(0.0, 0.0, 0, 0)
            time.sleep(DT)

        phase = "RISING"
        try:
            with SyncLogger(scf, build_logconf()) as logger:
                for _ts, data, _ in logger:
                    pos = np.array([data["stateEstimate.x"],
                                    data["stateEstimate.y"],
                                    data["stateEstimate.z"]])
                    vel = np.array([data["stateEstimate.vx"],
                                    data["stateEstimate.vy"],
                                    data["stateEstimate.vz"]])
                    print("Position")
                    print(pos)
                    print("Velocity")
                    print(vel)

                    # --- FSM di missione (statica, piattaforma ignorata) ---
                    if phase == "RISING":
                    #if True:
                        target, final, landing = HOVER, None, False
                        if np.linalg.norm(pos - HOVER) < 0.15 and np.linalg.norm(vel) < 0.2:
                            phase = "LANDING"
                    else:  # LANDING
                        target, final, landing = LAND, LAND, True

                    # --- Controller ---
                    force, roll, pitch, yaw, mode = ctrl.compute(
                        pos, vel, target, target_yaw=0.0,
                        a_xy_lim=0.17, final_pos=final, landing=landing)

                    # --- setpoint verso CrazySim ---
                    # NB: roll/pitch in GRADI; se il drone si inclina al
                    # contrario, INVERTI il segno (convenzioni sim vs cflib).
                    cf.commander.send_setpoint(
                        rad2deg(roll), rad2deg(pitch), 0.0, force_to_cmd(force))

                    # --- fine: atterrato ---
                    if phase == "LANDING" and pos[2] < 0.2 and abs(vel[2]) < 0.1:
                        break
        finally:
            for _ in range(20):
                cf.commander.send_setpoint(0.0, 0.0, 0, 0)
                time.sleep(DT)
            cf.commander.send_stop_setpoint()


if __name__ == "__main__":
    main()