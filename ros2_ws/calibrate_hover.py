"""
calibrate_hover.py — trova l'HOVER_CMD vero in closed-loop.

Idea: un PI verticale che lavora DIRETTAMENTE in unita' di cmd (non passa mai
da force_to_cmd/MASS), quindi non eredita nessun bias di calibrazione. Il
termine integrale, all'equilibrio (z stabile, vz~0), converge esattamente al
comando che tiene il drone in hover — quello e' il tuo HOVER_CMD vero.

Uso: lancialo da solo (stessa URI/env di main_alpha_landing.py), aspetta che
salga a TARGET_Z e si stabilizzi, leggi il valore stampato, mettilo in
HOVER_CMD dentro main_alpha_landing.py.
"""

import time
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='udp://127.0.0.1:19850')
DT = 0.02

# ---- parametri di calibrazione ------------------------------------------
TARGET_Z = 0.6          # quota di hover per la calibrazione [m]
DURATION = 40.0          # durata totale del volo [s]
SETTLE_AFTER = 8.0       # scarta i primi N secondi (transitorio di salita)
ERR_Z_OK = 0.03           # soglia |z - target| per considerare "assestato"
VZ_OK = 0.05              # soglia |vz| per considerare "assestato"

# guadagni PID in SPAZIO CMD (non Newton): l'integrale trova l'hover da solo
CMD_GUESS0 = 30000.0      # punto di partenza dell'integratore (piu' vicino al vero
                          # hover: accorcia lo stallo a terra e il relativo windup)
KP = 2500.0               # cmd per metro di errore (ridotto: con 6000 oscillava)
KI = 400.0                # cmd per (metro*secondo) integrato (ridotto: accumulava troppo)
KD = 2500.0               # cmd per (metro/secondo) di vz — smorzamento, mancava del tutto
CMD_MIN, CMD_MAX = 10001, 60000
CMD_SLEW_MAX = 400.0      # variazione massima di cmd per ciclo (smorza i salti bruschi)
INTEGRAL_ERR_CLAMP = 0.35 # l'integrale "vede" al massimo questo errore (in valore
                          # assoluto): cresce sempre, ma a ritmo limitato quando sei
                          # ancora lontano dal target (es. fermo a terra a inizio volo)


def reset_estimator(cf):
    cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.5)


def build_logconf():
    lg = LogConfig(name="state", period_in_ms=int(DT * 1000))
    for v in ("stateEstimate.z", "stateEstimate.vz"):
        lg.add_variable(v, "float")
    return lg


def pid_step(err, vz, integral_cmd, last_cmd):
    """Un passo di P + I(anti-windup a clamp) + D(su vz) + slew-rate limiter.
    Ritorna (cmd, nuovo integral_cmd)."""
    err_i = float(np.clip(err, -INTEGRAL_ERR_CLAMP, INTEGRAL_ERR_CLAMP))
    integral_cmd += KI * err_i * DT
    integral_cmd = float(np.clip(integral_cmd, CMD_MIN, CMD_MAX))

    cmd_raw = integral_cmd + KP * err - KD * vz
    cmd_raw = float(np.clip(cmd_raw, CMD_MIN, CMD_MAX))

    # rate limiter: non lasciare che il comando salti troppo da un ciclo all'altro
    delta = np.clip(cmd_raw - last_cmd, -CMD_SLEW_MAX, CMD_SLEW_MAX)
    cmd = int(np.clip(last_cmd + delta, CMD_MIN, CMD_MAX))
    return cmd, integral_cmd


def main():
    cflib.crtp.init_drivers()
    print(f"Connessione a {URI} ...")
    scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache"))
    scf.open_link()
    cf = scf.cf

    cf.supervisor.send_arming_request(True); time.sleep(1.0)
    reset_estimator(cf)
    for _ in range(10):
        cf.commander.send_setpoint(0.0, 0.0, 0, 0); time.sleep(DT)

    latest = {}
    logconf = build_logconf()
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(lambda ts, data, lc: latest.update(data))
    logconf.start()
    time.sleep(0.3)  # aspetta il primo pacchetto

    integral_cmd = CMD_GUESS0
    last_cmd = CMD_GUESS0
    samples = []  # (t, z, vz, cmd)
    t0 = time.perf_counter()

    print("Calibrazione in corso... (Ctrl+C per interrompere in sicurezza)")
    try:
        while True:
            t = time.perf_counter() - t0
            if t > DURATION:
                break
            if "stateEstimate.z" not in latest:
                time.sleep(DT); continue

            z = latest["stateEstimate.z"]
            vz = latest["stateEstimate.vz"]
            err = TARGET_Z - z

            cmd, integral_cmd = pid_step(err, vz, integral_cmd, last_cmd)
            last_cmd = cmd

            cf.commander.send_setpoint(0.0, 0.0, 0.0, cmd)
            samples.append((t, z, vz, cmd))

            if t % 1.0 < DT:
                print(f"t={t:5.1f}s  z={z:5.2f}  vz={vz:+5.2f}  cmd={cmd}")

            time.sleep(DT)
    except KeyboardInterrupt:
        print("Interrotto dall'utente.")

    # --- atterraggio morbido: abbassa il target gradualmente ---
    print("Atterraggio...")
    land_t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - land_t0
        if t > 4.0 or "stateEstimate.z" not in latest:
            break
        z = latest["stateEstimate.z"]
        vz = latest["stateEstimate.vz"]
        target = max(0.05, TARGET_Z * (1.0 - t / 4.0))
        err = target - z
        cmd, integral_cmd = pid_step(err, vz, integral_cmd, last_cmd)
        last_cmd = cmd
        cf.commander.send_setpoint(0.0, 0.0, 0.0, cmd)
        time.sleep(DT)

    cf.commander.send_setpoint(0.0, 0.0, 0, 0)
    time.sleep(0.1)
    cf.commander.send_stop_setpoint()
    logconf.stop()
    scf.close_link()

    # --- risultato: media del cmd nella finestra "assestata" ---
    settled = [c for (t, z, vz, c) in samples
               if t > SETTLE_AFTER and abs(z - TARGET_Z) < ERR_Z_OK and abs(vz) < VZ_OK]

    print("\n" + "=" * 50)
    if len(settled) >= 20:
        mean_cmd = float(np.mean(settled))
        std_cmd = float(np.std(settled))
        print(f"HOVER_CMD calibrato ~= {mean_cmd:.0f}  (std={std_cmd:.1f}, n={len(settled)} campioni)")
        print(f"Metti questo valore in HOVER_CMD dentro main_alpha_landing.py")
    else:
        print(f"Pochi campioni assestati ({len(settled)}) - il volo non si e' "
              f"stabilizzato abbastanza. Aumenta DURATION o rivedi KP/KI.")
    print("=" * 50)


if __name__ == "__main__":
    main()
