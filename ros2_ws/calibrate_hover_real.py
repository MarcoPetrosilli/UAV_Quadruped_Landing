"""
calibrate_hover_real.py — trova l'HOVER_CMD vero in closed-loop, su drone reale.

Stessa idea/logica di calibrate_hover.py (PI+D in spazio cmd, indipendente da
force_to_cmd/MASS). Cambia quello che deve cambiare per il volo reale:

  - URI: radio invece di UDP loopback (CrazySim).
  - WATCHDOG sul link: su UDP loopback un pacchetto perso e' innocuo (localhost),
    su radio reale no — se non arriva stato fresco per STATE_TIMEOUT secondi,
    taglio motori immediato, non aspetto la fine del programma.
  - Ctrl+C = STOP immediato (motori a zero), non piu' la rampa di atterraggio
    morbida: se stai interrompendo a mano sul reale probabilmente e' perche'
    qualcosa non va, meglio tagliare che continuare a volare 4s in piu'.
  - Guadagni e punto di partenza piu' cauti: la dinamica reale (attrito,
    ground effect, thrust curve del firmware) e' meno prevedibile del
    simulatore, quindi si parte piu' in basso e piu' piano.
  - TARGET_Z piu' basso e DURATION piu' corta per il primo test: alza dopo
    aver visto che si comporta bene.

Uso: prova questa PRIMA su tether/gabbia. Aspetta che salga a TARGET_Z e si
stabilizzi, leggi HOVER_CMD stampato, mettilo in HOVER_CMD dentro
main_alpha_landing.py (versione reale).
"""

import time
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DT = 0.02

# ---- parametri di calibrazione (reale: piu' cauti che in sim) -----------
TARGET_Z = 0.5             # quota di hover per il primo test [m] (piu' bassa che in sim)
DURATION = 15.0            # durata totale del volo [s] (piu' corta per il primo test)
SETTLE_AFTER = 6.0         # scarta i primi N secondi (transitorio di salita)
ERR_Z_OK = 0.03            # soglia |z - target| per considerare "assestato"
VZ_OK = 0.05               # soglia |vz| per considerare "assestato"
STATE_TIMEOUT = 0.3        # se non arriva stato fresco entro questo tempo -> STOP

# guadagni PID in SPAZIO CMD (non Newton): l'integrale trova l'hover da solo
CMD_GUESS0 = 25000.0       # parte piu' basso che in sim: preferisco un decollo
                          # lento e prevedibile a uno scatto su hardware vero
KP = 2000.0                # cmd per metro di errore (leggermente piu' basso che in sim)
KI = 300.0                 # cmd per (metro*secondo) integrato
KD = 2500.0                # cmd per (metro/secondo) di vz — smorzamento
CMD_MIN, CMD_MAX = 10001, 45000   # tetto piu' basso che in sim (45000 non 60000):
                                   # limite di sicurezza per il primo test calibrazione
CMD_SLEW_MAX = 300.0        # variazione massima di cmd per ciclo (piu' stretta che in sim)
INTEGRAL_ERR_CLAMP = 0.35   # l'integrale "vede" al massimo questo errore: cresce
                            # sempre, ma a ritmo limitato quando sei lontano dal target


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

    delta = np.clip(cmd_raw - last_cmd, -CMD_SLEW_MAX, CMD_SLEW_MAX)
    cmd = int(np.clip(last_cmd + delta, CMD_MIN, CMD_MAX))
    return cmd, integral_cmd


def emergency_stop(cf, reason):
    print(f"\n!!! STOP DI EMERGENZA: {reason} !!!")
    cf.commander.send_setpoint(0.0, 0.0, 0, 0)
    time.sleep(0.05)
    cf.commander.send_stop_setpoint()


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
    last_state_wall_t = [None]  # timestamp locale dell'ultimo pacchetto ricevuto

    def on_state(ts, data, lc):
        latest.update(data)
        last_state_wall_t[0] = time.perf_counter()

    logconf = build_logconf()
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(on_state)
    logconf.start()
    time.sleep(0.3)  # aspetta il primo pacchetto

    if last_state_wall_t[0] is None:
        print("Nessuno stato ricevuto dal link radio, abort.")
        cf.commander.send_setpoint(0.0, 0.0, 0, 0)
        scf.close_link()
        return

    integral_cmd = CMD_GUESS0
    last_cmd = CMD_GUESS0
    samples = []  # (t, z, vz, cmd)
    t0 = time.perf_counter()
    aborted = False

    print("Calibrazione in corso... (Ctrl+C = STOP immediato)")
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if t > DURATION:
                break

            # --- watchdog link: stato non fresco -> stop, non continuare al buio ---
            if last_state_wall_t[0] is None or (now - last_state_wall_t[0]) > STATE_TIMEOUT:
                emergency_stop(cf, "nessuno stato fresco dal link radio")
                aborted = True
                break

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
        # sul reale: interruzione manuale = STOP subito, niente rampa morbida
        emergency_stop(cf, "interrotto dall'utente (Ctrl+C)")
        aborted = True

    if not aborted:
        # --- atterraggio morbido: abbassa il target gradualmente ---
        print("Atterraggio...")
        land_t0 = time.perf_counter()
        while True:
            now = time.perf_counter()
            t = now - land_t0
            if t > 4.0:
                break
            if last_state_wall_t[0] is None or (now - last_state_wall_t[0]) > STATE_TIMEOUT:
                emergency_stop(cf, "link perso durante l'atterraggio")
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
    if aborted:
        print("Volo interrotto/abortito: nessun risultato di calibrazione affidabile.")
    elif len(settled) >= 20:
        mean_cmd = float(np.mean(settled))
        std_cmd = float(np.std(settled))
        print(f"HOVER_CMD calibrato ~= {mean_cmd:.0f}  (std={std_cmd:.1f}, n={len(settled)} campioni)")
        print(f"Metti questo valore in HOVER_CMD dentro main_alpha_landing.py (versione reale)")
    else:
        print(f"Pochi campioni assestati ({len(settled)}) - il volo non si e' "
              f"stabilizzato abbastanza. Aumenta DURATION o rivedi KP/KI.")
    print("=" * 50)


if __name__ == "__main__":
    main()
