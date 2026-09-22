"""
main_rising_test.py  —  IDENTICO a main_alpha_landing.py, unica differenza:
la posizione arriva dal mocap (/cf_drone/pose) invece che dal solo stimatore
onboard — quindi qui c'e' in piu' la sottoscrizione mocap (send_extpos verso
il Kalman filter) e l'attesa/reset iniziale del filtro prima di partire.
Tutto il resto (FSM, guida, controllore, plotting) e' lo stesso file.
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
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped   # topic mocap /cf_drone/pose

try:
    from drone_landing.controller_deploy import HybridController   # dentro package ROS
except ImportError:
    from controller_deploy import HybridController                 # standalone

URI = uri_helper.uri_from_env(default='radio://0/80/2M')
MOCAP_TOPIC = "/cf_drone/pose"

# ---- piattaforma dal mocap ----------------------------------------------------
# Il target di atterraggio (TARGET_XY e Z_LAND) viene preso dalla posa della
# piattaforma, letta PRIMA del decollo e poi tenuta fissa (piattaforma ferma).
#   quota superficie   = z_piattaforma + PLATFORM_TOP_DZ
#   Z_LAND             = superficie + quota del drone appoggiato + LAND_CLEARANCE
#   suolo per l'MPC    = superficie + quota del drone appoggiato
# "quota del drone appoggiato" e' la z mocap del drone fermo sul pavimento
# prima del decollo (pavimento a z = FLOOR_Z): e' l'offset tra l'origine del
# corpo rigido del drone e la superficie su cui poggia, e vale anche sulla
# piattaforma. LAND_CLEARANCE = 0.015 riproduce il vecchio Z_LAND = 0.05 con
# drone a 0.035 sul pavimento.
USE_PLATFORM     = False
PLATFORM_TOPIC   = "/platform/pose"
PLATFORM_TOP_DZ  = 0.0     # [m] superficie di appoggio MENO origine del corpo
                           # rigido della piattaforma: MISURALO (positivo se la
                           # superficie sta sopra l'origine dei marker)
LAND_CLEARANCE   = 0.015   # [m] target sopra la quota di appoggio
FLOOR_Z          = 0.0     # [m] quota del pavimento nel frame mocap
PLATFORM_WAIT_T  = 5.0     # [s] attesa massima della prima posa piattaforma
PLATFORM_AVG_N   = 50      # campioni mediati per fissare il target (~0.5 s a 100 Hz)
DT = 0.02
G = 9.81

# ---- calibrazione spinta -----------------------------------------------------
# HOVER_CMD parte da HOVER_CMD_INIT e, con HOVER_ADAPT attivo, dopo l'aggancio
# insegue la stima scelta con HOVER_SOURCE (rate limit HOVER_APPLY_RATE):
#   "kf"  : filtro di Kalman a 1 stato (principale, vedi sotto)
#   "b50" : aggancio da fermo + passa-basso/rate limit (versione precedente)
# Le altre stime (B50, stimatore attuale, modello batteria) restano in log.
# Motivo (voli del 16/9, HOVER fisso): l'hover vero cambia del +-3-5% tra un
# volo e l'altro senza cause osservabili prima del decollo; con un valore fisso
# il landing va dal plateau (HOVER troppo alto, basta +1.7%) al touchdown duro.
# HOVER_ADAPT = False riporta al comportamento di caratterizzazione (fisso).
HOVER_CMD_INIT = 39200     # valore usato fino all'aggancio B50 (salita e primi secondi)
HOVER_CMD = HOVER_CMD_INIT # valore APPLICATO, aggiornato a runtime (globale)
HOVER_ADAPT = True
HOVER_SOURCE = "kf"        # "kf" oppure "b50"
# Margine in landing: l'MPC verticale non ha integratore, e con HOVER anche solo
# lo 0.3% sopra il vero si ferma sopra il target (plateau, in simulazione a
# ~6 cm) e il volo non termina. In landing si applica quindi la stima ridotta
# di questa frazione: il drone scende sempre, e i voli del 16/9 con HOVER
# 1-2% sotto il vero hanno toccato terra a |vz| ~0.03-0.05 m/s.
# 0.0 = nessun margine (stima pura).
HOVER_LAND_MARGIN = 0.01  # l'effetto suolo e' ora nel modello dell'MPC
                           # (controller_deploy.ge_*), questo copre solo
                           # l'errore residuo della stima
HOVER_APPLY_RATE = 1000.0  # variazione massima del valore applicato [unita'/s]:
                           # all'aggancio (~7.6 s dalla salita) lo scarto puo'
                           # arrivare a ~5.5k (16/9); a 1000 u/s si chiude in
                           # ~5.5 s, prima del landing (~16.5 s), senza gradino
                           # di spinta in navigazione
MASS = 0.0379
#MASS = 0.04
HOVER_FORCE = MASS * G

# ---- stimatori dell'hover (SOLO LOG) ------------------------------------------
# Misura diretta, in anello aperto:
#
#     acc_z = T/(m*g)        [g]          (forza specifica, ~1 in hover)
#     cmd   = T * HOVER_vero/HOVER_FORCE
#  => HOVER_vero = cmd / acc_z
#
# Girano in parallelo due stimatori con lo STESSO filtro (passa-basso + rate
# limit) e aggancio diverso:
#   - "cont": quello usato finora, aggancio sul primo campione valido sopra
#     HOVER_EST_Z_MIN (cioe' in piena accelerazione di salita);
#   - "b50":  aggancio sulla mediana dei primi B50_N campioni validi raccolti
#     da fermo (|vz| < B50_VZ_MAX, z > B50_Z_MIN, fuori dal landing), come
#     verificato offline sui 26 voli del 14/9 (bias da +1364 a -38).
# In piu' si logga il modello su batteria agganciato al valore b50:
#     hover_vbat = h_b50 + VBAT_SLOPE * (vbat - vbat_all_aggancio)
HOVER_EST_TAU     = 1.5      # costante del passa-basso [s]
HOVER_EST_RATE    = 500.0    # variazione massima della stima [unita'/s]
HOVER_EST_Z_MIN   = 0.4      # sotto questa quota non stimare: in effetto suolo
                              # la spinta e' ~9.5% maggiore (misurato sui run
                              # del 14/9) e la stima verrebbe falsata verso il basso
HOVER_EST_ACC_MIN = 0.5      # finestra di plausibilita' su acc.z [g]: fuori di
HOVER_EST_ACC_MAX = 1.6      # qui e' vibrazione o campione sporco, si scarta
HOVER_EST_CMD_MIN = 30000    # clamp di sicurezza sulla stima (il drone verde
                             # ha hover ~39k: con 40000 la stima restava bloccata)
HOVER_EST_CMD_MAX = 58000

B50_N      = 50              # campioni per l'aggancio da fermo (~1 s)
B50_VZ_MAX = 0.1             # |vz| massimo per considerare il drone fermo [m/s]
B50_Z_MIN  = 0.40            # quota minima per l'aggancio [m]
# ---- filtro di Kalman a 1 stato sull'hover (come mc_hover_thrust_estimator di PX4)
#   stato   theta = 1/h                 random walk, varianza q per passo
#   misura  acc.z = cmd * theta + v     v ~ N(0, R), lineare in theta
# Inizializzato all'aggancio B50: theta0 = mediana, P0 = var(campioni)/N,
# R = var(acc.z) negli stessi campioni, q scelto per avere a regime la
# costante di tempo KF_TAU:  tau = DT/(h*sqrt(q/R))  ->  q = R*(DT/(tau*h))^2.
# Offline sui 21 voli del 16/9: errore a inizio landing mediana 37, max 426
# (B50: 44 / 813); converge in ~1 s quando l'hover cambia dopo l'aggancio.
KF_TAU     = 3.0           # costante di tempo a regime [s]
KF_GATE    = 3.0           # misura scartata se |innovazione| > KF_GATE*sigma
KF_REC_N   = 50            # dopo tanti scarti consecutivi (~1 s) ...
KF_REC_P   = 10.0          # ... P viene moltiplicata per questo (recupero)

VBAT_SLOPE = 11221.0         # pendenza media hover/vbat sui 26 voli del 14/9 [u/V]

# ---- avvio motori: pre-spin + rampa di spunto --------------------------------
# Il comando passava da 0 a ~hover in un solo tick, saturando a 60000 e
# affondando vbat. Rimedio in due fasi, entrambe in ANELLO APERTO: il
# controllore non entra in gioco finche' il drone non e' staccato da terra.
#
#   1) PRE-SPIN: motori a PRESPIN_CMD, drone fermo a terra. La spinta va circa
#      col quadrato del PWM, quindi qui si resta ben sotto il peso: i rotori
#      accumulano inerzia senza staccarsi. Il gradino c'e' ancora, ma e' messo
#      dove la corrente e' bassa.
#   2) SPOOL: il comando sale a SPOOL_RATE unita'/s FINCHE' IL DRONE NON SI
#      STACCA (z > z_suolo + LIFTOFF_DZ). Nessun estremo prefissato: la rampa
#      non puo' chiudersi sotto l'hover vero, che e' ignoto per costruzione.
#
# Perche' non un tetto sull'uscita del PID (versione precedente, run del 14/9):
# la rampa chiudeva su HOVER_CMD = 47000 mentre l'hover vero era 52146, quindi
# il drone e' rimasto a terra per tutta la rampa mentre la carota saliva di
# 20 cm. Al rilascio il PID chiedeva gia' 60000, cioe' 13419 sopra il tetto:
# un gradino peggiore di quello che la rampa doveva togliere.
#
# In anello aperto il problema sparisce: al momento del passaggio di consegne
# il comando vale ~l'hover vero (e' quello che ha appena staccato il drone) e
# la carota parte da dove il drone si trova, quindi l'errore di posizione e'
# nullo e non c'e' niente di caricato da scaricare.
PRESPIN_ENABLED = True
PRESPIN_CMD   = 20000     # comando di pre-spin [unita' PWM]
PRESPIN_T     = 0.4       # durata del pre-spin [s] (tau di spin-up ~50-100 ms)
SPOOL_RATE    = 22000.0   # pendenza della rampa [unita'/s]
SPOOL_CMD_MAX = 58000     # tetto di sicurezza della rampa
SPOOL_T_MAX   = 3.5       # se non si stacca entro questo -> taglio motori
# Il distacco si rileva sulla VELOCITA' verticale, non sullo spostamento: per
# salire di 4 cm il drone deve prima accelerare, e a quel punto la rampa ha gia'
# superato l'hover vero di ~8500 unita'. Su vz la soglia scatta circa 0.23 s
# prima, cioe' ~3400 unita' di scarto, che il termine D del PID assorbe da solo.
# La soglia sullo spostamento resta come rete di sicurezza se vz fosse sporca.
LIFTOFF_VZ    = 0.05      # velocita' di salita per dichiarare il distacco [m/s]
LIFTOFF_DZ    = 0.04      # oppure salita sopra il suolo [m]


def force_to_cmd(force_N):
    return int(np.clip(HOVER_CMD * force_N / HOVER_FORCE, 10001, 60000))


def rad2deg(x):
    return x * 180.0 / math.pi


# ---- watchdog posa mocap ------------------------------------------------------
# Se la posa non si aggiorna da POSE_TIMEOUT, il controllo in anello chiuso si
# ferma (voli 163915 e 183415: posa congelata, drone guidato alla cieca).
POSE_TIMEOUT       = 0.25   # eta' massima della posa [s]
POSE_LOST_CUT_H    = 0.30   # sotto questa quota sul suolo: taglio motori [m]
POSE_LOST_VZ       = -0.3   # piu' in alto: discesa livellata a questa vz [m/s],
POSE_LOST_KV       = 2.0    # regolata sulla vz del Kalman (solo IMU) [1/s]
POSE_LOST_T_MARGIN = 1.5    # durata = quota/|vz| + margine, poi taglio [s]

# ---- missione (dinamica, percorso a L con piattaforma mobile) ----------------
TARGET_XY = np.array([0.0, -1.5])       # target parte qui
TARGET_VEL = np.array([0.0, 0.3, 0.0]) # si muove in +y (0.15 m/s; regola tu)
Z_CRUISE = 1.8
Z_LAND = 0.05
Z_HOLD = 1.8
ALPHA_CONE = 1.0
LOS_DELTA = 0.3
A_XY = 0.17
R_BASE_CONE = 0.3   
IDLE, RISING, NAV, HOLD, LANDING = 0, 1, 2, 3, 4


def trapz_profile(tau, L, V, t_acc):
    """Profilo posizione/velocita' trapezoidale (rampa ad accelerazione l10001, 60000)imitata,
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


def build_vbat_logconf():
    # blocco SEPARATO: il blocco "state" (6 float = 24 byte) e' gia' al limite
    # di dimensione di un singolo blocco CRTP — aggiungere pm.vbat li' dentro
    # supera il tetto ("log configuration is too large" -> segfault, visto ieri).
    # Un blocco a parte, anche piu' lento, evita il problema.
    lg = LogConfig(name="battery", period_in_ms=200)   # 5 Hz, basta per la batteria
    lg.add_variable("pm.vbat", "float")
    return lg


def build_accz_logconf():
    # Terzo blocco, separato come quello della batteria: il blocco "state" e'
    # gia' al limite CRTP. Qui gira alla stessa cadenza del controllo perche'
    # alimenta lo stimatore dell'hover.
    #
    # acc.z e' la FORZA SPECIFICA lungo l'asse di spinta, in unita' di g:
    # l'accelerometro non "vede" la gravita', quindi in hover legge ~1, non 0.
    # Vale T/(m*g), cioe' esattamente la spinta normalizzata — ed e' gia' lungo
    # l'asse dei rotori, quindi il coseno dell'inclinazione e' incluso e non
    # serve nessuna correzione di tilt.
    lg = LogConfig(name="accz", period_in_ms=int(DT * 1000))
    lg.add_variable("acc.z", "float")
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
            "target_x", "target_y", "target_z", "target_vx", "target_vy", "target_vz", "vbat",
            "cmd_ctrl", "spool_ceil", "accz", "hover_raw", "hover_cmd",
            "hover_est", "hover_b50", "hover_kf", "kf_sigma", "kf_used", "b50_n", "hover_vbat",
            "az_mpc", "eps0", "eps_max", "ge_az", "z_ground"]
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

    fig, ax = plt.subplots(12, 1, sharex=True, figsize=(11, 20))

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
    ax[2].plot(t, a["hover_cmd"], color="k", lw=0.8, ls=":", label="hover cmd applicato")
    ax[2].axhline(60000, color="gray", lw=0.6, ls=":")
    ax[2].set_ylabel("thrust cmd"); ax[2].legend(loc="upper right"); shade_mpc(ax[2])

    # az: comandata (uscita finale, blend incluso), uscita grezza dell'MPC
    # verticale, misurata dall'accelerometro. La misurata e' la forza specifica
    # lungo l'asse di spinta meno g: coincide con l'accelerazione verticale solo
    # a tilt piccolo, ed e' rumorosa (vibrazioni delle eliche).
    az_meas = (a["accz"] - 1.0) * G
    ax[3].plot(t, az_meas, color="0.6", lw=0.6, label="misurata (acc.z)")
    ax[3].plot(t, a["az"], color="tab:purple", lw=1.3, label="comandata")
    ax[3].plot(t, a["az_mpc"], color="tab:red", lw=1.3, ls="--", label="uscita MPC verticale")
    ax[3].plot(t, a["ge_az"], color="tab:brown", lw=1.2, ls=":", label="effetto suolo (modello)")
    ax[3].axhline(0, color="k", lw=0.6)
    fl = ~np.isin(state_str, ["prespin", "spool"])   # in prespin az e' fittizia (-g)
    azv = np.concatenate([a["az"][fl], a["az_mpc"][fl]]); azv = azv[np.isfinite(azv)]
    if len(azv):
        ax[3].set_ylim(min(azv.min(), -1.0) - 0.5, max(azv.max(), 1.0) + 0.5)
    ax[3].set_ylabel("az [m/s^2]"); ax[3].legend(loc="upper right", fontsize=7, ncol=3)
    shade_mpc(ax[3])

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
    ax[7].set_ylabel("a_xy [m/s^2]")
    ax[7].legend(loc="upper right"); shade_mpc(ax[7])

    ax[8].plot(t, a["vbat"], color="tab:cyan", lw=1.3)
    ax[8].axhline(4.1, color="green", lw=0.7, ls=":", label="carica piena (~4.1V)")
    ax[8].axhline(3.8, color="orange", lw=0.7, ls=":", label="attenzione (~3.8V)")
    ax[8].axhline(3.5, color="red", lw=0.7, ls=":", label="scarica (~3.5V)")
    ax[8].set_ylabel("batteria [V]")
    ax[8].legend(loc="upper right", fontsize=7); shade_mpc(ax[8])

    # --- hover: fisso usato, grezzo, stimatori e modello su batteria (log) ---
    ax[9].plot(t, a["hover_raw"], color="0.7", lw=0.5, label="cmd/acc.z grezzo")
    ax[9].plot(t, a["hover_cmd"], color="k", lw=1.4, ls=":",
               label=f"applicato ({HOVER_SOURCE.upper() if HOVER_ADAPT else 'fisso'})")
    ax[9].plot(t, a["hover_est"], color="tab:blue", lw=1.5, label="stimatore attuale")
    ax[9].plot(t, a["hover_b50"], color="tab:orange", lw=1.5, label="stimatore B50")
    ax[9].plot(t, a["hover_kf"], color="tab:red", lw=1.5, ls="--", label="filtro di Kalman")
    kf_s = a["kf_sigma"]
    ax[9].fill_between(t, a["hover_kf"] - 2 * kf_s, a["hover_kf"] + 2 * kf_s,
                       color="tab:red", alpha=0.12, lw=0)
    ax[9].plot(t, a["hover_vbat"], color="tab:cyan", lw=1.3, ls="--", label="modello vbat")
    hv = np.concatenate([a[k][np.isfinite(a[k])] for k in ("hover_est", "hover_b50", "hover_kf", "hover_vbat")]
                        + [a["hover_cmd"][np.isfinite(a["hover_cmd"])]])
    ax[9].set_ylim(hv.min() - 2000, hv.max() + 2000)
    ax[9].set_ylabel("hover [cmd]"); ax[9].legend(loc="upper right", fontsize=7, ncol=3)
    shade_mpc(ax[9])

    # --- slack del cono (solo in MPC) ---
    ax[10].plot(t, a["eps0"], color="tab:red", lw=1.3, label="eps primo passo")
    ax[10].plot(t, a["eps_max"], color="tab:brown", lw=1.0, ls="--", label="eps max orizzonte")
    ax[10].axhline(0, color="k", lw=0.6)
    ax[10].set_ylabel("slack cono [m]"); ax[10].legend(loc="upper right", fontsize=7)
    shade_mpc(ax[10])

    # --- quota sopra il suolo, per leggere l'effetto suolo sui pannelli sopra ---
    h_gnd = a["z"] - a["z_ground"]
    ax[11].plot(t, h_gnd, color="tab:green", lw=1.3)
    ax[11].axhline(HOVER_EST_Z_MIN - np.nanmean(a["z_ground"]), color="k", lw=0.7, ls=":",
                   label="stimatori congelati sotto")
    ax[11].set_ylabel("z - suolo [m]"); ax[11].set_xlabel("t [s]")
    ax[11].legend(loc="upper right", fontsize=7); shade_mpc(ax[11])

    eps_on = a["eps_max"][np.isfinite(a["eps_max"])]
    if len(eps_on):
        n_act = int(np.sum(eps_on > 1e-4))
        print(f"slack cono: max {eps_on.max():.4f} m, attivo (>1e-4) in {n_act}/{len(eps_on)} "
              f"tick MPC")
    for k, lab in (("hover_est", "attuale"), ("hover_b50", "B50"), ("hover_kf", "KF"), ("hover_vbat", "vbat")):
        v = a[k][np.isfinite(a[k])]
        if len(v):
            print(f"hover {lab}: primo {v[0]:.0f}, ultimo {v[-1]:.0f}")
    print(f"hover applicato: iniziale {HOVER_CMD_INIT}, finale {a['hover_cmd'][-1]:.0f} "
          f"(adattamento {'ATTIVO, sorgente ' + HOVER_SOURCE if HOVER_ADAPT else 'spento'})")
    ku = a["kf_used"][np.isfinite(a["kf_used"])]
    if len(ku):
        print(f"KF: misure usate {100 * ku.mean():.1f}% ({int((ku == 0).sum())} scartate dal gating)")
    if len(a["vbat"]):
        print(f"batteria: iniziale {a['vbat'][0]:.3f}V, finale {a['vbat'][-1]:.3f}V "
              f"(delta {a['vbat'][0]-a['vbat'][-1]:.3f}V)")

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
    plot_ground_effect(a, state_str, stamp, plt)
    plt.show()

    return stamp

# raggio elica Crazyflie 2.x (eliche da 45 mm), per il modello di confronto
PROP_R = 0.0225


def plot_ground_effect(a, state_str, stamp, plt):
    """Stima a posteriori dell'effetto suolo nel landing.

    Riferimento fuori effetto suolo: mediana di cmd/acc.z in hold (fermo, a
    Z_HOLD). Rapporto di spinta T_IGE/T_OGE a parita' di comando:
        k(h) = hover_rif / (cmd/acc.z)(h)
    Si usa acc.z grezzo (anche sotto HOVER_EST_Z_MIN, dove gli stimatori sono
    congelati). Confronto con Cheeseman-Bennett, 1/(1-(R/4h)^2), che e' un
    modello a rotore singolo: sui multirotori l'effetto misurato e' in genere
    maggiore. ATTENZIONE: in discesa il flusso entrante abbassa l'hover
    apparente indipendentemente dal suolo (visto sui voli del 14/9), quindi i
    punti con |vz| grande sono colorati a parte.
    """
    hold = (state_str == "hold") & (a["cmd"] < 59999)
    raw_all = a["cmd"] / a["accz"]
    ok_ref = hold & np.isfinite(raw_all) & (a["accz"] > HOVER_EST_ACC_MIN) \
        & (a["accz"] < HOVER_EST_ACC_MAX)
    land = (state_str == "landing") & np.isfinite(raw_all) & (a["cmd"] < 59999) \
        & (a["accz"] > HOVER_EST_ACC_MIN) & (a["accz"] < HOVER_EST_ACC_MAX)
    if ok_ref.sum() < 20 or land.sum() < 20:
        print("effetto suolo: dati insufficienti (serve hold e landing)")
        return
    ref = float(np.median(raw_all[ok_ref]))
    h = (a["z"] - a["z_ground"])[land]
    k = ref / raw_all[land]
    vz = a["vz"][land]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(h, k, c=np.abs(vz), cmap="viridis", s=10, alpha=0.6,
                    vmin=0, vmax=max(0.3, float(np.nanmax(np.abs(vz)))))
    fig.colorbar(sc, ax=ax).set_label("|vz| [m/s]")
    edges = np.arange(0.0, max(0.05, float(np.nanmax(h))) + 0.05, 0.05)
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h < hi)
        if m.sum() >= 5:
            mid.append(0.5 * (lo + hi)); med.append(float(np.median(k[m])))
    ax.plot(mid, med, "k-o", lw=1.8, ms=5, label="mediana ogni 5 cm")
    hh = np.linspace(max(PROP_R / 4 * 1.2, 0.01), max(edges), 200)
    ax.plot(hh, 1.0 / (1.0 - (PROP_R / (4 * hh)) ** 2), "r--",
            label=f"Cheeseman-Bennett (R={PROP_R*1000:.1f} mm)")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axvline(HOVER_EST_Z_MIN - float(np.nanmean(a["z_ground"])), color="0.5", ls=":",
               label="soglia congelamento stimatori")
    ax.set_xlabel("quota sopra il suolo [m]")
    ax.set_ylabel("T_IGE / T_OGE  (= hover_hold / (cmd/acc.z))")
    ax.set_title(f"Effetto suolo nel landing (rif. hold = {ref:.0f})")
    ax.set_ylim(0.8, 1.3); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = f"last_run_plots/flight_{stamp}_ground_effect.png"
    fig.savefig(png, dpi=110)
    if med:
        print(f"effetto suolo: rapporto mediano {med[0]:.3f} a h={mid[0]:.3f} m "
              f"(rif. hold {ref:.0f}); plot {png}")


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
    allx = np.concatenate([pos[0], los[0], target_pos[0]])
    ally = np.concatenate([pos[1], los[1], target_pos[1]])
    allz = np.concatenate([pos[2], los[2], target_pos[2]])
    xr = (allx.min(), allx.max()); yr = (ally.min(), ally.max()); zr = (allz.min(), allz.max())
    mr = max(xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0], 1e-3) / 2.0
    xm = 0.5*(xr[0]+xr[1]); ym = 0.5*(yr[0]+yr[1]); zm = 0.5*(zr[0]+zr[1])
    ax3d.set_xlim(xm-mr, xm+mr); ax3d.set_ylim(ym-mr, ym+mr); ax3d.set_zlim(zm-mr, zm+mr)
    ax3d.set_box_aspect((1, 1, 1))
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
        self.ctrl = HybridController(dt=DT, mass=MASS)
        self.latest_pose_t = None       # istante di arrivo dell'ultima posa
        self._pose_lost_t0 = None       # inizio discesa per posa persa
        import inspect as _insp
        _cf = _insp.getsourcefile(HybridController)
        _ok = hasattr(self.ctrl, "log_az_mpc")
        (self.get_logger().info if _ok else self.get_logger().warn)(
            f"controller caricato da {_cf} "
            f"({'con' if _ok else 'SENZA'} diagnostica az_mpc/slack)")
        self.rows = []
        self.t_start = time.perf_counter()
        self.RAMP_T = 1.0
        
        self.V_LAND = 0.5
        self.V_NAV  = 0.3     # velocita' carrot in rising/nav [m/s]
        self.V_HOLD = 0.3     # velocita' carrot in hold [m/s]

        # stato del drone (aggiornato dal callback asincrono del logger) e
        # posizione dal mocap (aggiornata dalla callback _on_pose)
        self.latest_state = None
        self.latest_pose = None
        self.SETTLE_T = 3.0   # assestamento stima prima di partire [s]

        # --- connessione e setup sequenziale (arming, reset, warmup) ---
        # NB: teniamo il SyncCrazyflie aperto per tutta la vita del nodo.
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
        self._pose_sub = self.create_subscription(
            PoseStamped, MOCAP_TOPIC, self._on_pose, 10)

        # --- piattaforma (target di atterraggio) ---
        self._plat_buf = []                 # ultime pose piattaforma
        self.plat_xyz = None                # posa fissata come target
        self._land_ground = None            # "suolo" di atterraggio per l'MPC
        if USE_PLATFORM:
            self._plat_sub = self.create_subscription(
                PoseStamped, PLATFORM_TOPIC, self._on_platform, 10)

        # --- logger ASINCRONO: aggiorna self.latest_state ad ogni pacchetto ---
        self._logconf = build_logconf()
        self.cf.log.add_config(self._logconf)
        self._logconf.data_received_cb.add_callback(self._on_state)
        self._logconf.start()

        # --- blocco di log SEPARATO per la batteria (vedi build_vbat_logconf:
        #     va tenuto fuori dal blocco "state", gia' al limite CRTP) ---
        self.latest_vbat = float("nan")
        self._logconf_vbat = build_vbat_logconf()
        self.cf.log.add_config(self._logconf_vbat)
        self._logconf_vbat.data_received_cb.add_callback(self._on_vbat)
        self._logconf_vbat.start()

        # --- terzo blocco: acc.z per lo stimatore dell'hover ---
        self._logconf_accz = build_accz_logconf()
        self.cf.log.add_config(self._logconf_accz)
        self._logconf_accz.data_received_cb.add_callback(self._on_accz)
        self._logconf_accz.start()

        # --- reset Kalman + attesa mocap/stato + assestamento prima di partire ---
        self._warmup()

        # --- stato FSM (erano variabili locali del for, ora attributi) ---
        # Si parte da "prespin" (se abilitato): wp_counter/old_wp_id restano gia'
        # impostati su rising, cosi' la transizione prespin->rising non tocca
        # nient'altro della FSM.
        self.state = "prespin" if PRESPIN_ENABLED else "rising"
        self.wp_counter = RISING
        self.old_wp_id = IDLE
        self.latest_accz = None   # ultimo acc.z ricevuto [g]
        # stimatori dell'hover, solo log (NaN finche' non agganciati)
        self.hover_est = float("nan")    # stimatore attuale
        self.hover_b50 = float("nan")    # stimatore con aggancio da fermo
        self._b50_buf = []               # campioni cmd/acc.z raccolti da fermo
        self._b50_acc = []               # acc.z degli stessi campioni (per il KF)
        self._b50_cmd = []               # cmd degli stessi campioni (per il KF)
        self._vbat_at_b50 = float("nan") # vbat all'aggancio B50 (modello vbat)
        self._b50_ref = float("nan")     # valore B50 all'aggancio (modello vbat)
        # filtro di Kalman (inizializzato all'aggancio B50)
        self.kf_theta = None             # 1/h
        self.kf_P = None
        self.kf_q = None
        self.kf_R = None
        self.kf_rej = 0                  # scarti consecutivi
        self.kf_last = float("nan")      # 1 = misura usata, 0 = scartata, NaN = nessuna
        self._prespin_t0 = None   # inizio pre-spin (None = non ancora entrato)
        self._spool_t0 = None     # inizio rampa aperta (None = non attiva)
        self._z_ground = None     # quota del suolo campionata a inizio pre-spin
        self.stop_delta = 0.1
        self.WP = None
        self.prev_wp = self.wp_counter
        self.seg_t0 = time.perf_counter()
        self.dynamic_p_start = None
        self.land_t0 = None
        self.seg_p_start = None
        self.last_p_LOS = None   # ultima posizione reale della carota (per continuita' tra segmenti)
        self._finished = False
        self._emergency = False   # True dopo il primo Ctrl+C: congela la piattaforma mobile

        # --- timer di controllo a 1/DT Hz: E' il loop ---
        self.timer = self.create_timer(DT, self.tick)
        self.get_logger().info("Nodo avviato: loop di controllo attivo.")

    def _on_state(self, timestamp, data, logconf):
        """Callback asincrono del logger cflib. Gira nel thread di cflib."""
        self.latest_state = data

    def _on_vbat(self, timestamp, data, logconf):
        """Callback del blocco batteria separato."""
        self.latest_vbat = data["pm.vbat"]

    def _on_accz(self, timestamp, data, logconf):
        """Callback del blocco acc.z (forza specifica lungo l'asse di spinta)."""
        self.latest_accz = data["acc.z"]

    def _on_pose(self, msg):
        """Callback mocap: rimanda la posizione al Kalman (send_extpos, SOLO
        posizione, mai il quaternione — vedi note del progetto) e la salva
        per il controllo."""
        p = msg.pose.position
        try:
            self.cf.extpos.send_extpos(p.x, p.y, p.z)
        except Exception as e:
            self.get_logger().warn(f"send_extpos fallito: {e}")
        self.latest_pose = np.array([p.x, p.y, p.z])
        self.latest_pose_t = time.perf_counter()

    def _on_platform(self, msg):
        """Callback posa piattaforma: tiene le ultime PLATFORM_AVG_N pose."""
        p = msg.pose.position
        self._plat_buf.append((p.x, p.y, p.z))
        if len(self._plat_buf) > PLATFORM_AVG_N:
            self._plat_buf.pop(0)

    def _set_target_from_platform(self):
        """Fissa TARGET_XY e Z_LAND dalla posa della piattaforma (prima del decollo)."""
        global TARGET_XY, Z_LAND
        P = np.array(self._plat_buf)
        self.plat_xyz = np.median(P, axis=0)
        spread = np.ptp(P, axis=0) if len(P) > 1 else np.zeros(3)
        rest = float(self.latest_pose[2]) - FLOOR_Z      # drone appoggiato sul pavimento
        surface = float(self.plat_xyz[2]) + PLATFORM_TOP_DZ
        TARGET_XY = np.array([self.plat_xyz[0], self.plat_xyz[1]], dtype=float)
        Z_LAND = surface + rest + LAND_CLEARANCE
        self._land_ground = surface + rest
        self.get_logger().info(
            f"[PLATFORM] {len(P)} pose, piattaforma ({self.plat_xyz[0]:.3f}, "
            f"{self.plat_xyz[1]:.3f}, {self.plat_xyz[2]:.3f}) (escursione "
            f"{1000*spread.max():.1f} mm) -> TARGET_XY=({TARGET_XY[0]:.3f}, "
            f"{TARGET_XY[1]:.3f}), Z_LAND={Z_LAND:.3f}, suolo MPC={self._land_ground:.3f} "
            f"(drone appoggiato a {rest:.3f} m)")
        if spread.max() > 0.01:
            self.get_logger().warn("[PLATFORM] la piattaforma si muove (>1 cm): target fissato comunque")

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

        if USE_PLATFORM:
            self.get_logger().info(f"Attendo la posa della piattaforma su {PLATFORM_TOPIC}...")
            t0 = time.perf_counter()
            while rclpy.ok() and len(self._plat_buf) < PLATFORM_AVG_N:
                rclpy.spin_once(self, timeout_sec=0.02)
                self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
                if time.perf_counter() - t0 > PLATFORM_WAIT_T:
                    break
                time.sleep(0.005)
            if not self._plat_buf:
                try:
                    self._scf.close_link()
                except Exception:
                    pass
                raise RuntimeError(
                    f"nessuna posa su {PLATFORM_TOPIC} in {PLATFORM_WAIT_T:.0f} s: "
                    f"piattaforma non tracciata, volo annullato")
            self._set_target_from_platform()

        self.get_logger().info(f"Assestamento stima per {self.SETTLE_T:.1f}s...")
        t0 = time.perf_counter()
        while rclpy.ok() and (time.perf_counter() - t0) < self.SETTLE_T:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.cf.commander.send_setpoint(0.0, 0.0, 0, 0)
            time.sleep(DT)
        self.get_logger().info("Assestamento completato: avvio controllo.")

    @staticmethod
    def _filter_step(prev, raw):
        """Passo del filtro comune ai due stimatori: passa-basso + rate limit."""
        lam = DT / HOVER_EST_TAU
        target = (1.0 - lam) * prev + lam * raw
        step = float(np.clip(target - prev, -HOVER_EST_RATE * DT, HOVER_EST_RATE * DT))
        return float(np.clip(prev + step, HOVER_EST_CMD_MIN, HOVER_EST_CMD_MAX))

    def _update_hover_estimates(self, cmd_sent, z, vz):
        """Aggiorna gli stimatori dell'hover (HOVER_CMD lo applica il tick).

        Ritorna (accz_usato, stima_grezza); la grezza e' NaN quando la misura
        e' stata scartata, cosi' nei CSV si vede quando e perche'.
        """
        if self.latest_accz is None:
            return float("nan"), float("nan")
        accz = float(self.latest_accz)
        self.kf_last = float("nan")

        # In landing le stime restano ferme al valore precedente: in discesa il
        # flusso entrante abbassa l'hover apparente di ~2% e l'errore, piccolo e
        # costante, passa il gating del KF (verificato offline sul 16/9).
        if self.state == "landing":
            return accz, float("nan")

        # Congelamenti. In effetto suolo la spinta e' maggiore a parita' di
        # comando, quindi la stima uscirebbe troppo bassa; a comando saturo il
        # cmd inviato non e' quello richiesto e il rapporto perde significato.
        if z < HOVER_EST_Z_MIN:
            return accz, float("nan")
        if not (HOVER_EST_ACC_MIN <= accz <= HOVER_EST_ACC_MAX):
            return accz, float("nan")
        if cmd_sent >= 59999 or cmd_sent <= 10001:
            return accz, float("nan")

        raw = float(np.clip(cmd_sent / accz, HOVER_EST_CMD_MIN, HOVER_EST_CMD_MAX))

        # --- stimatore attuale: aggancio sul primo campione valido ---
        if not np.isfinite(self.hover_est):
            self.hover_est = raw
            self.get_logger().info(
                f"[HOVER_EST] aggancio attuale: {raw:.0f} (acc.z={accz:.3f}, "
                f"z={z:.2f} m, vz={vz:+.2f})")
        else:
            self.hover_est = self._filter_step(self.hover_est, raw)

        # --- stimatore B50: aggancio sulla mediana di campioni da fermo ---
        if not np.isfinite(self.hover_b50):
            if (z > B50_Z_MIN and abs(vz) < B50_VZ_MAX
                    and self.state in ("rising", "nav_to_wp", "hold")):
                self._b50_buf.append(raw)
                self._b50_acc.append(accz)
                self._b50_cmd.append(float(cmd_sent))
                if len(self._b50_buf) >= B50_N:
                    self.hover_b50 = float(np.median(self._b50_buf))
                    self._kf_init()
                    self._vbat_at_b50 = float(self.latest_vbat)
                    self._b50_ref = self.hover_b50
                    self.get_logger().info(
                        f"[HOVER_EST] aggancio B50: {self.hover_b50:.0f} "
                        f"({B50_N} campioni da fermo, vbat={self._vbat_at_b50:.3f} V)")
        else:
            self.hover_b50 = self._filter_step(self.hover_b50, raw)
            self._kf_step(float(cmd_sent), accz)

        return accz, raw

    def _kf_init(self):
        th = np.array(self._b50_acc) / np.array(self._b50_cmd)
        self.kf_theta = float(np.median(th))
        self.kf_P = float(np.var(th)) / len(th)        # incertezza della mediana
        self.kf_R = max(float(np.var(self._b50_acc)), 1e-6)
        h = 1.0 / self.kf_theta
        self.kf_q = self.kf_R * (DT / (KF_TAU * h)) ** 2
        self.get_logger().info(
            f"[HOVER_KF] init: h={h:.0f}  sigma_h={np.sqrt(self.kf_P) * h * h:.0f}  "
            f"sigma_acc={np.sqrt(self.kf_R):.3f} g")

    def _kf_step(self, cmd, accz):
        """Predizione + correzione con gating e recupero (stile PX4)."""
        if self.kf_theta is None:
            return
        self.kf_P += self.kf_q
        nu = accz - cmd * self.kf_theta
        S = cmd * cmd * self.kf_P + self.kf_R
        if nu * nu / S > KF_GATE ** 2:
            self.kf_rej += 1
            self.kf_last = 0.0
            if self.kf_rej >= KF_REC_N:
                self.kf_P *= KF_REC_P
                self.kf_rej = 0
                self.get_logger().warn("[HOVER_KF] misure scartate per 1 s: recupero (P aumentata)")
            return
        self.kf_rej = 0
        K = self.kf_P * cmd / S
        self.kf_theta += K * nu
        self.kf_P *= (1.0 - K * cmd)
        self.kf_last = 1.0

    def _hover_kf(self):
        if self.kf_theta is None or self.kf_theta <= 0:
            return float("nan")
        return float(1.0 / self.kf_theta)

    def _hover_kf_sigma(self):
        if self.kf_theta is None or self.kf_theta <= 0:
            return float("nan")
        h = 1.0 / self.kf_theta
        return float(np.sqrt(self.kf_P) * h * h)

    def _hover_vbat(self):
        """Modello su batteria, agganciato al valore B50 (solo log)."""
        if not (np.isfinite(self.hover_b50) and np.isfinite(self._vbat_at_b50)):
            return float("nan")
        return float(self._b50_ref + VBAT_SLOPE * (self.latest_vbat - self._vbat_at_b50))

    def _log_extras(self, hover_raw):
        """Colonne diagnostiche comuni a tutte le righe di log."""
        c = self.ctrl
        return dict(
            hover_raw=hover_raw, hover_cmd=HOVER_CMD,
            hover_est=self.hover_est, hover_b50=self.hover_b50,
            hover_kf=self._hover_kf(), kf_sigma=self._hover_kf_sigma(), kf_used=self.kf_last,
            b50_n=len(self._b50_buf), hover_vbat=self._hover_vbat(),
            # getattr: con un controller_deploy senza i campi diagnostici il
            # volo prosegue e queste colonne restano NaN
            az_mpc=getattr(c, "log_az_mpc", float("nan")),
            eps0=getattr(c, "log_eps0", float("nan")),
            eps_max=getattr(c, "log_eps_max", float("nan")),
            ge_az=getattr(c, "log_ge0", float("nan")),
            z_ground=(self._z_ground if self._z_ground is not None else float("nan")))

    def _append_row(self, pos, vel, state, mode, p_LOS, force, cmd, az, roll, pitch,
                    solve_ms, v_ff, current_target_vel, cmd_ctrl, spool_ceil, wp_landing,
                    accz=float("nan"), hover_raw=float("nan")):
        """Riga di log con le stesse chiavi del percorso principale.

        Serve al pre-spin, che non passa dal controllore: save_and_plot() legge
        le colonne per nome e andrebbe in KeyError su una riga incompleta.
        """
        self.rows.append(dict(
            t=time.perf_counter() - self.t_start, state=state, mode=mode,
            x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2],
            carrot_x=p_LOS[0], carrot_y=p_LOS[1], carrot_z=p_LOS[2],
            force=force, cmd=cmd, az=az, roll=roll, pitch=pitch, solve_ms=solve_ms,
            ref_vx=v_ff[0], ref_vy=v_ff[1], ref_vz=v_ff[2],
            target_x=wp_landing[0], target_y=wp_landing[1], target_z=wp_landing[2],
            target_vx=current_target_vel[0], target_vy=current_target_vel[1],
            target_vz=current_target_vel[2],
            vbat=self.latest_vbat, cmd_ctrl=cmd_ctrl, spool_ceil=spool_ceil,
            accz=(self.latest_accz if self.latest_accz is not None else float("nan")),
            **self._log_extras(hover_raw)))

    def tick(self):
        """Un ciclo di controllo. Identico al corpo del vecchio for-loop."""
        global TARGET_XY, HOVER_CMD

        if self._finished:
            return
        data = self.latest_state
        pose_xyz = self.latest_pose
        if data is None or pose_xyz is None:
            return  # nessuno stato/posizione ancora ricevuto

        # ---- WATCHDOG POSA MOCAP ----
        now_w = time.perf_counter()
        pose_age = now_w - self.latest_pose_t if self.latest_pose_t is not None else 1e9
        if pose_age > POSE_TIMEOUT or self._pose_lost_t0 is not None:
            self._on_pose_lost(pose_xyz, pose_age, now_w)
            return

        # POSIZIONE dal mocap (/cf_drone/pose); VELOCITA' dal filtro di Kalman
        # onboard (stateEstimate.vx/vy/vz del logger cflib) — identico al
        # resto della logica di main_alpha_landing.py.
        pos = pose_xyz.copy()
        vel = np.array([data["stateEstimate.vx"], data["stateEstimate.vy"],
                        data["stateEstimate.vz"]])

        # ---------------- DECOLLO IN ANELLO APERTO: PRE-SPIN + SPOOL ----------
        # Si esce PRIMA dell'inizializzazione di self.WP, quindi i waypoint
        # vengono ancorati alla posizione reale del drone all'istante del
        # distacco: la carota parte da li', con errore di posizione nullo.
        if self.state in ("prespin", "spool"):
            now = time.perf_counter()

            if self.state == "prespin":
                if self._prespin_t0 is None:
                    self._prespin_t0 = now
                    self._z_ground = float(pos[2])
                    self.get_logger().info(
                        f"[SPOOL] pre-spin a {PRESPIN_CMD} per {PRESPIN_T:.2f}s "
                        f"(suolo z={self._z_ground:.3f} m)")
                cmd_ol = float(PRESPIN_CMD)
                if now - self._prespin_t0 >= PRESPIN_T:
                    self.state = "spool"
                    self._spool_t0 = now
                    self.get_logger().info(
                        f"[SPOOL] rampa aperta da {PRESPIN_CMD} a {SPOOL_RATE:.0f} u/s "
                        f"fino al distacco (max {SPOOL_CMD_MAX})")
            else:
                elapsed = now - self._spool_t0
                cmd_ol = min(PRESPIN_CMD + SPOOL_RATE * elapsed, float(SPOOL_CMD_MAX))

                if vel[2] > LIFTOFF_VZ or pos[2] > self._z_ground + LIFTOFF_DZ:
                    # Passaggio di consegne. cmd_ol e' il comando che ha appena
                    # staccato il drone, cioe' ~l'hover vero: lo usiamo per
                    # pre-caricare l'integratore del PID (trasferimento bumpless)
                    # invece di lasciarlo ripartire da zero e far sprofondare il
                    # drone. Il clamp interno del DSL-PID (+-0.15 su z) limita
                    # comunque il recupero: il resto lo assorbe il termine P.
                    force_lift = cmd_ol * HOVER_FORCE / HOVER_CMD
                    i_z = (force_lift - self.ctrl.GRAVITY) / self.ctrl.I_COEFF_FOR[2]
                    self.ctrl.integral_pos_e[:] = 0.0
                    self.ctrl.integral_pos_e[2] = float(np.clip(i_z, -0.15, 0.15))
                    self.state = "rising"
                    self._spool_t0 = None
                    self.seg_t0 = now          # la carota parte ORA
                    self.get_logger().info(
                        f"[SPOOL] distacco a z={pos[2]:.3f} m con cmd={cmd_ol:.0f} "
                        f"dopo {elapsed:.2f}s -> rising (integratore z precaricato a "
                        f"{self.ctrl.integral_pos_e[2]:.3f})")

                elif elapsed > SPOOL_T_MAX:
                    self.get_logger().error(
                        f"[SPOOL] nessun distacco dopo {elapsed:.2f}s a cmd={cmd_ol:.0f}: "
                        f"taglio motori (batteria scarica? elica danneggiata?)")
                    self._shutdown_flight()
                    return

            if self.state in ("prespin", "spool"):
                self.cf.commander.send_setpoint(0.0, 0.0, 0.0, int(cmd_ol))
                self.ctrl.integral_pos_e[:] = 0.0
                self._append_row(pos, vel, state=self.state, mode=0,
                                 p_LOS=pos, force=0.0, cmd=int(cmd_ol), az=-G,
                                 roll=0.0, pitch=0.0, solve_ms=0.0,
                                 v_ff=np.zeros(3), current_target_vel=np.zeros(3),
                                 cmd_ctrl=float("nan"), spool_ceil=float(cmd_ol),
                                 wp_landing=pos)
                return
            # appena passati a "rising": si prosegue col controllo in questo tick

        if self.WP is None:
            hx, hy = pos[0], pos[1]
            self.WP = np.array([
                [hx, hy, pos[2]],                    # start
                [hx, hy, Z_CRUISE],                  # rising: sale a 1.0m dove sei
                [0.0, -2.0, Z_CRUISE],               # nav: va a (0, -2)
                [TARGET_XY[0], TARGET_XY[1], Z_HOLD],# hold: sopra il target (poi ricalcolato sul cono)
                [TARGET_XY[0], TARGET_XY[1], Z_LAND],# landing
            ])

            d_nav = float(np.linalg.norm(self.WP[NAV][0:2] - TARGET_XY))
            if d_nav < (Z_HOLD - Z_LAND) / ALPHA_CONE:
                self.get_logger().warn(
                    f"[PLATFORM] waypoint di nav a {d_nav:.2f} m dal target: il punto di "
                    f"hold ({(Z_HOLD - Z_LAND) / ALPHA_CONE:.2f} m dal target) cade dietro "
                    f"al nav, il drone fara' un tratto all'indietro")

        if self.state == "idle":
            self._shutdown_flight()
            return

        if not self._emergency:
            landing = (self.state == "landing")
        else:
            landing = False
        
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

        moving = self.state in ["hold", "landing"] and not self._emergency

        step_disp = TARGET_VEL * DT if moving else np.zeros(3)

        TARGET_XY += step_disp[0:2]
        self.WP[HOLD] += step_disp
        self.WP[LANDING] += step_disp

        if self.state in ["hold", "landing"] and not self._emergency:
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
            a_xy_lim=A_XY, final_pos=self.WP[LANDING], landing=landing,
            z_ground=(self._land_ground if self._land_ground is not None else self._z_ground))
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Nessun tetto qui: la rampa di spunto vive interamente nella fase
        # "spool" in anello aperto e si e' gia' conclusa col distacco.
        cmd_ctrl = force_to_cmd(force)
        cmd = cmd_ctrl
        spool_ceil_log = float("nan")

        self.cf.commander.send_setpoint(rad2deg(roll), rad2deg(pitch), 0.0, cmd)

        # Stime aggiornate DOPO l'invio: cmd e acc.z devono riferirsi allo
        # stesso istante fisico. HOVER_CMD si aggiorna a fine tick.
        accz_used, hover_raw = self._update_hover_estimates(cmd, float(pos[2]), float(vel[2]))

        az = force / MASS - G
        self.rows.append(dict(
            t=time.perf_counter() - self.t_start, state=self.state, mode=mode,
            x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2],
            carrot_x=p_LOS[0], carrot_y=p_LOS[1], carrot_z=p_LOS[2], force=force, cmd=cmd, az=az,
            roll=roll, pitch=pitch, solve_ms=solve_ms,
            ref_vx=v_ff[0], ref_vy=v_ff[1], ref_vz=v_ff[2],
            target_x=self.WP[LANDING][0], target_y=self.WP[LANDING][1], target_z=self.WP[LANDING][2],
            target_vx=current_target_vel[0], target_vy=current_target_vel[1], target_vz=current_target_vel[2],
            vbat=self.latest_vbat,
            cmd_ctrl=cmd_ctrl, spool_ceil=spool_ceil_log,
            accz=accz_used, **self._log_extras(hover_raw)))

        # Aggiornamento del valore applicato: DOPO il log, cosi' la colonna
        # hover_cmd e' quella usata in questo tick; vale dal tick successivo.
        # Sotto HOVER_EST_Z_MIN lo stimatore e' congelato, quindi anche il
        # valore applicato resta fermo per il tratto finale del landing.
        h_src = self._hover_kf() if HOVER_SOURCE == "kf" else self.hover_b50
        if self.state == "landing":
            h_src *= (1.0 - HOVER_LAND_MARGIN)
        if HOVER_ADAPT and np.isfinite(h_src):
            step = float(np.clip(h_src - HOVER_CMD,
                                 -HOVER_APPLY_RATE * DT, HOVER_APPLY_RATE * DT))
            HOVER_CMD = float(np.clip(HOVER_CMD + step, HOVER_EST_CMD_MIN, HOVER_EST_CMD_MAX))

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
        if self.state == "landing" and pos[2] <= self.WP[LANDING][2] + 1e-2:
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

    def _on_pose_lost(self, last_pos, age, now):
        """Posa mocap non aggiornata: niente controllo in anello chiuso."""
        zg = self._z_ground if self._z_ground is not None else float(last_pos[2])
        h_last = float(last_pos[2]) - zg
        if self._pose_lost_t0 is None:
            self._pose_lost_t0 = now
            self.get_logger().error(
                f"[WATCHDOG] posa mocap ferma da {age*1000:.0f} ms (ultima quota "
                f"{h_last:.2f} m sul suolo, stato {self.state})")
            if self.state in ("prespin", "spool") or h_last < POSE_LOST_CUT_H:
                self.get_logger().error("[WATCHDOG] quota bassa: taglio motori")
                self._shutdown_flight()
                return
            self._pose_lost_T = h_last / abs(POSE_LOST_VZ) + POSE_LOST_T_MARGIN
            self.get_logger().error(
                f"[WATCHDOG] discesa livellata a vz={POSE_LOST_VZ} m/s, "
                f"taglio dopo {self._pose_lost_T:.1f} s")
        if now - self._pose_lost_t0 > self._pose_lost_T:
            self.get_logger().error("[WATCHDOG] fine discesa aperta: taglio motori")
            self._shutdown_flight()
            return
        vel = np.full(3, float("nan"))
        k = 1.0 - 0.05   # senza stato: leggermente sotto l'hover
        if self.latest_state is not None:
            d = self.latest_state
            vel = np.array([d["stateEstimate.vx"], d["stateEstimate.vy"], d["stateEstimate.vz"]])
            a_cmd = POSE_LOST_KV * (POSE_LOST_VZ - vel[2])
            # max 1.0: a terra (vz=0) non deve poter ridecollare
            k = float(np.clip(1.0 + a_cmd / G, 0.85, 1.0))
        cmd = int(k * HOVER_CMD)
        self.cf.commander.send_setpoint(0.0, 0.0, 0.0, cmd)
        self._append_row(np.full(3, float("nan")), vel, state="pose_lost", mode=0,
                         p_LOS=np.full(3, float("nan")), force=float("nan"), cmd=cmd,
                         az=float("nan"), roll=0.0, pitch=0.0, solve_ms=0.0,
                         v_ff=np.zeros(3), current_target_vel=np.zeros(3),
                         cmd_ctrl=float("nan"), spool_ceil=float("nan"),
                         wp_landing=np.full(3, float("nan")))

    def _start_emergency_landing(self):
        """Primo Ctrl+C: invece di tagliare subito i motori (caduta libera dalla
        quota attuale), passa in modalita' atterraggio verso un target creato al
        volo esattamente sopra la posizione corrente — riusa la stessa rampa
        trapezoidale + gate MPC del landing normale (stessa inizializzazione che
        usa la transizione hold->landing). La piattaforma mobile viene congelata
        (self._emergency=True) cosi' il target d'emergenza resta fermo invece di
        continuare a seguirla. Un secondo Ctrl+C durante questa fase fa comunque
        il taglio immediato (vedi main())."""
        if self._finished:
            return
        pos = self.latest_pose
        if pos is None or self.WP is None:
            self.get_logger().warn("Nessuna posizione disponibile: taglio motori diretto.")
            self._shutdown_flight()
            return

        self._emergency = True
        z_em = (self._z_ground + LAND_CLEARANCE) if self._z_ground is not None else Z_LAND
        self.WP[LANDING] = np.array([pos[0], pos[1], z_em])
        self.state = "landing"
        self.old_wp_id, self.wp_counter, self.stop_delta = HOLD, LANDING, 0.05
        self.dynamic_p_start = (self.last_p_LOS.copy() if self.last_p_LOS is not None
                                 else self.WP[LANDING].copy())
        self.land_t0 = time.perf_counter()
        self.get_logger().warn(
            f"ATTERRAGGIO DI EMERGENZA verso ({pos[0]:.2f}, {pos[1]:.2f}, {z_em:.2f}) "
            f"— piattaforma congelata — Ctrl+C di nuovo per taglio motori immediato.")

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
                self._logconf_vbat.stop()
            except Exception:
                pass
            try:
                self._logconf_accz.stop()
            except Exception:
                pass
            try:
                self._scf.close_link()
            except Exception:
                pass

        self.get_logger().info("Volo terminato: salvo CSV e plot...")
        stamp = save_and_plot(self.rows)
        try:
            # le righe "pose_lost" non hanno posizione: fuori dai grafici 3D
            plot_advanced_diagnostics(rows=[r for r in self.rows if r["state"] != "pose_lost"],
                                      stamp=stamp, alpha_cone=ALPHA_CONE, z_cut=1.0, r_base=0.3)
        except Exception as e:
            self.get_logger().warn(f"diagnostica avanzata non generata: {e}")
        self.get_logger().info("Fatto. Puoi chiudere con Ctrl+C.")
        rclpy.shutdown()


def main(args=None):
    # signal_handler_options=NO: rclpy per default installa un suo gestore di
    # SIGINT che comincia a spegnere il context non appena arriva Ctrl+C, in
    # parallelo/prima del nostro except KeyboardInterrupt — e' questo che
    # invalidava il context (visto: "publisher's context is invalid" subito
    # dopo il primo Ctrl+C). Disattivandolo, SIGINT arriva SOLO come
    # KeyboardInterrupt Python, gestito per intero dal nostro codice.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = LandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node._start_emergency_landing()
            # NB: un secondo rclpy.spin()/spin_once() qui NON e' affidabile —
            # dopo il primo SIGINT il context rclpy puo' gia' essere invalidato
            # (visto empiricamente: "publisher's context is invalid" subito dopo
            # il primo Ctrl+C), quindi spin() torna senza eseguire nessun tick e
            # si cade nel taglio motori immediato. Il comando ai motori passa da
            # cflib (radio/USB), indipendente da ROS: guidiamo il loop a mano.
            t_emerg0 = time.perf_counter()
            while not node._finished:
                # le callback mocap vanno pompate a mano (rclpy.spin non gira
                # piu'): senza, la posa resterebbe ferma. Se il context rclpy e'
                # gia' invalido la posa invecchia e interviene il watchdog.
                try:
                    rclpy.spin_once(node, timeout_sec=0.0)
                except Exception:
                    pass
                node.tick()
                if time.perf_counter() - t_emerg0 > 15.0:   # tetto di sicurezza
                    print("Atterraggio di emergenza troppo lungo: taglio motori.")
                    break
                time.sleep(DT)
        except KeyboardInterrupt:
            print("Ctrl+C di nuovo: taglio motori immediato!")
        except Exception as e:
            print(f"Errore durante l'atterraggio di emergenza: {e}")
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