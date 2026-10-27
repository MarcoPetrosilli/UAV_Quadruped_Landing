"""
main_rising_test.py  —  IDENTICO a main_alpha_landing.py, unica differenza:
la posizione arriva dal mocap (/cf_drone/pose) invece che dal solo stimatore
onboard — quindi qui c'e' in piu' la sottoscrizione mocap (send_extpos verso
il Kalman filter) e l'attesa/reset iniziale del filtro prima di partire.
Tutto il resto (FSM, guida, controllore, plotting) e' lo stesso file.

INDICE DELLE COSE AGGIUNTE PER LO YAW E LA PIATTAFORMA (per non cercarle):
  - YAW_*                  blocco "controllo di imbardata": abilitazione,
                           guadagno, rate massimo, banda morta, i due segni
                           (YAW_RATE_SIGN, YAW_FW_SIGN) e YAW_TARGET.
  - PLATFORM_YAW_*         lettura dell'orientamento della piattaforma:
                           ENABLED, OFFSET (None = autocalibra), MOVE_LOG
                           (solo stampa). Niente soglie, niente blocchi.
  - PLATFORM_VEL_*         velocita' della piattaforma passata a target_vel:
                           ENABLED, WIN, TAU, MAX. Nessuna zona morta.
  - _yaw_frames()          costruisce R_drone (IMU) e R_target (piattaforma).
  - _platform_yaw()        yaw corretto della piattaforma.
  - _platform_vel()        stima della velocita' xy.
  - _calibrate_platform_yaw()  misura l'offset all'avvio e lo stampa.
  - colonne CSV            yaw_drone, yaw_err, yaw_rate, yaw_plat,
                           yaw_plat_raw, pose_age (tutte solo diagnostica).
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
MOCAP_TOPIC = "/drone_003/pose"

# ---- piattaforma dal mocap ----------------------------------------------------
# Il target di atterraggio (TARGET_XY e Z_LAND) viene preso dalla posa della
# piattaforma, letta PRIMA del decollo e poi tenuta fissa (piattaforma ferma).
#   quota superficie   = z_piattaforma + PLATFORM_TOP_DZ
#   Z_LAND             = superficie + quota del drone appoggiato + LAND_CLEARANCE
#   suolo per l'MPC    = superficie + quota del drone appoggiato
#   suolo effetto suolo= superficie - GE_ROTOR_DZ        <-- NON lo stesso!
# Le ultime due sono quantita' diverse e vanno tenute separate. "suolo per
# l'MPC" e' la quota a cui l'origine del corpo si trova a contatto avvenuto:
# giusta per la macchina a stati e per congelare l'osservatore. Il modello di
# effetto suolo invece vuole la distanza PIANO ROTORI - SUPERFICIE, che a
# contatto NON e' zero ma vale la quota del drone appoggiato (~3.5 cm).
# Usando la prima al posto della seconda il modello sottostimava h di quella
# costante e saturava a k_max in ogni atterraggio (vedi commento in
# controller_deploy.py).
# "quota del drone appoggiato" e' la z mocap del drone fermo sul pavimento
# prima del decollo (pavimento a z = FLOOR_Z): e' l'offset tra l'origine del
# corpo rigido del drone e la superficie su cui poggia, e vale anche sulla
# piattaforma. LAND_CLEARANCE = 0.015 riproduce il vecchio Z_LAND = 0.05 con
# drone a 0.035 sul pavimento.
USE_PLATFORM     = True
PLATFORM_TOPIC   = "/platform/pose"
PLATFORM_TOP_DZ  = -0.012  # [m] superficie di appoggio MENO origine del corpo
                           # rigido della piattaforma. Misurato il 21/9 con il
                           # drone appoggiato a mano: 0.2281 invece di
                           # 0.2053 + 0.035 -> superficie 1.2 cm SOTTO il pivot
# Offset orizzontale tra pivot della piattaforma e punto in cui deve stare il
# PIVOT DEL DRONE quando il drone e' centrato, espresso nel frame della
# piattaforma (ruota con il suo yaw).
#
# Misura 21/9, appoggiando a mano il drone al centro della piastra:
#   drone - piattaforma = (+1.62, -1.14) cm nel mondo, yaw piattaforma -90.3 deg
#   -> PLATFORM_XY_OFFSET = (+1.14, +1.62) cm        [valore usato fino al 18/10]
#
# Misura 18/10, stesso metodo:
#   drone - piattaforma = (+1.158, -0.574) cm nel mondo, yaw piattaforma +168.845
#   -> PLATFORM_XY_OFFSET = (-1.247, +0.339) cm      [valore attuale]
#
# Le due misure NON coincidono: 2.71 cm di differenza. La seconda e' validata
# in quota (z drone appoggiato previsto 0.2220, misurato 0.2231: 1.1 mm), quindi
# i frame sono giusti e non e' un errore di conto. La differenza e' coerente con
# il bias sistematico misurato a posteriori su 291 atterraggi a piattaforma
# ferma, che dava una media non nulla dell'errore laterale.
#
# ATTENZIONE, da verificare: le due misure sono state prese a yaw piattaforma
# molto diversi (-90.3 e +168.8 deg). Se l'offset nel frame del corpo fosse
# davvero costante dovrebbero coincidere. Finche' non si ripete la misura a piu'
# orientamenti e si verifica che il valore non cambia, questo numero vale per la
# configurazione del 18/10 e non va trasferito all'indietro sui voli vecchi.
PLATFORM_XY_OFFSET = np.array([-0.01247, 0.00339])   # [m] nel frame piattaforma
LAND_CLEARANCE   = 0.0     # [m] target sopra la quota di appoggio. Era 0.015,
                           # eredita' del vecchio Z_LAND = 0.05 sul pavimento.
                           # Con quel valore piu' la soglia di fine landing
                           # (+1 cm) il drone veniva lasciato 2.3-2.5 cm sopra
                           # il contatto e i motori tagliati di colpo: gli
                           # ultimi centimetri erano CADUTA LIBERA. Volo 131434:
                           # vz controllata -0.094 m/s, ma impatto reale
                           # sqrt(0.094^2 + 2g*0.023) = 0.68 m/s, sette volte
                           # tanto. Ora il controllo arriva fino al contatto e
                           # a fermare il drone ci pensa il contatto stesso.
FLOOR_Z          = 0.0     # [m] quota del pavimento nel frame mocap
# Offset tra il PIANO DEI ROTORI e l'origine del corpo rigido del drone nel
# frame mocap, positivo se i rotori stanno SOPRA l'origine. Serve solo al
# modello di effetto suolo:
#     h_effetto_suolo = z_mocap + GE_ROTOR_DZ - quota_superficie
# 0.0 significa "uso l'origine del corpo come piano rotori", che e' la scelta
# neutra e comunque corretta a meno di qualche mm: basta misurarlo una volta
# (drone appoggiato, quota mocap meno quota del piano eliche) se si vuole
# essere precisi. NON confonderlo con la quota del drone appoggiato: quella
# e' la distanza origine-superficie, questa e' rotori-origine.
GE_ROTOR_DZ      = 0.0     # [m]
# Copia del clamp del controller, usata solo per disegnare la riga di
# saturazione nei plot: se ge_az ci si appoggia, il modello sta saturando e il
# valore non e' fisico.
CTRL_GE_K_MAX    = 1.25
CTRL_GE_R_EFF    = 0.075   # copia di ge_r_eff, per confrontare modello e dati
# ---- inseguimento della piattaforma in volo -----------------------------------
# Fino al 1/10 il target veniva letto UNA SOLA VOLTA prima del decollo e poi
# congelato: spostare la piattaforma durante il volo non cambiava nulla e il
# drone atterrava dove la piattaforma era al warmup. Con PLATFORM_TRACK la posa
# viene riletta a ogni tick e il waypoint di landing (piu' quota di contatto e
# superficie per l'effetto suolo) la segue, SENZA filtri sul movimento: dove si
# legge la piattaforma, li' sta il target. Niente rate limit ne' soglie sul
# salto: erano comodi come protezione ma rendevano il comportamento dipendente
# da quanto e quanto in fretta si sposta la piattaforma, quindi impossibile da
# interpretare durante le prove a mano.
# Restano solo due controlli sulla VALIDITA' del dato, invisibili a chi muove
# la piattaforma: pose nulle scartate (il topic ha pubblicato zeri piu' volte)
# e pose vecchie ignorate. La mediana su pochi campioni serve a non far
# tremolare la quota di contatto, non a rallentare l'inseguimento.
PLATFORM_TRACK       = True
PLATFORM_TRACK_N     = 15    # campioni per la mediana corrente della posa
PLATFORM_MIN_NORM    = 1e-3  # [m] posa praticamente nulla = non valida
PLATFORM_TRACK_MAXAGE = 0.3  # [s] pose piu' vecchie: il target si congela

# ---- freschezza della posa -------------------------------------------------
# La mediana su PLATFORM_TRACK_N campioni non aggiunge rumore ma aggiunge un
# RITARDO: per un segnale che avanza in modo monotono la mediana coincide con il
# campione CENTRALE della finestra, cioe' (N-1)/2 campioni indietro. A 50 Hz con
# N = 15 sono 0.14 s, che a 0.25 m/s valgono 3.5 cm di waypoint vecchio.
# Qualunque passa-basso causale avrebbe lo stesso problema: un EMA con costante
# di tempo tau ritarda una rampa esattamente di tau.
#
# Ma questo segnale non ha bisogno di essere filtrato. Il rumore della posa e'
# sotto il millimetro (deviazione standard misurata del target a piattaforma
# ferma: 0.01-0.16 mm). L'unico disturbo vero e' il 2-3% di campioni che salta
# di 2-3 cm - buchi e riacquisizioni del rigid body - e un salto del genere
# entrerebbe INTERO nel waypoint, perche' _track_platform non limita il delta e
# sposta in blocco WP[LANDING], WP[HOLD], dynamic_p_start e TARGET_XY.
#
# Quella non e' roba da filtro, e' roba da soglia: si prende la posa GREZZA
# (ritardo zero) e si scarta il campione se si discosta troppo dal precedente.
# PLATFORM_JUMP_MAX e' il limite sullo spostamento fra due campioni
# consecutivi. ATTENZIONE: va tenuto LARGO. Misurato sul volo 150911, su 431
# aggiornamenti distinti della posa, il residuo fra il salto osservato e il moto
# spiegato dal tempo trascorso non supera mai 12.4 mm: NON ci sono outlier da
# tagliare. I salti grossi (fino a 26 mm) sono moto legittimo su intervalli di
# pubblicazione lunghi - lo stream della piattaforma ha buchi fino a 80 ms.
# Con la soglia a 2 cm il gate scartava moto vero, e siccome uno scarto congela
# il riferimento il campione dopo sembrava ancora piu' lontano: si auto-agganciava.
# Sul volo 150911 ha prodotto 5 scarti di fila e poi un gradino di 5.7 cm sul
# waypoint a 4.7 cm di quota. A 0.15 m resta solo un rivelatore di
# teletrasporto, per il caso in cui il rigid body venga riacquisito altrove.
# Dopo PLATFORM_JUMP_N scarti consecutivi la posa viene accettata comunque: se
# il rigid body e' stato riacquisito davvero altrove, il target deve
# risincronizzarsi invece di restare congelato per sempre.
PLATFORM_RAW_POSE   = True   # False = vecchio comportamento (mediana su N)
PLATFORM_JUMP_MAX   = 0.15   # [m] salto massimo accettato fra due pose
PLATFORM_JUMP_N     = 2      # scarti consecutivi dopo i quali si riaggancia

# ---- velocita' della piattaforma (feed-forward su target_vel) -----------------
# Il tracking sposta il waypoint dove la piattaforma SI TROVA, ma all'MPC non
# diceva a che velocita' si sta muovendo: target_vel restava TARGET_VEL (la
# traiettoria sintetica, di solito zero). Con il target fermo non cambia nulla,
# ma su un target in moto continuo il drone resta sempre indietro di circa
# v*tau, perche' inseguire una posizione che scappa e' un problema di
# regolazione con errore a regime. Passando la velocita' stimata l'MPC ottiene
# (a) un riferimento di posizione che rampa, xref_k = wp + v*k*dt, (b) un
# riferimento di velocita' non nullo nel costo, (c) un gate reachable-set
# calcolato sull'errore RELATIVO. La stima e' la pendenza ai minimi quadrati di
# x(t) e y(t) sulla finestra: deriva meno del rumore di una differenza finita.
# Solo xy: la vz della piattaforma entra nella barriera d'impatto, che e' la
# specifica di touchdown, e un valore rumoroso li' sposta la velocita' di
# contatto ammessa. Si aggiunge dopo, se serve.
PLATFORM_VEL_ENABLED = True
PLATFORM_VEL_WIN     = 0.35  # [s] finestra di regressione
PLATFORM_VEL_TAU     = 0.25  # [s] passa-basso sulla stima
PLATFORM_VEL_MAX     = 0.35  # [m/s] saturazione per componente
# Nessuna zona morta e nessun limite di pendenza: la stima e' quella che esce
# dalla regressione, solo filtrata e saturata.
PLATFORM_WAIT_T  = 5.0     # [s] attesa massima della prima posa piattaforma
PLATFORM_AVG_N   = 50      # campioni mediati per fissare il target (~0.5 s a 100 Hz)

# ---- riferimento di quota dell'MPC in landing ---------------------------------
# Con l'MPC offset-free il drone converge ESATTAMENTE al riferimento, con
# velocita' che tende a zero: se il riferimento coincidesse con la soglia di
# fine landing (Z_LAND + 1 cm) l'ultimo tratto sarebbe asintotico (in
# simulazione fino a 60 s). Il riferimento viene quindi messo LAND_UNDERSHOOT
# sotto Z_LAND: la soglia viene attraversata con pochi cm/s, e il contatto
# fisico ferma comunque il drone.
LAND_UNDERSHOOT  = 0.0    # [m]. Alzato da 0.02 il 9/10. Con l'anello verticale
                           # tarato su Q_vrt=[2,1.5] e R_vrt=3 la costante di
                           # tempo vale 1.37 s, quindi la velocita' prodotta e'
                           # circa (quota - riferimento)/tau: con 2 cm di
                           # undershoot, a 3 cm dal contatto l'errore e' 5 cm e
                           # la discesa vale 3.6 cm/s, misurati 2.4 cm/s. Il
                           # drone non strisciava per un disturbo ma perche' il
                           # bersaglio era quasi raggiunto. Con 5 cm l'errore a
                           # 3 cm diventa 8 cm e la discesa ~5.8 cm/s, cioe' la
                           # specifica imp_vtd; al contatto resta ~3.6 cm/s. La
                           # barriera di impatto fa comunque da tetto (a 1 cm
                           # concede 6.3 cm/s con k_v = 1.3).

# ---- rilevamento del contatto -------------------------------------------------
# La fine del landing non e' piu' una soglia di quota ma il contatto vero,
# riconosciuto dalla dinamica. Firme in OR, tutte valide solo sotto
# CONTACT_H_MAX dal contatto previsto (sopra sarebbero falsi positivi).
#
# NOTA sulla prima versione (voli 143225 e 143358): il drone toccava e il
# rilevatore scattava 0.3-0.5 s dopo, perche' le tre firme di allora erano
# tutte deboli per un atterraggio MORBIDO:
#   - il picco di acc.z non esiste. L'accelerometro misura la FORZA SPECIFICA:
#     con il drone appoggiato e fermo spinta + reazione del terreno fanno
#     esattamente il peso, quindi acc.z legge 1.0 g qualunque cosa facciano i
#     motori (misurato: 0.88-1.10, mai vicino a 1.45). Un picco c'e' solo nel
#     transitorio d'urto, che a 5 cm/s e' trascurabile. Firma abbandonata.
#   - l'inversione di vz non scattava: il rimbalzo a 5 cm/s dava +0.027 contro
#     una soglia di +0.04.
#   - l'azzeramento di vz scattava tardi: il rumore della vz onboard (+-0.03)
#     faceva ripartire il timer dei 0.15 s continuativi.
#
# Firme attuali, dalla piu' pulita:
#   1) STALLO DELLA QUOTA: la posizione dal mocap e' molto meno rumorosa della
#      vz stimata a bordo. In discesa il drone fa ~12 mm in 0.2 s; appoggiato
#      ne fa meno di 2, con rumore del mocap sotto il mm. Fattore sei di
#      separazione su una grandezza pulita.
#   2) CROLLO DELL'HOVER APPARENTE: a terra e' il terreno a reggere il peso,
#      quindi cmd/acc.z scende molto sotto il comando di hover (misurato
#      28000-34000 contro 40500). Richiede anche h piccola, perche' pure
#      l'effetto suolo abbassa quel rapporto, ma solo del ~12% a 5 cm.
#   3) inversione di vz, soglia abbassata.
#   4) azzeramento di vz, soglia piu' larga, come rete.
# Due reti ulteriori: quota sotto il contatto e tempo massimo in prossimita'.
CONTACT_H_MAX    = 0.06    # [m] sopra questa quota sul contatto non si cerca
CONTACT_DZ_WIN   = 0.20    # [s] finestra per lo stallo della quota
CONTACT_DZ_MAX   = 0.004   # [m] spostamento massimo nella finestra = appoggiato
CONTACT_DZ_H     = 0.015   # [m] quota massima per la firma dello stallo. OBBLIGATORIA:
                           # 4 mm in 0.20 s e' una soglia di VELOCITA' implicita di
                           # 20 mm/s, e la discesa terminale ormai sta sui 30-60 mm/s.
                           # Nel volo 162908 (R_hrz = 10) il drone scendeva a 29 mm/s
                           # e ha coperto 4.8 mm in 0.24 s: la firma e' scattata a
                           # 39.7 mm dalla piattaforma, tagliando i motori prima del
                           # contatto. Lo stallo distingue 'fermo perche' appoggiato'
                           # da 'in discesa lenta' solo se si e' gia' quasi a terra.
CONTACT_HOVER_ON = False   # regola 2 (hover apparente) attiva?
CONTACT_VZUP_ON  = False   # regola 4 (inversione di vz) attiva?
                           # SEMPLIFICAZIONE del 10/10. Rigirando il rilevatore su
                           # 27 atterraggi, le sole regole 1 (quota ferma) e 3
                           # (urto, salto di acc.z sopra 0.50 g) rilevano 27/27,
                           # nessuna in anticipo di piu' di 40 ms, nessuna in
                           # ritardo di piu' di 0.3 s, nessuna rete di sicurezza
                           # mai usata. Le regole 2 e 4 non aggiungono copertura e
                           # sono le due che hanno prodotto TUTTI i falsi scatti in
                           # volo (hover apparente: 6 voli su 11 prima della
                           # scalatura con l'effetto suolo; inversione vz: 190221 e
                           # 104551). Restano nel codice, spente, per poterle
                           # riaccendere nell'ablazione.
                           # NB: il solo urto non basta -- con soglia 0.50 rileva
                           # 12/27, perche' un contatto morbido a 3-5 cm/s produce
                           # un salto di 0.2-0.3 g, sotto il rumore di acc.z
                           # campionato a 50 Hz. Abbassare la soglia a 0.15 porta a
                           # 24/27 ma con 6 scatti anticipati. Serve la quota ferma
                           # come conferma lenta.
CONTACT_HOVER_F  = 0.80    # cmd/acc.z sotto questa frazione dell'hover ATTESO ...
CONTACT_HOVER_H  = 0.025   # [m] ... e sotto questa quota = il terreno regge
CONTACT_HOVER_N  = 3       # campioni consecutivi
CONTACT_HOVER_GE = True    # soglia scalata con l'effetto suolo modellato.
                           # CORREZIONE del 10/10. Il confronto era con HOVER_CMD
                           # secco, ma a un centimetro dal suolo il controllo toglie
                           # il 18% di spinta per compensare ge + d_hat (1.4-1.8
                           # m/s^2), e la soglia e' il 20%: la regola finiva dentro
                           # la banda di lavoro normale e dichiarava il contatto
                           # mentre il drone era ancora in aria. Volo 185923: flag a
                           # 2.7 mm con vz = -0.058, nessun picco di acc.z prima
                           # (1.02 g; il massimo, 1.12, arriva DOPO), motori tagliati
                           # al 55% e ultimi millimetri in caduta. Il rapporto
                           # cmd/acc.z nell'ultimo campione prima del flag sta fra
                           # 0.775 e 0.89 su tutti i voli del 9/10: mezzo gruppo era
                           # sul filo. Ora il confronto e' con l'hover atteso A QUELLA
                           # QUOTA, HOVER_CMD * (1 - (a_ge + d_hat)/g): a 1 cm il
                           # fattore vale ~0.85 e la soglia scende da 32600 a 27600,
                           # ben sotto i ~31500 del volo normale. Un contatto vero
                           # (terreno che regge, acc.z ~ 1 g con cmd in crollo) la
                           # supera lo stesso. False = comportamento vecchio.
CONTACT_ACC_JUMP = 0.50    # [g] SALTO di acc.z sopra la mediana recente. Non
                           # una soglia assoluta (vedi sopra: appoggiato si
                           # legge 1.0 g), ma il transitorio d'urto, che per un
                           # contatto piu' deciso c'e' eccome: volo 142650,
                           # tocco a 0.097 m/s, acc.z salta a 2.15 g con
                           # mediana 0.99. Prende gli atterraggi duri, dove lo
                           # stallo della quota arriverebbe dopo il rimbalzo.
CONTACT_ACC_WIN  = 0.40    # [s] finestra per la mediana di riferimento
CONTACT_VZ_UP    = 0.02    # [m/s] vz risalita sopra questo = inversione
CONTACT_VZ_UP_ACC = 0.20   # [g] l'inversione vale solo se accompagnata da una
CONTACT_VZ_UP_WIN = 0.12   # [s] traccia di accelerazione in questa finestra.
                           # SECONDA CORREZIONE, 10/10 mattina. Due campioni non
                           # bastavano: volo 104551, vz ha letto +0.044 e +0.029 su
                           # due tick consecutivi a 3.9 mm dal suolo, con acc.z a
                           # 0.90 e 0.94 g, cioe' NESSUNA forza verso l'alto. Era un
                           # salto di 1.7 mm nella quota mocap, dentro il rumore.
                           # Motori tagliati a 3.9 mm, il drone e' caduto e ha
                           # toccato a -0.114 m/s con 2.31 g, 80 ms dopo il flag.
                           # Un rimbalzo vero decelera il drone, quindi deve vedersi
                           # in acc.z: la soglia e' meta' di CONTACT_ACC_JUMP, cosi'
                           # la regola 4 resta la versione "morbida" della 3 (salto
                           # debole + inversione) invece di fidarsi della sola vz.
CONTACT_VZ_UP_N  = 2       # campioni CONSECUTIVI sopra la soglia. CORREZIONE del
                           # 10/10: con un campione solo bastava uno sbuffo del
                           # mocap. Volo 190221: vz e' passata da -0.077 a +0.022
                           # in un campione a 4 mm dal suolo, la regola e' scattata,
                           # i motori sono stati tagliati al 52% e il drone ha fatto
                           # in caduta gli ultimi 6 mm, leggendo -0.092 ottanta ms
                           # dopo. Due campioni consecutivi costano 20 ms su un
                           # contatto vero (dove vz resta positiva per decine di ms
                           # dopo il rimbalzo) e tolgono i falsi da singolo campione.
CONTACT_VZ_ZERO  = 0.020   # [m/s] |vz| sotto questo ...
CONTACT_T_ZERO   = 0.20    # [s]   ... per questa durata, e sotto CONTACT_VZ_H,
                           # = appoggiato. NON si puo' allargare: la discesa
                           # terminale e' a 0.043-0.048 m/s, quindi una soglia
                           # a 0.045 scattava a 3 cm dal contatto (141749).
CONTACT_VZ_H     = 0.020   # [m] quota massima per questa rete
CONTACT_H_HARD   = -0.012  # [m] rete: quota sotto il contatto di 12 mm
CONTACT_T_MAX    = 4.0     # [s] rete: tempo massimo sotto CONTACT_H_MAX
# Dopo il contatto NON si taglia di colpo: si scende a una frazione dell'hover
# per TOUCHDOWN_T, cosi' il drone resta piantato invece di rimbalzare, poi zero.
TOUCHDOWN_CMD_F  = 0.55    # frazione di HOVER_CMD subito dopo il contatto
TOUCHDOWN_T      = 0.30    # [s] durata della rampa di spegnimento

# ---- controllo di imbardata ---------------------------------------------------
# SORGENTE: lo yaw del drone viene dalla IMU di bordo (stabilizer.yaw), non dal
# mocap. Il rigid body del drone in Motive puo' essere agganciato ruotato di 180
# gradi con un valore che cambia fra una sessione e l'altra, e una rotazione del
# frame sbagliata di 180 trasforma l'anello orizzontale in retroazione positiva.
# L'IMU non puo' mentire sulla direzione perche' misura la rotazione fisica; in
# cambio e' relativa alla posa al reset e deriva. Quindi: POSARE IL DRONE
# ALLINEATO prima del reset dell'estimatore.
# TARGET: per ora si assume la piattaforma a yaw zero (R_target = identita'),
# cioe' il drone tiene l'orientamento che aveva al reset. La lettura vera della
# posa della piattaforma si aggiunge dopo.
YAW_ENABLED      = True
YAW_KP           = 2.0     # [1/s]
YAW_RATE_MAX     = 60.0    # [deg/s]
YAW_DEADBAND     = 2.0     # [deg]
YAW_SYMMETRY     = 360.0   # [deg] 360 = allineamento vero; 90 sfrutta la
                           # simmetria del telaio in X
YAW_RATE_SIGN    = -1.0    # DA VERIFICARE A TERRA: il commander legacy del
                           # firmware nega il campo yawrate del pacchetto CRTP,
                           # quindi un valore positivo fa girare in senso orario
                           # mentre l'errore e' in convenzione mondo. Se diverge,
                           # il controller spegne lo yaw da solo dopo ~0.8 s.
YAW_FW_SIGN      = +1.0    # da verificare: in alcune versioni stabilizer.yaw e'
                           # negato rispetto alla convenzione del mondo
YAW_TARGET       = 0.0     # [deg] imbardata del target quando la lettura della
                           # piattaforma e' disattivata o non disponibile
# Stati della FSM in cui il controllo di imbardata e' ATTIVO (comanda yaw rate).
# Fuori da questi il rate comandato e' zero, ma la compensazione mondo->corpo
# resta attiva (vedi _yaw_frames): l'MPC e il DSL-PID producono roll/pitch nel
# frame del MONDO e il firmware li interpreta nel frame del DRONE, quindi la
# rotazione va fatta SEMPRE, anche quando non si sta imbardando.
#
# Perche' solo hold e landing. Il ritardo comando->accelerazione misurato e'
# ~180 ms (0.08 s di trasporto + 0.10 s di polo d'assetto, identificati su 13
# voli). _world_to_body ruota il comando con lo yaw campionato a inizio tick,
# quindi mentre il drone imbarda a 40-60 deg/s la direzione applicata e'
# vecchia di 7-11 deg e il 12-19% del comando orizzontale finisce sull'asse
# sbagliato. In rising e nav_to_wp il comando orizzontale e' grande e quel
# costo si paga per intero senza nessun beneficio: l'allineamento serve solo
# al contatto. In hold il drone e' fermo sopra il waypoint, il comando e'
# quasi nullo, quindi ruotare li' e' gratis e si arriva al landing allineati.
YAW_ACTIVE_STATES = ("nav_to_wp", "hold", "landing")

# ---- imbardata della piattaforma dal mocap -----------------------------------
# Il frame del corpo rigido della piattaforma in Motive NON e' allineato al
# mondo: il 21/9, con la piattaforma fisicamente a zero, il mocap leggeva
# -90.3 deg. Lo stesso -90 e' dentro PLATFORM_XY_OFFSET: l'offset misurato nel
# mondo (+1.62, -1.14) cm e' quello nel frame piattaforma (1.14, 1.62) cm
# ruotato di -90.00 deg. Dare quel valore grezzo al controllo di imbardata
# vorrebbe dire comandare al drone una rotazione di 90 gradi.
# Quindi si lavora sullo yaw CORRETTO = grezzo - offset, dove l'offset e' la
# lettura del mocap con la piattaforma a zero:
#   PLATFORM_YAW_OFFSET = <deg> -> offset FISSO (default): il target e'
#       assoluto, quindi una piattaforma storta prima del decollo viene
#       inseguita. E' la modalita' da usare per allineare il drone.
#   PLATFORM_YAW_OFFSET = None -> AUTOCALIBRAZIONE sulle pose di avvio: azzera
#       qualunque cosa legga all'accensione, quindi una rotazione fatta PRIMA
#       del lancio viene calibrata via e il target resta zero (volo 161445:
#       piattaforma storta a mano, yaw_plat fermo a 0.00 per tutto il volo e
#       drone mai raddrizzato). Serve solo per inseguire rotazioni RELATIVE
#       fatte durante il volo.
# Per trovare l'offset: piattaforma dritta, si lancia il nodo e si legge il
# valore grezzo nella riga [YAW] all'avvio. La correzione vera e' ricreare il
# corpo rigido allineato a +x del mondo: con quello l'offset e' zero.
PLATFORM_YAW_ENABLED    = True
PLATFORM_YAW_OFFSET     = -1.35  # [deg] misurato il 5/10 (run 142511, dev.std
                                 # 0.00 mdeg su 62 campioni); None = autocalibra
PLATFORM_YAW_MOVE_LOG   = 5.0    # [deg] variazione dello yaw grezzo che fa
                                 # stampare una riga: serve a vedere a terra se
                                 # l'asset segue davvero la rotazione fisica

# --- ROTAZIONE DEL RIFERIMENTO (tgt_omega) -----------------------------------
# L'MPC estrapola il bersaglio sull'orizzonte (N*mpc_dt = 1.6 s). Con
# tgt_omega = 0 lo fa in RETTA: su una traiettoria curva il bersaglio predetto
# esce dall'arco verso l'ESTERNO di circa v^2 T^2 / (2R), che a 0.35 m/s, T =
# 1.6 s e R = 3.5 m vale 4.5 cm. E' l'errore radiale misurato nei voli del
# 26/10 mattina (115628 +5.3, 115847 +2.3, 120042 +7.1 cm, tutti verso fuori).
# Passando tgt_omega il controllore estrapola sull'ARCO e quel termine sparisce.
#
# PLAT_TRAJ_MODE: che traiettoria fa la piattaforma in questo volo.
#   "circle" -> omega stimato e passato al controllore
#   "line"   -> omega forzato a ZERO (su una retta non c'e' niente da ruotare,
#               e una stima rumorosa farebbe solo danno)
# Questo main e' solo per i casi in MOVIMENTO: non esiste il caso statico.
PLAT_TRAJ_MODE      = "circle"   # "circle" | "line"

TGT_OMEGA_ENABLED   = True   # False -> si vola come prima (riferimento in retta)
TGT_OMEGA_WIN       = 0.35   # [s] finestra della regressione sullo yaw. NON alzarla
                             # oltre quello che il buffer puo' dare: _plat_buf e'
                             # tappato a PLATFORM_AVG_N pose e il callback del
                             # mocap gira piu' veloce del loop, quindi 50 campioni
                             # coprono MENO di mezzo secondo. Per filtrare di piu'
                             # si alza TGT_OMEGA_TAU, non la finestra.
TGT_OMEGA_SPAN_MIN  = 0.15   # [s] span minimo ASSOLUTO della finestra. Prima qui
                             # c'era 0.5*WIN: alzando WIN la condizione diventava
                             # insoddisfacibile e lo stimatore restava incantato
                             # sullo zero senza dirlo (voli 174512/174717/174912).
TGT_OMEGA_MIN_N     = 7      # campioni minimi nella finestra
TGT_OMEGA_TAU       = 3.5    # [s] costante del passa-basso. lam = span/TAU, quindi
                             # con TAU <= span viene lam = 1 e il filtro NON filtra:
                             # era il caso di TAU = 0.25 (lam saturato a 1.0), da cui
                             # i 0.05 rad/s di rumore del volo 173931. A 3.5 viene
                             # lam = 0.1, cioe' una costante di tempo di ~0.2 s.
TGT_OMEGA_MAX       = 0.40   # [rad/s] TETTO. A 0.35 m/s corrisponde a un raggio
                             # di riferimento di 0.9 m: e' gia' il quadruplo
                             # dell'omega atteso sui cerchi veri (R = 3.5-4.6 m
                             # -> 0.075-0.10 rad/s), quindi lascia margine ma
                             # rende innocua una stima sballata.
TGT_OMEGA_VMIN      = 0.05   # [m/s] sotto questa velocita' di piattaforma
                             # l'omega non e' osservabile: si forza a zero
TGT_OMEGA_JUMP      = 0.17   # [rad] ~10 deg: scarto massimo di un campione di
                             # yaw dalla mediana della finestra. Oltre, il
                             # campione e' una riacquisizione del corpo rigido
                             # e va buttato: la posa ha gia' la sua guardia
                             # (PLATFORM_JUMP_MAX), lo YAW no.
TGT_OMEGA_STATES    = ("hold", "landing")   # dove la rotazione e' attiva

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
HOVER_LAND_MARGIN = 0.0   # con l'MPC offset-free (disturbo stimato, vedi
                           # controller_deploy.dist_*) il margine non serve piu':
                           # sarebbe un bias che lo stimatore compenserebbe.
                           # Prima: 0.01 (effetto suolo nel modello dell'MPC)
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
#
# Il watchdog guarda l'ISTANTE DI ARRIVO del messaggio. Quando Motive perde il
# corpo rigido continua a pubblicare ripetendo l'ultima posa, quindi i messaggi
# arrivano, l'eta' resta zero e il watchdog non vede niente: e' successo nel
# volo 151206 (posa identica per 1.52 s, poi un salto di 1.42 m in un campione)
# e in 154909 (3.74 s). Questo caso NON e' gestito: la colonna pose_age del CSV
# misura il tempo dall'ultima posa diversa, cosi' si vede a posteriori, ma non
# cambia niente in volo.
POSE_TIMEOUT       = 0.25   # eta' massima della posa [s]
POSE_LOST_CUT_H    = 0.30   # sotto questa quota sul suolo: taglio motori [m]
POSE_LOST_VZ       = -0.3   # piu' in alto: discesa livellata a questa vz [m/s],
POSE_LOST_KV       = 2.0    # regolata sulla vz del Kalman (solo IMU) [1/s]
POSE_LOST_T_MARGIN = 1.5    # durata = quota/|vz| + margine, poi taglio [s]

# ---- missione (dinamica, percorso a L con piattaforma mobile) ----------------
TARGET_XY = np.array([1.0, -1.5])       # target parte qui
TARGET_VEL = np.array([0.0, 0.0, 0.0]) # si muove in +y (0.15 m/s; regola tu)
Z_CRUISE = 1.8
Z_LAND = 0.05
Z_HOLD = 1.8
ALPHA_CONE = 2.0
LOS_DELTA = 0.3
A_XY = 0.17
R_BASE_CONE = 0.3   
IDLE, RISING, NAV, HOLD, LANDING = 0, 1, 2, 3, 4

# ---- B2: sequenza di inseguimento con piattaforma mobile ----------------------
# Il drone sale, va al waypoint di nav e LI' RESTA FERMO finche' la piattaforma
# non ha preso il distacco di rilascio (arriva da dietro e va in +x), poi passa
# a hold. In nav_to_wp il feed-forward di velocita' e' nullo
# (vedi current_target_vel), quindi il drone sta davvero fermo mentre la
# piattaforma gli passa sotto; si aggancia solo all'ingresso in hold.
#
# IL RILASCIO E' SU DISTANZA, NON SU TEMPO. Si aspetta che la piattaforma sia
# avanti di PLAT_PASS_GAP, che per default vale la distanza della GENERATRICE
# del cono, (Z_HOLD - Z_LAND)/ALPHA_CONE -- la stessa formula con cui si
# costruisce il punto di hold nel caso statico. Cosi' il drone comincia la
# discesa esattamente sul bordo del cono, nello stesso punto geometrico del
# caso statico, senza doverci volare: ci arriva la piattaforma. L'unica
# differenza fra il blocco statico e quello mobile resta il moto del target,
# che e' quello che si vuole isolare.
# Il criterio su distanza si autocalibra con alpha_cone (1.57 m a 1.0, 0.79 m a
# 2.0) e non va ritarato cambiando la velocita' della piattaforma.
PLAT_CHASE_ENABLED = True
PLAT_PASS_GAP      = None   # [m] distacco di rilascio; None = generatrice del
                            #     cono, calcolata a runtime (Z_LAND viene letto
                            #     dalla piattaforma, quindi non e' nota qui)
PLAT_PASS_MARGIN   = 0.05   # [m] solo per il log del sorpasso: TARGET_XY[0]-pos[0]
                            #     sopra questo = la piattaforma ha passato il drone
                            #     (il target include gia' PLATFORM_XY_OFFSET,
                            #     quindi e' il punto di appoggio, non il pivot)
PLAT_PASS_TIMEOUT  = 40.0   # [s] se il distacco non viene raggiunto entro questo
                            #     tempo dall'ingresso in nav si prosegue lo stesso:
                            #     meglio un volo da scartare che uno appeso
# ---- permanenza minima in hold ---------------------------------------------
# Senza questo l'hold collassa: il gate rilascia quando il distacco raggiunge la
# generatrice del cono, che e' ESATTAMENTE la distanza a cui viene costruito il
# punto di hold, quindi il punto cade sul drone e non resta niente da volare.
# Misurato: 0.02 s di hold (2 campioni) sul volo 150911, con il drone che entra
# in landing a velocita' relativa +0.31 m/s, cioe' fermo rispetto al mondo
# mentre la piattaforma scorre, da 1.58 m di quota e 81 cm di errore laterale.
# Tutto l'inseguimento finisce dentro la discesa.
# Con la permanenza minima il drone resta a Z_HOLD sopra la piattaforma che
# scorre, inseguendo il waypoint di hold che trasla con lei. Il feed-forward di
# velocita' e' gia' attivo in hold, quindi dopo il transitorio il drone si muove
# alla velocita' della piattaforma: la discesa parte a velocita' relativa nulla,
# la stessa condizione iniziale del caso statico. E' quello che rende
# confrontabili il blocco fermo e il blocco in moto.
# Costo in spazio: la piattaforma avanza di v*PLAT_HOLD_MIN. A 0.31 m/s e 2.5 s
# sono 0.78 m, che sul volo 150911 sposterebbero il contatto da x=-0.46 a -1.24
# (i voli del 16/10 atterravano a -1.95 senza problemi di volume mocap).
#
# RIBALTATO il 18/10 dopo il volo 153859: la zona dove cadeva il contatto con
# l'hold lungo (x ~ -1.5) e' scoperta dal mocap. In quel volo la posa del drone
# si e' congelata per 0.548 s a 3.6 cm di quota, e il watchdog non se n'e'
# accorto perche' il mocap continuava a PUBBLICARE ripetendo l'ultimo valore.
# Il contatto cade piu' indietro quanto piu' dura l'hold: a 0.32 m/s ogni
# secondo di hold sposta il touchdown di 32 cm verso -x. Misurato:
#   hold 0.02 s (volo 150911) -> contatto a x = -0.46
#   hold 2.20 s (volo 153859) -> contatto a x = -1.53, congelamento a -1.43
# Quindi l'hold si tiene CORTO e limitato dall'alto, per far cadere il contatto
# a x ~ -0.8/-1.0, dove la copertura e' buona.
# Il prezzo e' che la discesa parte con velocita' relativa non nulla (+0.31 m/s
# nel volo 150911) e tutto l'inseguimento finisce dentro il landing. E' un
# compromesso imposto dal volume di misura, non una scelta di controllo.
PLAT_HOLD_MIN      = 0.0    # [s] permanenza MINIMA. 0 = nessuna attesa
PLAT_HOLD_MAX      = 0.3    # [s] permanenza MASSIMA: scaduta, il landing parte
                            #     comunque, senza aspettare il gate di distanza.
                            #     Serve a rendere l'hold RIPETIBILE: lasciato
                            #     libero e' durato 0.02 s in un volo e 2.20 s in
                            #     un altro, a parita' di configurazione.
                            #     0 = disattivato (nessun limite superiore)


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
    # stabilizer.yaw: imbardata stimata a bordo. E' l'integrazione del
    # giroscopio dal reset dell'estimatore (il Kalman ne stima il bias, ma non
    # c'e' riferimento assoluto), quindi vale psi_fisico - psi_fisico(al reset)
    # e deriva piano. Sta in questo blocco e non in "state", che e' gia' al
    # limite CRTP con i suoi 6 float.
    lg.add_variable("stabilizer.yaw", "float")
    return lg


def reset_estimator(cf):
    cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.5)


def _fmt_cfg(v):
    """Formattazione compatta di un parametro per il file di configurazione."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    if isinstance(v, np.ndarray):
        if v.ndim == 2 and v.shape[0] == v.shape[1]:
            v = np.diag(v)                      # Q/R sono diagonali: basta la diagonale
        return "[" + ", ".join(f"{float(x):.6g}" for x in np.ravel(v)) + "]"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(str(x) for x in v) + ")"
    return str(v)


def _config_snapshot(ctrl=None):
    """Tutti i parametri del volo, raccolti automaticamente.

    Due sorgenti:
      MAIN.*  le costanti in MAIUSCOLO di questo modulo (soglie, abilitazioni,
              geometria, parametri della piattaforma, criteri di contatto);
      CTRL.*  gli attributi scalari o di piccola dimensione del controllore
              (pesi, CBF, effetto suolo, osservatore, imbardata).

    La raccolta e' automatica: aggiungendo un parametro nuovo al controllore o
    una costante nuova qui, finisce nel file senza toccare questa funzione.
    Vengono scartati gli oggetti cvxpy, le matrici del politopo e tutto cio'
    che non e' un numero, un booleano, una stringa o un array piccolo.
    """
    import numbers
    cfg = {}

    def take(prefix, k, v):
        if isinstance(v, str):
            if len(v) <= 120 and "\n" not in v:      # no docstring, no blocchi di testo
                cfg[f"{prefix}.{k}"] = v
        elif isinstance(v, (numbers.Number, bool)):
            cfg[f"{prefix}.{k}"] = _fmt_cfg(v)
        elif isinstance(v, np.ndarray):
            n = v.size if v.ndim == 1 else (v.shape[0] if v.ndim == 2 and v.shape[0] == v.shape[1] else 999)
            if n <= 8:
                cfg[f"{prefix}.{k}"] = _fmt_cfg(v)
        elif isinstance(v, (tuple, list)) and len(v) <= 8 and all(
                isinstance(x, (numbers.Number, bool, str)) for x in v):
            cfg[f"{prefix}.{k}"] = _fmt_cfg(v)

    g = globals()
    for k in sorted(g):
        if k.isupper() and not k.startswith("_"):
            take("MAIN", k, g[k])
    if ctrl is not None:
        for k in sorted(vars(ctrl)):
            if k.startswith("_") or k.startswith("log_") or k.startswith("p_") \
                    or k in ("x_hrz", "u_hrz", "x_vrt", "u_vrt", "prob_hrz", "prob_vrt",
                             "eps_cone", "eps_imp"):
                continue
            take("CTRL", k, vars(ctrl)[k])
    return cfg


def _eps_imp_first(c):
    """Slack della barriera d'impatto al PRIMO passo dell'orizzonte [m/s].

    Il controllore logga solo il MASSIMO sull'orizzonte (log_eps_imp), che non
    dice DOVE sta l'infattibilita': uno slack pagato venti passi avanti, dove la
    traiettoria predetta e' gia' sotto il piatto, e' una cosa diversa da uno
    slack pagato adesso. Il passo 0 e' quello confrontabile con h_v(0) misurato.

    Si legge direttamente dalla variabile cvxpy del controllore, cosi' questa
    diagnostica non richiede modifiche a controller_deploy.py. Se il campo non
    c'e', o il solve verticale non ha prodotto valori, resta NaN e la curva
    semplicemente non compare.
    """
    try:
        if not getattr(c, "imp_enabled", False):
            return float("nan")
        ei = c.eps_imp.value
        if ei is None:
            return float("nan")
        return float(max(float(np.asarray(ei, float).ravel()[0]), 0.0))   # OSQP puo' dare -1e-6
    except Exception:
        return float("nan")


def save_and_plot(rows, ctrl=None):
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
            "az_mpc", "eps0", "eps_max", "ge_az", "ge_h", "d_hat", "d_x", "d_y",
            "z_ground", "ge_ground", "hv0", "eps_imp", "eps_imp0", "yaw_drone", "yaw_err", "yaw_rate",
            "yaw_plat", "yaw_plat_raw", "pose_age",
            "plat_x_raw", "plat_y_raw", "plat_z_raw", "plat_skip",
            "plat_age", "e_x", "e_y", "tgt_omega"]
    # --- configurazione del volo, salvata insieme ai dati ---------------------
    # Due copie, per motivi diversi:
    #  - un file di testo accanto al CSV, leggibile e greppabile ("quale volo
    #    aveva imp_kv = 0.5?" -> grep);
    #  - una colonna "cfg" nel CSV, riempita solo nella prima riga, cosi' la
    #    configurazione viaggia con i dati anche se il file viene spostato.
    #    Una colonna in piu' vuota non disturba pandas ne' i plot, perche' "cfg"
    #    non entra nella lista cols usata per costruire gli array.
    cfg = _config_snapshot(ctrl)
    cfg_path = f"last_run_plots/flight_{stamp}_config.txt"
    with open(cfg_path, "w") as f:
        f.write(f"# volo {stamp}\n")
        for k, v in cfg.items():
            f.write(f"{k} = {v}\n")
    cfg_str = "; ".join(f"{k}={v}" for k, v in cfg.items())

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols + ["cfg"])
        for i, r in enumerate(rows):
            w.writerow([r[c] for c in cols] + [cfg_str if i == 0 else ""])
    print(f"CSV salvato: {csv_path}  ({len(rows)} righe)")
    print(f"configurazione salvata: {cfg_path}  ({len(cfg)} parametri)")
    for k in ("CTRL.imp_kv", "CTRL.imp_vtd", "CTRL.R_hrz", "CTRL.R_vrt",
              "CTRL.att_tau", "CTRL.vrt_tau", "CTRL.att_lag_enabled",
              "CTRL.vrt_lag_enabled", "CTRL.imp_enabled", "CTRL.dist_freeze_h",
              "MAIN.YAW_ACTIVE_STATES"):
        if k in cfg:
            print(f"   {k:28s} = {cfg[k]}")

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
    ge_sat = G * (CTRL_GE_K_MAX - 1.0)
    ax[3].plot(t, a["ge_az"], color="tab:brown", lw=1.2, ls=":", label="effetto suolo (modello)")
    if np.isfinite(a["ge_az"]).any() and np.nanmax(a["ge_az"]) > 0.9 * ge_sat:
        ax[3].axhline(ge_sat, color="tab:brown", lw=0.7, ls="--",
                      label=f"clamp k_max ({ge_sat:.2f})")
    ax[3].plot(t, a["d_hat"], color="tab:olive", lw=1.4, ls="-.", label="disturbo stimato d")
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
    ax7b = ax[7].twinx()
    ax7b.plot(t, a["yaw_drone"], color="k", lw=1.0, label="yaw drone [deg]")
    ax7b.plot(t, a["yaw_err"], color="tab:red", lw=1.3, label="errore yaw [deg]")
    ax7b.plot(t, a["yaw_rate"], color="tab:green", lw=1.0, ls=":", label="rate yaw [deg/s]")
    if "yaw_plat" in a and np.isfinite(a["yaw_plat"]).any():
        ax7b.plot(t, a["yaw_plat"], color="tab:purple", lw=1.0, ls="--", label="yaw piattaforma [deg]")
    ax7b.axhline(0, color="0.5", lw=0.4)
    ax7b.set_ylabel("yaw [deg, deg/s]"); ax7b.legend(loc="lower right", fontsize=7)
    ax[7].plot(t, a["d_x"], color="tab:blue", lw=1.2, ls="-.", label="d_x stimato")
    ax[7].plot(t, a["d_y"], color="tab:orange", lw=1.2, ls="-.", label="d_y stimato")
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
    ax[10].plot(t, a["eps_imp"], color="tab:blue", lw=1.2, ls="-.",
                label="slack impatto, max orizzonte")
    ax[10].plot(t, a["eps_imp0"], color="tab:cyan", lw=1.6,
                label="slack impatto, PRIMO PASSO")
    ax[10].plot(t, a["hv0"], color="tab:green", lw=1.3,
                label="h_v (margine di impatto) [m/s]")
    ax[10].axhline(0, color="k", lw=0.6)
    ax[10].set_ylabel("slack cono [m] / h_v"); ax[10].legend(loc="upper right", fontsize=7)
    shade_mpc(ax[10])

    # --- quota sopra il suolo, per leggere l'effetto suolo sui pannelli sopra ---
    h_gnd = a["z"] - a["z_ground"]
    ax[11].plot(t, h_gnd, color="tab:green", lw=1.3, label="z - quota di contatto")
    ax[11].plot(t, a["ge_h"], color="tab:brown", lw=1.2, ls="--",
                label="h usata dall'effetto suolo")
    ax[11].axhline(HOVER_EST_Z_MIN - np.nanmean(a["z_ground"]), color="k", lw=0.7, ls=":",
                   label="stimatori congelati sotto")
    ax[11].set_ylabel("quote [m]"); ax[11].set_xlabel("t [s]")
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
    ye = a["yaw_err"][np.isfinite(a["yaw_err"])]
    yd = a["yaw_drone"][np.isfinite(a["yaw_drone"])]
    if len(ye):
        print(f"yaw: errore iniziale {ye[0]:+.1f} deg, finale {ye[-1]:+.1f} deg, "
              f"max |.| {np.abs(ye).max():.1f} deg; drone da {yd[0]:+.1f} a {yd[-1]:+.1f} deg")
    if "yaw_plat" in a:
        yp = a["yaw_plat"][np.isfinite(a["yaw_plat"])]
        if len(yp):
            msg = (f"yaw piattaforma (corretto): da {yp[0]:+.1f} a {yp[-1]:+.1f} deg, "
                   f"escursione {np.ptp(yp):.1f} deg")
            if "yaw_plat_raw" in a:
                yr = a["yaw_plat_raw"][np.isfinite(a["yaw_plat_raw"])]
                if len(yr):
                    msg += (f" | grezzo da {yr[0]:+.1f} a {yr[-1]:+.1f} deg "
                            f"(escursione {np.ptp(yr):.1f} deg)")
                    if np.ptp(yr) < 1.0:
                        msg += "  <-- la piattaforma NON ha ruotato nella lettura mocap"
            print(msg)
    if "pose_age" in a:
        pa = a["pose_age"][np.isfinite(a["pose_age"])]
        if len(pa):
            k = int(np.argmax(pa))
            print(f"posa mocap: eta' mediana {np.median(pa)*1e3:.1f} ms, "
                  f"99mo pct {np.percentile(pa,99)*1e3:.1f} ms, "
                  f"massima {pa.max()*1e3:.0f} ms a t={a['t'][np.isfinite(a['pose_age'])][k]:.2f} s"
                  + ("   <-- POSA CONGELATA, volo alla cieca" if pa.max() > 0.08 else ""))
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
    plot_final_phase(a, state_str, stamp, plt)
    plot_ground_effect(a, state_str, stamp, plt)
    plt.show()

    return stamp

# ---------------------------------------------------------------------------
# ZOOM SULLA FASE FINALE
# Tutte le grandezze che contano negli ultimi centimetri su una scala in cui si
# vedano: quota, chiusura verticale, errore xy, effetto suolo, offset-free,
# barriere. Asse dei tempi riferito al contatto (t=0), cosi' due voli diversi
# sono sovrapponibili.
# ---------------------------------------------------------------------------
FINAL_H_WINDOW = 0.35      # [m] quota sopra il contatto da cui parte lo zoom
FINAL_T_AFTER  = 0.40      # [s] coda da mostrare dopo il contatto
IMPACT_WIN     = 0.20      # [s] finestra PRIMA del flag di contatto su cui si
                           # cerca il minimo di vz. MISURA DI RISERVA dal 10/10:
                           # vale solo se il flag arriva entro IMPACT_WIN dal
                           # contatto vero. Vedi IMPACT_ACC_JUMP.
# --- velocita' al PRIMO contatto, indipendente dal flag -----------------------
# La finestra sopra e' ancorata al FLAG, che e' una rilevazione, non l'urto. La
# regola "quota ferma" richiede 0.20 s di quota immobile prima di scattare,
# quindi e' strutturalmente in ritardo di almeno due decimi: quando il ritardo
# supera IMPACT_WIN la finestra cade TUTTA DOPO l'urto e il minimo che ci trovi
# e' l'assestamento, non l'impatto. Volo 115522 del 10/10: urto a t-0.28 s con
# vz = -0.134 e acc.z 1.37 g, flag 0.28 s dopo, finestra da -0.20 a 0.00 tutta
# post-urto, minimo -0.041. Errore 0.093 m/s sulla grandezza che si dichiara.
# Qui il contatto si cerca nel segnale: PRIMO campione in cui acc.z supera di
# IMPACT_ACC_JUMP la mediana dei IMPACT_ACC_WIN precedenti, sotto IMPACT_ACC_H.
#   - PRIMO e non massimo: il drone rimbalza e il secondo tocco puo' essere piu'
#     duro del primo. Su 115522 il massimo era il rimbalzo a t-0.11 (1.47 g) e
#     dava 0.022 invece di 0.134.
#   - salto sulla MEDIANA e non soglia assoluta: appoggiato a terra acc.z legge
#     ~1.0 g come in hover, la firma e' il salto rispetto al livello recente.
#   - cancello sulla quota: una raffica piu' in alto non e' un contatto.
# Sui nove voli del 9-10/10 con firma netta le due misure coincidono entro
# 0.006 m/s; divergono solo dove la rilevazione e' in ritardo. Se il salto non
# si trova il contatto NON e' databile (tocchi sotto ~1.2 g): si scrive n/d, non
# si inventa un numero - vale come informazione a se'.
IMPACT_ACC_JUMP = 0.35     # [g] salto di acc.z sopra la mediana recente. Meta'
                           # di CONTACT_ACC_JUMP: qui si analizza a terra e si
                           # vogliono prendere anche i tocchi morbidi, e il
                           # cancello sulla quota protegge dai falsi.
IMPACT_ACC_WIN  = 0.40     # [s] finestra per la mediana di riferimento
IMPACT_ACC_H    = 0.03     # [m] quota massima perche' un salto sia un contatto
IMPACT_ACC_AFTER = 0.02    # [s] tolleranza oltre i_td. i_td e' il campione di
                           # QUOTA MINIMA (vedi _touchdown_index), non il flag di
                           # stato: un contatto non puo' stare dopo il punto piu'
                           # basso, quindi la ricerca si chiude li'. Con una
                           # finestra piu' larga si prendono i rimbalzi: sui voli
                           # 141648 e 145951 dell'8-9/10 dava -0.218 e -0.216,
                           # cioe' la caduta dopo il taglio dei motori, invece di
                           # -0.055 e -0.040.
STALL_BAND     = (0.01, 0.08)   # [m] fascia sopra il contatto in cui si misura
                           # lo stallo (minimo di |vz| e tempo da li' al
                           # contatto). 1-8 cm: sotto 1 cm il contatto e' in
                           # corso, sopra 8 cm la discesa e' ancora in frenata.


POSE_VALID_AGE = 0.15      # [s] eta' massima di pose_age/plat_age perche' un
                           # campione valga. In volo sano il massimo misurato e'
                           # 0.10 s; un congelamento patologico arriva a 0.59.


def _touchdown_index(a, state_str):
    """Indice del campione di CONTATTO.

    E' il primo campione in stato "touchdown", cioe' l'istante in cui il
    rilevatore di contatto della FSM ha scattato. NON e' il minimo di quota:
    dopo il contatto il drone resta appoggiato mentre la piattaforma continua a
    scorrere, quindi l'errore letto piu' avanti cresce di qualche centimetro
    senza che il drone si sia mosso, e non e' un errore di atterraggio.
    Stessa ragione per cui non si usa l'ultimo campione della registrazione.

    L'indice viene poi arretrato fino al primo campione VALIDO: se al contatto
    la posa del drone o quella della piattaforma erano congelate, il valore
    letto li' e' un fotogramma vecchio (volo 153859: posa ferma 0.548 s prima
    del contatto, errore apparente 7.8 cm contro 2.0 cm reali).

    Fallback per i CSV vecchi senza lo stato touchdown: minimo di quota nella
    fase di landing, come prima.
    """
    h = a["z"] - a["z_ground"]
    td = np.where(state_str == "touchdown")[0]
    if len(td):
        k = int(td[0])
    else:
        land = np.where(state_str == "landing")[0]
        pool = land if len(land) else np.arange(len(h))
        seg = h[pool]
        if not np.isfinite(seg).any():
            return int(pool[-1])
        k = int(pool[int(np.nanargmin(np.where(np.isfinite(seg), seg, np.inf)))])
    # arretramento sul primo campione con pose fresche
    def _age(name):
        v = a.get(name)
        if v is None or len(np.atleast_1d(v)) != len(h):
            return np.zeros(len(h))
        return np.nan_to_num(np.asarray(v, dtype=float), nan=0.0)
    bad = (_age("pose_age") > POSE_VALID_AGE) | (_age("plat_age") > POSE_VALID_AGE)
    k0 = k
    while k > 0 and bad[k]:
        k -= 1
    if k != k0:
        print(f"[PLOT] contatto arretrato da {k0} a {k} "
              f"({a['t'][k0]-a['t'][k]:.3f} s): pose non fresche al touchdown")
    return k


def plot_final_phase(a, state_str, stamp, plt,
                     h_window=FINAL_H_WINDOW, t_after=FINAL_T_AFTER):
    # i CSV piu' vecchi non hanno tutte le colonne: quelle mancanti diventano NaN
    # cosi' il plot si genera lo stesso con le curve vuote.
    n = len(a["t"])
    for c in ("z_ground", "ge_h", "ge_az", "hv0", "eps0", "eps_max", "eps_imp", "eps_imp0",
              "d_hat", "d_x", "d_y", "az_mpc", "accz", "hover_cmd",
              "target_x", "target_y", "target_vz", "ref_vz", "mode"):
        if c not in a or len(np.atleast_1d(a[c])) != n:
            a[c] = np.full(n, np.nan)
    h = a["z"] - a["z_ground"]
    i_td = _touchdown_index(a, state_str)
    dt_s = float(np.median(np.diff(a["t"]))) if len(a["t"]) > 2 else DT
    pre = np.where(h[:i_td + 1] > h_window)[0]
    i0 = int(pre[-1]) if len(pre) else 0
    i1 = int(min(len(h) - 1, i_td + round(t_after / max(dt_s, 1e-3))))
    if i1 - i0 < 10:
        print("fase finale troppo corta: zoom non generato"); return
    s = slice(i0, i1 + 1)
    tt = a["t"][s] - a["t"][i_td]           # 0 = contatto

    # errore: si usano le colonne e_x/e_y calcolate in volo dalla posa GREZZA
    # della piattaforma. target_x/target_y si congelano di un tick in touchdown
    # (_track_platform non gira in quello stato) e a ogni uscita anticipata di
    # _track_platform, quindi ricostruire l'errore da li' introduce un bias.
    if "e_x" in a and len(np.atleast_1d(a["e_x"])) == n:
        e_x = np.asarray(a["e_x"], dtype=float)[s]
        e_y = np.asarray(a["e_y"], dtype=float)[s]
    else:
        e_x = (a["x"] - a["target_x"])[s]; e_y = (a["y"] - a["target_y"])[s]
    e_n = np.hypot(e_x, e_y)
    az_meas = (a["accz"] - 1.0) * G
    ax_est = G * np.tan(a["pitch"]); ay_est = -G * np.tan(a["roll"])

    fig = plt.figure(figsize=(13.5, 13))
    gs = fig.add_gridspec(7, 2, width_ratios=[2.15, 1.0], hspace=0.10, wspace=0.26,
                          left=0.06, right=0.97, top=0.955, bottom=0.045)
    P = [fig.add_subplot(gs[0, 0])]
    for i in range(1, 7):
        P.append(fig.add_subplot(gs[i, 0], sharex=P[0]))
    for A in P[:-1]:
        A.tick_params(labelbottom=False)

    def mark(A):
        A.axvline(0.0, color="k", lw=1.1, ls="--", alpha=0.7)
        on = (a["mode"] > 0.5)[s]
        if on.any():
            start = None
            for i in range(len(on)):
                if on[i] and start is None:
                    start = tt[i]
                if (not on[i] or i == len(on) - 1) and start is not None:
                    A.axvspan(start, tt[i], color="orange", alpha=0.10); start = None
        A.grid(alpha=0.25, lw=0.5)

    # 0 - quota sopra il contatto
    P[0].plot(tt, 100 * h[s], color="tab:green", lw=1.8, label="z - quota di contatto")
    P[0].plot(tt, 100 * a["ge_h"][s], color="tab:brown", lw=1.2, ls="--",
              label="h usata dall'effetto suolo")
    P[0].axhline(0, color="k", lw=0.8)
    P[0].set_ylabel("quota [cm]"); P[0].legend(loc="upper right", fontsize=8); mark(P[0])

    # 1 - chiusura verticale
    P[1].plot(tt, 100 * a["vz"][s], color="tab:green", lw=1.5, label="vz drone")
    P[1].plot(tt, 100 * a["target_vz"][s], color="k", lw=1.0, ls=":", label="vz piattaforma")
    P[1].plot(tt, 100 * (a["vz"] - a["target_vz"])[s], color="tab:red", lw=1.3, ls="-.",
              label="chiusura relativa")
    P[1].plot(tt, 100 * a["ref_vz"][s], color="tab:green", lw=0.9, ls="--", label="ref vz")
    P[1].axhline(0, color="k", lw=0.6)
    P[1].set_ylabel("vz [cm/s]"); P[1].legend(loc="lower right", fontsize=8); mark(P[1])

    # 2 - errore orizzontale
    P[2].plot(tt, 100 * e_x, color="tab:blue", lw=1.1, label="e_x")
    P[2].plot(tt, 100 * e_y, color="tab:orange", lw=1.1, label="e_y")
    P[2].plot(tt, 100 * e_n, color="k", lw=1.8, label="|e_xy|")
    P[2].axhline(0, color="k", lw=0.6)
    P[2].set_ylabel("errore xy [cm]"); P[2].legend(loc="upper right", fontsize=8); mark(P[2])

    # 3 - accelerazione verticale: MPC, effetto suolo, offset-free
    P[3].plot(tt, az_meas[s], color="0.65", lw=0.7, label="misurata (acc.z)")
    P[3].plot(tt, a["az"][s], color="tab:purple", lw=1.6, label="comandata")
    P[3].plot(tt, a["az_mpc"][s], color="tab:red", lw=1.2, ls="--", label="uscita MPC")
    P[3].plot(tt, a["ge_az"][s], color="tab:brown", lw=1.4, ls=":", label="effetto suolo")
    P[3].plot(tt, a["d_hat"][s], color="tab:olive", lw=1.5, ls="-.", label="offset-free d_z")
    ge_sat = G * (CTRL_GE_K_MAX - 1.0)
    if np.isfinite(a["ge_az"][s]).any() and np.nanmax(a["ge_az"][s]) > 0.9 * ge_sat:
        P[3].axhline(ge_sat, color="tab:brown", lw=0.8, ls="--",
                     label=f"clamp k_max ({ge_sat:.2f})")
    P[3].axhline(0, color="k", lw=0.6)
    P[3].set_ylabel("az [m/s^2]"); P[3].legend(loc="upper right", fontsize=7, ncol=3); mark(P[3])

    # 4 - accelerazione orizzontale e offset-free xy
    P[4].plot(tt, ax_est[s], color="tab:blue", lw=1.4, label="ax comandata")
    P[4].plot(tt, ay_est[s], color="tab:orange", lw=1.4, label="ay comandata")
    P[4].plot(tt, a["d_x"][s], color="tab:blue", lw=1.3, ls="-.", label="offset-free d_x")
    P[4].plot(tt, a["d_y"][s], color="tab:orange", lw=1.3, ls="-.", label="offset-free d_y")
    P[4].axhline(0, color="k", lw=0.6)
    P[4].set_ylabel("a_xy [m/s^2]"); P[4].legend(loc="upper right", fontsize=7, ncol=2); mark(P[4])

    # 5 - comando di spinta
    P[5].plot(tt, a["cmd"][s], color="tab:red", lw=1.4, label="cmd applicato")
    P[5].plot(tt, a["hover_cmd"][s], color="k", lw=1.0, ls=":", label="hover cmd")
    P[5].set_ylabel("thrust [cmd]"); P[5].legend(loc="upper right", fontsize=8); mark(P[5])

    # 6 - barriere: h_v a sinistra (m/s), slack a destra in mm (sono millimetrici,
    #     sulla stessa scala di h_v sarebbero una riga piatta sullo zero)
    P[6].plot(tt, a["hv0"][s], color="tab:green", lw=1.6, label="h_v (margine impatto) [m/s]")
    P[6].axhline(0, color="k", lw=0.8)
    P[6].set_ylabel("h_v [m/s]")
    P6b = P[6].twinx()
    P6b.plot(tt, 1e3 * a["eps0"][s], color="tab:red", lw=1.2, label="slack cono, primo passo")
    P6b.plot(tt, 1e3 * a["eps_max"][s], color="tab:brown", lw=1.0, ls="--", label="slack cono, max")
    P6b.plot(tt, 1e3 * a["eps_imp"][s], color="tab:blue", lw=1.2, ls="-.", label="slack impatto, max")
    P6b.plot(tt, 1e3 * a["eps_imp0"][s], color="tab:cyan", lw=1.6, label="slack impatto, primo passo")
    P6b.set_ylabel("slack [mm]")
    hl, ll = P[6].get_legend_handles_labels(); h2, l2 = P6b.get_legend_handles_labels()
    P[6].legend(hl + h2, ll + l2, loc="upper right", fontsize=7)
    P[6].set_xlabel("t - t_contatto [s]"); mark(P[6])

    # --- vista dall'alto degli ultimi centimetri -------------------------
    AT = fig.add_subplot(gs[0:3, 1])
    AT.plot(a["target_x"][s], a["target_y"][s], color="k", lw=1.4, ls=":", label="piattaforma")
    sc = AT.scatter(a["x"][s], a["y"][s], c=100 * h[s], cmap="viridis_r", s=14, zorder=3)
    AT.plot(a["x"][s], a["y"][s], color="0.6", lw=0.7, zorder=2)
    AT.scatter([a["x"][i0]], [a["y"][i0]], marker="o", s=70, facecolor="none",
               edgecolor="tab:blue", lw=1.6, zorder=4, label=f"inizio (h={100*h[i0]:.0f} cm)")
    AT.scatter([a["x"][i_td]], [a["y"][i_td]], marker="X", s=110, color="tab:red",
               zorder=5, label="contatto drone")
    AT.scatter([a["target_x"][i_td]], [a["target_y"][i_td]], marker="P", s=110,
               color="k", zorder=5, label="piattaforma al contatto")
    # limiti quadrati e stretti sui dati, altrimenti aspect="equal" dilata l'asse y
    xs = np.concatenate([a["x"][s], a["target_x"][s]]); ys = np.concatenate([a["y"][s], a["target_y"][s]])
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    if len(xs) and len(ys):
        cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
        half = 0.5 * max(np.ptp(xs), np.ptp(ys), 0.10) * 1.25
        AT.set_xlim(cx - half, cx + half); AT.set_ylim(cy - half, cy + half)
    AT.set_aspect("equal", adjustable="box")
    AT.set_xlabel("x [m]", labelpad=1); AT.set_ylabel("y [m]")
    AT.set_title("vista dall'alto, ultimi cm", fontsize=9)
    AT.legend(loc="best", fontsize=7); AT.grid(alpha=0.25, lw=0.5)
    cb = fig.colorbar(sc, ax=AT, fraction=0.045, pad=0.03)
    cb.set_label("quota [cm]", fontsize=8); cb.ax.tick_params(labelsize=7)

    # --- riquadro con i numeri del contatto ------------------------------
    AI = fig.add_subplot(gs[3:, 1]); AI.axis("off")
    mpc_frac = float(np.mean((a["mode"] > 0.5)[s])) * 100.0
    eps_act = float(np.mean(a["eps_max"][s] > 1e-4) * 100.0) if np.isfinite(a["eps_max"][s]).any() else float("nan")
    hv_min = float(np.nanmin(a["hv0"][s])) if np.isfinite(a["hv0"][s]).any() else float("nan")
    ge_max = float(np.nanmax(a["ge_az"][s])) if np.isfinite(a["ge_az"][s]).any() else float("nan")
    # VELOCITA' DI IMPATTO. a['vz'][i_td] e' il campione DEL flag di contatto, e
    # il rilevatore scatta sul picco di acc.z, cioe' a urto gia' avvenuto: li' il
    # rimbalzo e' iniziato e il valore puo' essere positivo. Confrontare due voli
    # su quel campione e' un artefatto di campionamento - i voli 181044 e 181256
    # dell'8/10 danno +0.033 contro -0.038 su quel campione, mentre l'impatto
    # vero e' -0.089 in entrambi. L'impatto e' il MINIMO di vz nella finestra
    # IMPACT_WIN che precede il flag; il valore al flag resta stampato sotto,
    # come riferimento, ma non e' la grandezza da confrontare fra voli.
    m_imp = (a["t"] <= a["t"][i_td]) & (a["t"] >= a["t"][i_td] - IMPACT_WIN)
    vz_win = float(np.nanmin(a["vz"][m_imp])) if m_imp.any() else float("nan")
    # PRIMO CONTATTO, indipendente dal flag: primo salto di acc.z sotto quota.
    # Vedi il blocco IMPACT_ACC_* in testa al file. Si lavora sui campioni a
    # tempo DISTINTO perche' il CSV ripete le righe fra un tick e l'altro.
    # La scansione parte da i0 (ingresso nella finestra finale, h_window sopra
    # il contatto): piu' in alto non c'e' niente da trovare, e partendo
    # dall'inizio del volo il decollo - dove h e' piccola e acc.z salta - puo'
    # essere scambiato per un contatto.
    i_cont = None
    _tu, _iu = np.unique(np.round(a["t"], 4), return_index=True)
    _iu = np.sort(_iu)
    _iu = _iu[_iu >= i0]
    for _p, _k in enumerate(_iu):
        if a["t"][_k] > a["t"][i_td] + IMPACT_ACC_AFTER:
            break                      # oltre: i motori sono gia' a zero
        if _p < 8:
            continue
        if not (np.isfinite(h[_k]) and h[_k] < IMPACT_ACC_H):
            continue
        _ref = a["accz"][_iu[max(0, _p - 20):_p - 2]]
        _ref = _ref[np.isfinite(_ref)]
        if _ref.size < 6:
            continue
        if a["accz"][_k] > float(np.median(_ref)) + IMPACT_ACC_JUMP:
            i_cont = _k if _p == 0 else _iu[_p - 1]   # campione PRIMA dell'urto
            break
    vz_imp = float(a["vz"][i_cont]) if i_cont is not None else float("nan")
    vz_imp_rel = vz_imp - float(a["target_vz"][i_td])
    # stringhe del riquadro: n/d quando il contatto non e' databile, e una
    # bandiera quando le due misure divergono (= il flag e' arrivato tardi)
    s_imp = f"{vz_imp:+6.3f}" if np.isfinite(vz_imp) else "   n/d"
    s_rel = f"{vz_imp_rel:+6.3f}" if np.isfinite(vz_imp_rel) else "   n/d"
    s_cont = ("" if i_cont is None else
              f"  (h={100*h[i_cont]:.2f} cm, {a['t'][i_cont]-a['t'][i_td]:+.2f} s da quota min.)")
    s_div = ("   <-- DIVERGONO: contatto da controllare"
             if (np.isfinite(vz_imp) and np.isfinite(vz_win)
                 and abs(abs(vz_imp) - abs(vz_win)) > 0.015) else "")
    # STALLO. Minimo di |vz| nella fascia STALL_BAND sopra il contatto e tempo
    # speso da li' al contatto: e' la coppia di numeri che separa i voli con e
    # senza riferimento di discesa (vedi vz_ref_land nel controllore).
    m_st = (h > STALL_BAND[0]) & (h < STALL_BAND[1]) & (np.arange(n) <= i_td)
    if m_st.any():
        j_st = int(np.flatnonzero(m_st)[np.nanargmin(np.abs(a["vz"][m_st]))])
        vz_stall = float(abs(a["vz"][j_st])); h_stall = float(h[j_st])
        dwell = float(a["t"][i_td] - a["t"][j_st])
    else:
        vz_stall = h_stall = dwell = float("nan")
    txt = (
        f"finestra: h da {100*h[i0]:.0f} cm a contatto, {tt[-1]-tt[0]:.2f} s\n"
        f"in MPC: {mpc_frac:.0f}% dei campioni\n\n"
        f"AL CONTATTO (campione valido)\n"
        f"  errore xy      {100*e_n[i_td-i0]:6.2f} cm\n"
        f"  e_x, e_y      {100*e_x[i_td-i0]:+6.2f}, {100*e_y[i_td-i0]:+.2f} cm\n"
        f"  quota          {100*h[i_td]:6.2f} cm\n"
        f"  vz 1o contatto {s_imp} m/s{s_cont}\n"
        f"  vz impatto rel.{s_rel} m/s\n"
        f"  vz in finestra {vz_win:+6.3f} m/s{s_div}\n"
        f"  vz al flag     {a['vz'][i_td]:+6.3f} m/s\n\n"
        f"STALLO ({100*STALL_BAND[0]:.0f}-{100*STALL_BAND[1]:.0f} cm)\n"
        f"  |vz| minimo    {vz_stall:6.3f} m/s\n"
        f"  a quota        {100*h_stall:6.2f} cm\n"
        f"  dwell          {dwell:6.2f} s\n\n"
        f"NELLA FINESTRA\n"
        f"  |e_xy| max     {100*np.nanmax(e_n):6.2f} cm\n"
        f"  |e_xy| min     {100*np.nanmin(e_n):6.2f} cm\n"
        f"  h_v minimo     {hv_min:+6.3f}\n"
        f"  slack cono att.{eps_act:5.0f}%\n"
        f"  effetto suolo  {ge_max:6.2f} m/s^2 max\n"
        f"  d_z finale     {a['d_hat'][i_td]:+6.3f} m/s^2\n"
        f"  d_xy finale    ({a['d_x'][i_td]:+.3f}, {a['d_y'][i_td]:+.3f}) m/s^2"
    )
    AI.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=8.5)

    fig.suptitle(f"Volo {stamp} — fase finale (ultimi {100*h[i0]:.0f} cm)", y=0.995)
    png = f"last_run_plots/flight_{stamp}_finale.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print(f"plot fase finale salvato: {png}")


def replot_csv(path):
    """Rigenera i plot da un CSV gia' salvato, senza volare.
    Uso:  python3 -c "import main_rising_test as m; m.replot_csv('last_run_plots/flight_XXXX.csv')"
    """
    import matplotlib.pyplot as plt
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k in ("state", "cfg")
                             else (float(v) if v not in ("", "nan") else float("nan")))
                         for k, v in r.items()})
    if not rows:
        print("CSV vuoto"); return
    cols = [c for c in rows[0].keys()]
    a = {c: np.array([r[c] for r in rows], dtype=float)
         for c in cols if c not in ("state", "cfg")}
    state_str = np.array([r["state"] for r in rows])
    stamp = path.split("flight_")[-1].replace(".csv", "")
    plot_final_phase(a, state_str, stamp, plt)
    try:
        plot_ground_effect(a, state_str, stamp, plt)
    except KeyError as e:
        print(f"effetto suolo: colonna {e} assente in questo CSV, plot saltato")
    plt.show()


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
    # h = distanza piano rotori - superficie, la stessa che usa il modello.
    # Prima qui si usava z_ground (quota di contatto): h sottostimata di ~3.5 cm,
    # ed e' il motivo per cui il fit precedente ha dato r_eff = 41 mm.
    h = (a["z"] - a["ge_ground"])[land]
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
    ax.plot(hh, np.minimum(1.0 / np.maximum(1.0 - (CTRL_GE_R_EFF / (4 * hh)) ** 2, 1e-3),
                           CTRL_GE_K_MAX), "b-", lw=1.2, alpha=0.8,
            label=f"modello a bordo (R={CTRL_GE_R_EFF*1000:.0f} mm, clamp {CTRL_GE_K_MAX})")
    # --- rifit di r_eff sulla h corretta -------------------------------------
    # 1 - 1/k = (r/4h)^2  ->  r = 4h sqrt(1 - 1/k), stima robusta sui campioni
    # con effetto misurabile (k > 1.005) e dentro la validita' del modello.
    r_fit = float("nan")
    fm = (k > 1.005) & (h > 0.02) & np.isfinite(h) & np.isfinite(k)
    if fm.sum() >= 10:
        r_fit = float(np.median(4.0 * h[fm] * np.sqrt(1.0 - 1.0 / k[fm])))
        ax.plot(hh, np.minimum(1.0 / np.maximum(1.0 - (r_fit / (4 * hh)) ** 2, 1e-3), 2.0),
                "g-", lw=1.8, label=f"rifit su questo volo: R={r_fit*1000:.0f} mm")
        print(f"effetto suolo: r_eff rifittata = {r_fit*1000:.1f} mm "
              f"su {fm.sum()} campioni (a bordo {CTRL_GE_R_EFF*1000:.0f} mm)")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axhline(CTRL_GE_K_MAX, color="b", lw=0.6, ls=":")
    ax.axvline(HOVER_EST_Z_MIN - float(np.nanmean(a["ge_ground"])), color="0.5", ls=":",
               label="soglia congelamento stimatori")
    ax.set_xlabel("distanza piano rotori - superficie [m]")
    ax.set_ylabel("T_IGE / T_OGE  (= hover_hold / (cmd/acc.z))")
    ax.set_title(f"Effetto suolo nel landing (rif. hold = {ref:.0f})")
    ax.set_ylim(0.8, 1.3); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = f"last_run_plots/flight_{stamp}_ground_effect.png"
    fig.savefig(png, dpi=110)
    if med:
        print(f"effetto suolo: rapporto mediano {med[0]:.3f} a h={mid[0]:.3f} m "
              f"(rif. hold {ref:.0f}); plot {png}")


def plot_advanced_diagnostics(rows, stamp=None, alpha_cone=1.0, z_cut=1.5, r_base=0.1, polytope_path="reachable_polytope.npz"):
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
        self.ctrl.yaw_enabled = YAW_ENABLED
        self.ctrl.yaw_kp = YAW_KP
        self.ctrl.yaw_rate_max = math.radians(YAW_RATE_MAX)
        self.ctrl.yaw_deadband = math.radians(YAW_DEADBAND)
        self.ctrl.yaw_symmetry = math.radians(YAW_SYMMETRY)
        self.ctrl.yaw_rate_sign = YAW_RATE_SIGN
        self.latest_pose_t = None       # istante di arrivo dell'ultima posa
        self._pose_lost_t0 = None       # inizio discesa per posa persa
        self._pose_prev = None          # ultima terna xyz ricevuta (solo log)
        self._pose_change_t = None      # istante dell'ultima posa DIVERSA (solo log)
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
        self.V_HOLD = 0.6     # velocita' carrot in hold [m/s]. Alzata da 0.3 per
                              # l'inseguimento della piattaforma mobile: in hold il
                              # feed-forward e' gia' attivo, quindi questa e' la
                              # velocita' di chiusura RELATIVA alla piattaforma e
                              # con 0.3 il drone impiegava troppo a raggiungere il
                              # punto di hold che scorre. NB: la distanza di
                              # decelerazione del trapezio vale 0.5*V*RAMP_T, qui
                              # 0.30 m, cioe' esattamente lo stop_delta dell'hold:
                              # oltre 0.6 la decelerazione comincia fuori dal
                              # raggio di transizione (vedi nota su carrot_arrived).

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
        self._tgt_omega = 0.0               # [rad/s] rotazione del riferimento
        self._tgt_omega_warned = False      # avviso 'finestra corta' gia' stampato
        self.plat_xyz = None                # posa fissata come target
        self._plat_raw = None               # ultima posa piattaforma GREZZA
        self._plat_change_t = None          # istante dell'ultima posa plat DIVERSA
        self._plat_skip = 0                 # scarti consecutivi della soglia
        self._plat_skip_tot = 0             # scarti totali, per il log
        self._plat_t = None                 # istante dell'ultima posa piattaforma
        self._plat_warn_t = 0.0             # anti-spam dei warning sulle pose
        self._plat_vel = np.zeros(3)        # [m/s] velocita' stimata (filtrata)
        self._plat_vel_t = None             # istante dell'ultimo aggiornamento
        self.plat_yaw_off = None            # [rad] yaw mocap a piattaforma zero
        self.plat_yaw = float("nan")        # [rad] yaw CORRETTO in uso
        self.plat_yaw_raw = float("nan")    # [rad] yaw GREZZO del mocap
        self._plat_yaw_logged = 0.0         # ultimo grezzo stampato
        self._plat_pass_t = None            # istante del sorpasso (B2)
        self._hold_t0 = None                # istante di ingresso in hold (B2)
        self._nav_arrive_t = None           # istante di arrivo al nav (B2)
        self._rest = None                   # quota del drone appoggiato [m]
        self._land_ground = None            # "suolo" di atterraggio per l'MPC
        self._ge_ground = None              # superficie per il modello di effetto suolo
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
        self.latest_fw_yaw = None # imbardata stimata a bordo [deg]
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
        # --- rilevamento del contatto ---
        self._near_t0 = None        # istante di ingresso sotto CONTACT_H_MAX
        self._z_hist = []           # (t, z) recenti, per lo stallo della quota
        self._acc_hist = []         # (t, acc.z) recenti, per il salto d'urto
        self._hov_hits = 0          # campioni consecutivi con hover apparente crollato
        self._vzup_hits = 0         # campioni consecutivi con vz invertita
        self._vz_zero_t0 = None     # inizio della finestra a vz ~ 0
        self._vz_was_down = False   # il drone stava scendendo
        self._touch_t0 = None       # istante del contatto
        self._touch_cause = ""      # quale firma ha scattato (per il log)
        self.seg_p_start = None
        self.last_p_LOS = None   # ultima posizione reale della carota (per continuita' tra segmenti)
        self._finished = False
        self._last_cmd_sat = False   # comando saturo nel tick precedente (osservatore d)
        self.last_cmd_sent = int(HOVER_CMD_INIT)  # ultimo comando inviato
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

    def _contact_detected(self, pos, vel, now):
        """True quando il drone ha toccato. Vedi il blocco CONTACT_* in testa.

        Le firme valgono solo in prossimita' del contatto: piu' in alto un picco
        di acc.z e' una raffica o una vibrazione, e vz ~ 0 e' un plateau, non un
        appoggio. Ritorna (contatto, causa).
        """
        zc = self._land_ground if self._land_ground is not None else self._z_ground
        if zc is None:
            return False, ""
        h = float(pos[2]) - float(zc)
        vz = float(vel[2])

        if h > CONTACT_H_MAX:
            self._near_t0 = None
            self._z_hist = []
            self._acc_hist = []
            self._hov_hits = 0
            self._vzup_hits = 0
            self._vz_zero_t0 = None
            self._vz_was_down = False
            return False, ""

        if self._near_t0 is None:
            self._near_t0 = now
        if vz < -0.03:
            self._vz_was_down = True

        # 1) stallo della quota (firma primaria, usa il mocap: poco rumoroso)
        self._z_hist.append((now, float(pos[2])))
        while self._z_hist and now - self._z_hist[0][0] > CONTACT_DZ_WIN:
            self._z_hist.pop(0)
        if (h < CONTACT_DZ_H and self._z_hist
                and now - self._z_hist[0][0] >= 0.9 * CONTACT_DZ_WIN):
            zs = [p[1] for p in self._z_hist]
            dz = max(zs) - min(zs)
            if dz < CONTACT_DZ_MAX:
                return True, (f"quota ferma ({dz*1000:.1f} mm in "
                              f"{CONTACT_DZ_WIN:.2f} s) a h={h*1000:.0f} mm")

        # 2) crollo dell'hover apparente: a terra e' il terreno a reggere.
        # La soglia e' sull'hover ATTESO A QUELLA QUOTA, non su HOVER_CMD secco:
        # in effetto suolo il controllo toglie spinta per compensare ge + d_hat,
        # e senza questa scalatura la regola spara in volo (vedi CONTACT_HOVER_GE).
        accz = self.latest_accz
        hov_ref = float(HOVER_CMD)
        if not CONTACT_HOVER_ON:
            self._hov_hits = 0
        if CONTACT_HOVER_GE:
            _ge = float(getattr(self.ctrl, "log_ge0", 0.0) or 0.0)
            _d = float(getattr(self.ctrl, "d_hat", 0.0) or 0.0)
            if not np.isfinite(_ge):
                _ge = 0.0
            if not np.isfinite(_d):
                _d = 0.0
            # solo la parte che SOLLEVA: una stima negativa non deve alzare la soglia
            hov_ref *= float(np.clip(1.0 - max(_ge + _d, 0.0) / G, 0.5, 1.0))
        if (CONTACT_HOVER_ON and accz is not None and h < CONTACT_HOVER_H
                and HOVER_EST_ACC_MIN < float(accz) < HOVER_EST_ACC_MAX
                and float(self.last_cmd_sent) / float(accz) < CONTACT_HOVER_F * hov_ref):
            self._hov_hits += 1
        else:
            self._hov_hits = 0
        if self._hov_hits >= CONTACT_HOVER_N:
            return True, (f"hover apparente {float(self.last_cmd_sent)/float(accz):.0f} "
                          f"contro {CONTACT_HOVER_F * hov_ref:.0f} atteso "
                          f"(hover {HOVER_CMD:.0f}) a h={h*1000:.0f} mm")

        # 3) transitorio d'urto: salto di acc.z rispetto alla mediana recente
        if accz is not None:
            self._acc_hist.append((now, float(accz)))
            while self._acc_hist and now - self._acc_hist[0][0] > CONTACT_ACC_WIN:
                self._acc_hist.pop(0)
            if len(self._acc_hist) >= 8:
                med = float(np.median([a for _, a in self._acc_hist[:-2]]))
                if float(accz) > med + CONTACT_ACC_JUMP:
                    return True, (f"urto: acc.z {float(accz):.2f} g contro "
                                  f"{med:.2f} di mediana, a h={h*1000:.0f} mm")

        # 4) inversione della velocita' verticale: piu' campioni consecutivi E una
        # traccia in acc.z. Senza la seconda condizione uno sbuffo del mocap da
        # qualche millimetro taglia i motori in volo (vedi CONTACT_VZ_UP_ACC).
        if CONTACT_VZUP_ON and self._vz_was_down and vz > CONTACT_VZ_UP:
            self._vzup_hits += 1
        else:
            self._vzup_hits = 0
        if self._vzup_hits >= CONTACT_VZ_UP_N and len(self._acc_hist) >= 8:
            _med = float(np.median([a for _, a in self._acc_hist[:-2]]))
            _rec = [a for tt, a in self._acc_hist if now - tt <= CONTACT_VZ_UP_WIN]
            if _rec and max(_rec) > _med + CONTACT_VZ_UP_ACC:
                return True, (f"inversione vz={vz:+.3f} m/s per {self._vzup_hits} "
                              f"campioni, acc.z {max(_rec):.2f} g contro {_med:.2f} "
                              f"di mediana, a h={h*1000:.0f} mm")

        # 5) azzeramento prolungato della velocita' (rete)
        if abs(vz) < CONTACT_VZ_ZERO and h < CONTACT_VZ_H:
            if self._vz_zero_t0 is None:
                self._vz_zero_t0 = now
            elif now - self._vz_zero_t0 >= CONTACT_T_ZERO:
                return True, f"vz ~ 0 per {CONTACT_T_ZERO:.2f} s a h={h*1000:.0f} mm"
        else:
            self._vz_zero_t0 = None

        # reti di sicurezza
        if h <= CONTACT_H_HARD:
            return True, f"quota sotto il contatto ({h*1000:.0f} mm)"
        if now - self._near_t0 > CONTACT_T_MAX:
            return True, f"timeout {CONTACT_T_MAX:.1f} s sotto {CONTACT_H_MAX*100:.0f} cm"
        return False, ""

    @staticmethod
    def _Rz(psi):
        """Matrice di rotazione attorno a z."""
        c, s_ = math.cos(psi), math.sin(psi)
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    @staticmethod
    def _wrap_pi(a):
        """Angolo in (-pi, pi]."""
        return float(a - 2.0 * math.pi * math.floor(a / (2.0 * math.pi) + 0.5))

    def _platform_yaw(self, now=None):
        """Yaw CORRETTO della piattaforma [rad], o None se non utilizzabile.

        Grezzo dal mocap (mediana circolare sulle ultime pose) meno l'offset di
        calibrazione. La si legge dove e', senza filtri sull'eta' o sul salto.
        """
        if not (PLATFORM_YAW_ENABLED and USE_PLATFORM) or self.plat_yaw_off is None:
            return None
        B = self._plat_buf[-PLATFORM_TRACK_N:]
        if len(B) < 3:
            return self.plat_yaw if np.isfinite(self.plat_yaw) else None
        s_ = np.median([math.sin(b[3]) for b in B])
        c_ = np.median([math.cos(b[3]) for b in B])
        self.plat_yaw_raw = math.atan2(s_, c_)
        self.plat_yaw = self._wrap_pi(self.plat_yaw_raw - self.plat_yaw_off)
        # traccia della rotazione fisica: se si gira la piattaforma a mano e
        # questa riga non compare, l'asset in Motive non segue l'orientamento
        if abs(self._wrap_pi(self.plat_yaw_raw - self._plat_yaw_logged)) > math.radians(PLATFORM_YAW_MOVE_LOG):
            self._plat_yaw_logged = self.plat_yaw_raw
            self.get_logger().info(
                f"[YAW] piattaforma ruotata: grezzo {math.degrees(self.plat_yaw_raw):+.1f} deg "
                f"-> target {math.degrees(self.plat_yaw):+.1f} deg")
        return self.plat_yaw

    def _yaw_frames(self):
        """(R_drone, R_target) rispetto al mondo, per il controllo di imbardata.

        R_drone dalla IMU di bordo. R_target dallo yaw corretto della
        piattaforma letto dal mocap; se la lettura non e' disponibile si ricade
        su YAW_TARGET. (None, None) finche' la IMU non ha ancora dato un valore:
        il controller allora non ruota e non comanda yaw.
        """
        if not YAW_ENABLED or self.latest_fw_yaw is None:
            return None, None
        y = float(self.latest_fw_yaw)
        if not np.isfinite(y):
            return None, None
        R_d = self._Rz(math.radians(YAW_FW_SIGN * y))
        # GATE: fuori da YAW_ACTIVE_STATES si restituisce R_target = None.
        # _yaw_rate_cmd vede R_target None e comanda rate zero, ma il controller
        # aggiorna comunque self.yaw_drone da R_drone e _world_to_body continua
        # a compensare l'imbardata corrente. NON restituire (None, None): quello
        # spegnerebbe anche la rotazione mondo->corpo e con un'imbardata residua
        # il controllo di posizione andrebbe a pezzi.
        if self.state not in YAW_ACTIVE_STATES:
            return R_d, None
        psi_t = self._platform_yaw()
        if psi_t is None:
            psi_t = math.radians(YAW_TARGET)
        return R_d, self._Rz(psi_t)

    def _on_accz(self, timestamp, data, logconf):
        """Callback del blocco acc.z (forza specifica lungo l'asse di spinta)."""
        self.latest_accz = data["acc.z"]
        self.latest_fw_yaw = data.get("stabilizer.yaw", None)

    def _on_pose(self, msg):
        """Callback mocap: rimanda la posizione al Kalman (send_extpos, SOLO
        posizione, mai il quaternione — vedi note del progetto) e la salva
        per il controllo.

        _pose_change_t serve SOLO alla colonna pose_age del CSV: non entra nel
        watchdog, che resta sull'istante di arrivo.
        """
        p = msg.pose.position
        pose = np.array([p.x, p.y, p.z])
        if self._pose_prev is None or not np.array_equal(pose, self._pose_prev):
            self._pose_prev = pose
            self._pose_change_t = time.perf_counter()
        try:
            self.cf.extpos.send_extpos(p.x, p.y, p.z)
        except Exception as e:
            self.get_logger().warn(f"send_extpos fallito: {e}")
        self.latest_pose = pose
        self.latest_pose_t = time.perf_counter()

    def _on_platform(self, msg):
        """Callback posa piattaforma: tiene le ultime PLATFORM_AVG_N pose."""
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # il timestamp va in coda: gli indici 0:3 (posizione) e 3 (yaw) usati
        # altrove restano quelli
        self._plat_buf.append((p.x, p.y, p.z, yaw, time.perf_counter()))
        # copia della posa GREZZA: la usa _track_platform con PLATFORM_RAW_POSE
        # e finisce comunque nel log, per poter misurare l'errore di
        # atterraggio contro il dato non elaborato
        if (self._plat_raw is None
                or abs(p.x - self._plat_raw[0]) > 1e-12
                or abs(p.y - self._plat_raw[1]) > 1e-12):
            self._plat_change_t = time.perf_counter()   # solo per il log
        self._plat_raw = (p.x, p.y, p.z)
        self._plat_t = time.perf_counter()
        if len(self._plat_buf) > PLATFORM_AVG_N:
            self._plat_buf.pop(0)

    def _platform_vel(self, now=None):
        """Velocita' xy della piattaforma [m/s], stimata ai minimi quadrati.

        Pendenza di x(t) e y(t) sulla finestra PLATFORM_VEL_WIN, passa-basso con
        PLATFORM_VEL_TAU, saturazione a PLATFORM_VEL_MAX. Ritorna un vettore 3
        con vz = 0 (vedi il blocco PLATFORM_VEL_* in testa).
        """
        if not (PLATFORM_VEL_ENABLED and PLATFORM_TRACK and USE_PLATFORM):
            return np.zeros(3)
        now = time.perf_counter() if now is None else now
        if self._plat_t is None or now - self._plat_t > PLATFORM_TRACK_MAXAGE:
            self._plat_vel[:] = 0.0            # posa vecchia: nessun feed-forward
            return np.zeros(3)
        B = [b for b in self._plat_buf if len(b) >= 5 and now - b[4] <= PLATFORM_VEL_WIN]
        if len(B) < 5:
            return self._plat_vel.copy()
        t = np.array([b[4] for b in B]) - B[0][4]
        span = float(t[-1] - t[0])
        if span < 0.5 * PLATFORM_VEL_WIN:       # finestra troppo corta: pendenza
            return self._plat_vel.copy()        # mal condizionata
        tc = t - t.mean()
        den = float(np.dot(tc, tc))
        if den <= 0.0:
            return self._plat_vel.copy()
        raw = np.array([float(np.dot(tc, np.array([b[i] for b in B]))) / den
                        for i in (0, 1)])
        lam = min(1.0, span / PLATFORM_VEL_TAU)
        self._plat_vel[0:2] += lam * (raw - self._plat_vel[0:2])
        self._plat_vel_t = now
        return np.clip(self._plat_vel, -PLATFORM_VEL_MAX, PLATFORM_VEL_MAX)

    def _platform_omega(self, now=None):
        """Velocita' di rotazione della piattaforma [rad/s], da passare all'MPC.

        Pendenza dello yaw srotolato sulla finestra TGT_OMEGA_WIN, stimata con
        Theil-Sen (mediana delle pendenze a coppie) invece che ai minimi
        quadrati: su ~17 campioni un solo outlier di 10 gradi al bordo della
        finestra sposterebbe la pendenza ai minimi quadrati di 0.17 rad/s su un
        valore vero di 0.10, cioe' la raddoppierebbe. La mediana a coppie regge
        fino a meta' campioni sporchi.

        Zero (nessuna rotazione) quando: la modalita' non e' "circle", la
        stima e' disabilitata, la posa e' vecchia, i campioni sono pochi o la
        piattaforma e' sostanzialmente ferma. Il valore e' poi passato al
        passa-basso TGT_OMEGA_TAU e saturato a +-TGT_OMEGA_MAX.
        """
        if not (TGT_OMEGA_ENABLED and PLAT_TRAJ_MODE == "circle"):
            self._tgt_omega = 0.0
            return 0.0
        if not (PLATFORM_YAW_ENABLED and PLATFORM_TRACK and USE_PLATFORM):
            self._tgt_omega = 0.0
            return 0.0
        now = time.perf_counter() if now is None else now
        if self._plat_t is None or now - self._plat_t > PLATFORM_TRACK_MAXAGE:
            self._tgt_omega = 0.0           # posa vecchia: nessuna rotazione
            return 0.0
        # la rotazione non e' osservabile se la piattaforma e' ferma
        if float(np.hypot(self._plat_vel[0], self._plat_vel[1])) < TGT_OMEGA_VMIN:
            self._tgt_omega = 0.0
            return 0.0

        B = [b for b in self._plat_buf if len(b) >= 5 and now - b[4] <= TGT_OMEGA_WIN]
        if len(B) < TGT_OMEGA_MIN_N:
            return float(self._tgt_omega)
        t = np.array([b[4] for b in B], dtype=float)
        y = np.unwrap(np.array([b[3] for b in B], dtype=float))
        span = float(t[-1] - t[0])
        if span < TGT_OMEGA_SPAN_MIN:
            # span assoluto, non legato a WIN: se il buffer e' corto si usa
            # quello che c'e' invece di bloccarsi. E se succede lo si dice.
            if not self._tgt_omega_warned:
                self._tgt_omega_warned = True
                print("[tgt_omega] finestra troppo corta: span %.3f s su %d pose "
                      "(serve %.2f s). Rotazione del riferimento NON attiva."
                      % (span, len(B), TGT_OMEGA_SPAN_MIN), flush=True)
            return float(self._tgt_omega)

        # guardia sullo yaw: via i campioni lontani dalla mediana della finestra
        # (riacquisizioni del corpo rigido). La posa ha PLATFORM_JUMP_MAX, lo
        # yaw non ha nessuna guardia a monte.
        res = y - np.median(y)
        keep = np.abs(res - np.median(res)) <= TGT_OMEGA_JUMP
        if int(keep.sum()) < TGT_OMEGA_MIN_N:
            return float(self._tgt_omega)
        t, y = t[keep], y[keep]

        # Theil-Sen: mediana delle pendenze a coppie
        dt_ = t[None, :] - t[:, None]
        dy_ = y[None, :] - y[:, None]
        iu = np.triu_indices(len(t), k=1)
        dt_, dy_ = dt_[iu], dy_[iu]
        good = np.abs(dt_) > 1e-3
        if not np.any(good):
            return float(self._tgt_omega)
        raw = float(np.median(dy_[good] / dt_[good]))
        if not np.isfinite(raw):
            return float(self._tgt_omega)

        lam = min(1.0, span / TGT_OMEGA_TAU)
        self._tgt_omega += lam * (raw - self._tgt_omega)
        self._tgt_omega = float(np.clip(self._tgt_omega,
                                        -TGT_OMEGA_MAX, TGT_OMEGA_MAX))
        return float(self._tgt_omega)

    def _track_platform(self, now):
        """Aggiorna il waypoint di landing sulla posa corrente della piattaforma.

        Sposta in blocco WP[LANDING], WP[HOLD], l'inizio del segmento di landing
        e TARGET_XY dello stesso delta, cosi' la geometria del segmento (e
        quindi la carota LOS) resta coerente. Aggiorna anche la quota di
        contatto e la superficie dell'effetto suolo, che dipendono dalla z
        della piattaforma. Vedi il blocco PLATFORM_TRACK_* in testa.
        """
        global TARGET_XY, Z_LAND
        if not (PLATFORM_TRACK and USE_PLATFORM) or self._emergency:
            return
        if self.plat_xyz is None or self.WP is None or self._rest is None:
            return
        if self.state not in ("rising", "nav_to_wp", "hold", "landing"):
            return
        if self._plat_t is None or now - self._plat_t > PLATFORM_TRACK_MAXAGE:
            return
        B = np.array(self._plat_buf[-PLATFORM_TRACK_N:], dtype=float)
        if len(B) < 3:
            return
        if PLATFORM_RAW_POSE:
            # posa grezza, nessun filtro: ritardo zero. L'unica difesa e' la
            # soglia sul salto (vedi il blocco PLATFORM_JUMP_* in testa).
            p = np.array(B[-1, 0:3], dtype=float)
            if np.linalg.norm(p) < PLATFORM_MIN_NORM:
                return                               # posa nulla: topic sporco
            prev = self.plat_xyz
            if prev is not None:
                salto = float(np.linalg.norm(p[0:2] - np.asarray(prev)[0:2]))
                if salto > PLATFORM_JUMP_MAX:
                    self._plat_skip += 1
                    if self._plat_skip <= PLATFORM_JUMP_N:
                        self._plat_skip_tot += 1
                        return                       # campione sballato: scartato
                    self.get_logger().warn(
                        f"[PLATFORM] {PLATFORM_JUMP_N} pose scartate di fila "
                        f"(ultimo salto {100*salto:.1f} cm): riaggancio sulla "
                        f"posa corrente")
            self._plat_skip = 0
        else:
            p = np.median(B[:, 0:3], axis=0)
            if np.linalg.norm(p) < PLATFORM_MIN_NORM:
                return                               # posa nulla: topic sporco

        # yaw GREZZO, non corretto: PLATFORM_XY_OFFSET e' un vettore misurato
        # nel frame del corpo rigido come lo legge Motive, quindi va ruotato
        # della lettura cruda. La correzione (plat_yaw_off) serve solo al
        # controllo di imbardata, che inquadra il mondo.
        # Lo yaw resta mediato anche con PLATFORM_RAW_POSE: serve solo a ruotare
        # PLATFORM_XY_OFFSET, che e' lungo 2 cm, quindi il suo ritardo non
        # sposta niente di misurabile (la piattaforma e' risultata ferma entro
        # 0.4 deg durante il landing) mentre il rumore si', se non filtrato.
        yaw = float(np.arctan2(np.median(np.sin(B[:, 3])), np.median(np.cos(B[:, 3]))))
        c, s_ = math.cos(yaw), math.sin(yaw)
        off_w = np.array([c * PLATFORM_XY_OFFSET[0] - s_ * PLATFORM_XY_OFFSET[1],
                          s_ * PLATFORM_XY_OFFSET[0] + c * PLATFORM_XY_OFFSET[1]])
        surface = p[2] + PLATFORM_TOP_DZ
        wp_new = np.array([p[0] + off_w[0], p[1] + off_w[1],
                           surface + self._rest + LAND_CLEARANCE - LAND_UNDERSHOOT])

        # il target E' dove si legge la piattaforma: nessun limite sul delta
        # QUI. Il controllo sui salti sta a monte, sulla posa (PLATFORM_JUMP_*),
        # dove si distingue un campione sballato da un movimento vero; su delta
        # non si potrebbe, perche' delta include anche il moto legittimo.
        delta = wp_new - self.WP[LANDING]
        if float(np.linalg.norm(delta)) < 1e-5:
            return
        self.WP[LANDING] += delta
        self.WP[HOLD][0:2] += delta[0:2]             # l'hold resta a Z_HOLD
        if self.dynamic_p_start is not None:
            self.dynamic_p_start[0:2] += delta[0:2]
        TARGET_XY = TARGET_XY + delta[0:2]
        self.plat_xyz = p.copy()

        # quote derivate, nello stesso ordine di _set_target_from_platform
        Z_LAND = float(self.WP[LANDING][2] + LAND_UNDERSHOOT)
        self._land_ground = Z_LAND - LAND_CLEARANCE
        self._ge_ground = self._land_ground - self._rest - GE_ROTOR_DZ

    def _set_target_from_platform(self):
        """Fissa TARGET_XY e Z_LAND dalla posa della piattaforma (prima del decollo)."""
        global TARGET_XY, Z_LAND
        B = np.array(self._plat_buf)
        P = B[:, 0:3]
        self.plat_xyz = np.median(P, axis=0)
        yaw = float(np.arctan2(np.median(np.sin(B[:, 3])), np.median(np.cos(B[:, 3]))))
        c, s_ = math.cos(yaw), math.sin(yaw)
        off_w = np.array([c * PLATFORM_XY_OFFSET[0] - s_ * PLATFORM_XY_OFFSET[1],
                          s_ * PLATFORM_XY_OFFSET[0] + c * PLATFORM_XY_OFFSET[1]])
        spread = np.ptp(P, axis=0) if len(P) > 1 else np.zeros(3)
        rest = float(self.latest_pose[2]) - FLOOR_Z      # drone appoggiato sul pavimento
        self._rest = rest
        surface = float(self.plat_xyz[2]) + PLATFORM_TOP_DZ
        TARGET_XY = np.array([self.plat_xyz[0], self.plat_xyz[1]], dtype=float) + off_w
        Z_LAND = surface + rest + LAND_CLEARANCE
        self._land_ground = surface + rest        # quota di contatto (FSM, osservatore)
        self._ge_ground = surface - GE_ROTOR_DZ   # superficie vera (effetto suolo)
        self.get_logger().info(
            f"[PLATFORM] {len(P)} pose, piattaforma ({self.plat_xyz[0]:.3f}, "
            f"{self.plat_xyz[1]:.3f}, {self.plat_xyz[2]:.3f}) (escursione "
            f"{1000*spread.max():.1f} mm), yaw {math.degrees(yaw):.1f} deg, offset mondo "
            f"({100*off_w[0]:+.1f}, {100*off_w[1]:+.1f}) cm -> TARGET_XY=({TARGET_XY[0]:.3f}, "
            f"{TARGET_XY[1]:.3f}), Z_LAND={Z_LAND:.3f}, suolo MPC={self._land_ground:.3f} "
            f"(drone appoggiato a {rest:.3f} m), superficie={surface:.3f}, "
            f"suolo effetto suolo={self._ge_ground:.3f}")
        if spread.max() > 0.01:
            self.get_logger().warn("[PLATFORM] la piattaforma si muove (>1 cm): target fissato comunque")
        self._calibrate_platform_yaw(B[:, 3], yaw)

    def _calibrate_platform_yaw(self, yaws, yaw_raw):
        """Fissa l'offset di imbardata della piattaforma.

        yaw_raw e' la mediana circolare delle pose di avvio. L'offset e' la
        lettura del mocap con la piattaforma fisicamente a zero: in
        autocalibrazione (PLATFORM_YAW_OFFSET = None) e' yaw_raw stesso, cioe'
        il target parte da zero comunque e si inseguono le rotazioni fatte
        DURANTE il volo; con un valore fisso il target e' assoluto.
        Nessun controllo di soglia: qui si misura e si stampa, basta.
        """
        if not (PLATFORM_YAW_ENABLED and YAW_ENABLED):
            self.get_logger().info(
                f"[YAW] lettura piattaforma disattivata: target fisso a {YAW_TARGET:+.1f} deg "
                f"(mocap grezzo {math.degrees(yaw_raw):+.1f} deg)")
            return
        auto = PLATFORM_YAW_OFFSET is None
        self.plat_yaw_off = yaw_raw if auto else math.radians(PLATFORM_YAW_OFFSET)
        self.plat_yaw = self._wrap_pi(yaw_raw - self.plat_yaw_off)
        self.plat_yaw_raw = yaw_raw
        self._plat_yaw_logged = yaw_raw
        self.get_logger().info(
            f"[YAW] piattaforma: mocap grezzo {math.degrees(yaw_raw):+.2f} deg, offset "
            f"{math.degrees(self.plat_yaw_off):+.2f} deg "
            f"({'autocalibrato' if auto else 'fisso'}) -> target "
            f"{math.degrees(self.plat_yaw):+.2f} deg")

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
        _praw = self._plat_raw if self._plat_raw is not None else (float("nan"),) * 3
        _err = (float("nan"), float("nan"))
        if (self._plat_raw is not None and self.latest_pose is not None
                and np.isfinite(self.plat_yaw_raw)):
            _cy, _sy = math.cos(self.plat_yaw_raw), math.sin(self.plat_yaw_raw)
            _aim = (_praw[0] + _cy * PLATFORM_XY_OFFSET[0] - _sy * PLATFORM_XY_OFFSET[1],
                    _praw[1] + _sy * PLATFORM_XY_OFFSET[0] + _cy * PLATFORM_XY_OFFSET[1])
            _err = (float(self.latest_pose[0]) - _aim[0],
                    float(self.latest_pose[1]) - _aim[1])
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
            ge_h=getattr(c, "log_ge_h0", float("nan")),
            d_hat=getattr(c, "log_d_hat", float("nan")),
            d_x=float(getattr(c, "log_d_xy", [float("nan")] * 2)[0]),
            d_y=float(getattr(c, "log_d_xy", [float("nan")] * 2)[1]),
            # z_ground: quota di CONTATTO effettivamente passata al controller
            # (prima qui finiva self._z_ground, che e' un'altra cosa: da qui
            # l'incoerenza tra la colonna e il valore usato in volo).
            # ge_ground: superficie usata dal modello di effetto suolo.
            z_ground=(self._land_ground if self._land_ground is not None
                      else (self._z_ground if self._z_ground is not None else float("nan"))),
            ge_ground=(self._ge_ground if self._ge_ground is not None else float("nan")),
            hv0=getattr(c, "log_hv0", float("nan")),
            eps_imp=getattr(c, "log_eps_imp", float("nan")),
            eps_imp0=_eps_imp_first(c),
            yaw_drone=math.degrees(getattr(c, "yaw_drone", float("nan"))),
            yaw_err=math.degrees(getattr(c, "log_yaw_err", float("nan"))),
            yaw_rate=math.degrees(getattr(c, "log_yaw_rate", float("nan"))),
            # yaw_plat: imbardata della piattaforma CORRETTA (grezza - offset),
            # cioe' quella che il controllo insegue
            yaw_plat=math.degrees(self.plat_yaw),
            yaw_plat_raw=math.degrees(self.plat_yaw_raw),
            # eta' della posa: tempo dall'ultima misura mocap DIVERSA dalla
            # precedente. In volo sano sta sotto i 20 ms; se sale, il mocap sta
            # ripetendo l'ultimo valore e si sta volando alla cieca.
            pose_age=(time.perf_counter() - self._pose_change_t
                      if self._pose_change_t is not None else float("nan")),
            # posa GREZZA della piattaforma, ultimo campione ricevuto, nessun
            # filtro; e numero totale di campioni scartati dalla soglia. Con
            # PLATFORM_RAW_POSE, target_x/target_y seguono questa colonna a meno
            # di PLATFORM_XY_OFFSET, quindi l'errore di atterraggio si legge
            # direttamente dal log senza correzioni a posteriori.
            plat_x_raw=_praw[0], plat_y_raw=_praw[1], plat_z_raw=_praw[2],
            plat_skip=getattr(self, "_plat_skip_tot", 0),
            # eta' della posa PIATTAFORMA: tempo dall'ultima misura DIVERSA
            # dalla precedente, stessa semantica di pose_age per il drone.
            plat_age=(time.perf_counter() - self._plat_change_t
                      if self._plat_change_t is not None else float("nan")),
            # ERRORE DI ATTERRAGGIO, calcolato qui e non ricostruito a
            # posteriori. e_x,e_y = posa drone - punto di mira, dove il punto di
            # mira e' la posa GREZZA della piattaforma a questo tick piu'
            # PLATFORM_XY_OFFSET ruotato dello yaw grezzo. Non passa da
            # target_x/target_y, che si congelano di un tick in touchdown perche'
            # _track_platform non gira piu' in quello stato, e che in generale
            # restano fermi su ogni uscita anticipata di _track_platform.
            # VALIDITA': una riga vale solo se pose_age e plat_age sono piccoli.
            # Nel volo 153859 la posa del drone e' rimasta congelata 0.548 s
            # prima del contatto e l'errore letto al touchdown (7.8 cm) era
            # interamente un artefatto: il drone fermo in un fotogramma vecchio
            # mentre il bersaglio continuava a scorrere.
            e_x=_err[0], e_y=_err[1],
            # rotazione del riferimento effettivamente passata al controllore
            tgt_omega=float(getattr(self, "_tgt_omega", 0.0)))

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

        # ---------------- APPOGGIO: spegnimento in anello aperto --------------
        # Contatto rilevato: il controllo non serve piu' (il terreno regge il
        # drone) e tenerlo attivo farebbe inseguire un riferimento sotto la
        # superficie con il drone fermo. Si scende a una frazione dell'hover e
        # poi a zero su TOUCHDOWN_T: un taglio secco farebbe ricadere il drone
        # da qualche millimetro e rimbalzare sulle gambe.
        if self.state == "touchdown":
            now = time.perf_counter()
            frac = (now - self._touch_t0) / TOUCHDOWN_T
            if frac >= 1.0:
                self.state = "idle"
                self._shutdown_flight()
                return
            cmd_td = int(np.clip(HOVER_CMD * TOUCHDOWN_CMD_F * (1.0 - frac), 0, 60000))
            self.cf.commander.send_setpoint(0.0, 0.0, 0.0, cmd_td)
            self.ctrl.integral_pos_e[:] = 0.0
            self._append_row(pos, vel, state="touchdown", mode=0,
                             p_LOS=pos, force=0.0, cmd=cmd_td, az=float("nan"),
                             roll=0.0, pitch=0.0, solve_ms=0.0,
                             v_ff=np.zeros(3), current_target_vel=np.zeros(3),
                             cmd_ctrl=float("nan"), spool_ceil=float("nan"),
                             wp_landing=self.WP[LANDING] if self.WP is not None else pos)
            return

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
                    if self._ge_ground is None:
                        # senza piattaforma si atterra sul pavimento: la
                        # superficie e' FLOOR_Z, non self._z_ground (che e' la
                        # quota dell'origine del corpo col drone appoggiato)
                        self._ge_ground = FLOOR_Z - GE_ROTOR_DZ
                    self.get_logger().info(
                        f"[SPOOL] pre-spin a {PRESPIN_CMD} per {PRESPIN_T:.2f}s "
                        f"(contatto z={self._z_ground:.3f} m, superficie "
                        f"{self._ge_ground:.3f} m)")
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
                [1.8, 0.0, Z_CRUISE],                # nav: punto di attesa
                [TARGET_XY[0], TARGET_XY[1], Z_HOLD],# hold: sopra il target (poi ricalcolato sul cono)
                [TARGET_XY[0], TARGET_XY[1], Z_LAND - LAND_UNDERSHOOT],# landing (rif. MPC)
            ])

            d_nav = float(np.linalg.norm(self.WP[NAV][0:2] - TARGET_XY))
            if d_nav < (Z_HOLD - Z_LAND) / ALPHA_CONE:
                self.get_logger().warn(
                    f"[PLATFORM] waypoint di nav a {d_nav:.2f} m dal target: il punto di "
                    f"hold ({(Z_HOLD - Z_LAND) / ALPHA_CONE:.2f} m dal target) cade dietro "
                    f"al nav: verra' tagliato alla posizione del drone")

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

        # il target segue la piattaforma (solo posizione, vedi _track_platform)
        self._track_platform(time.perf_counter())

        moving = self.state in ["hold", "landing"] and not self._emergency

        step_disp = TARGET_VEL * DT if moving else np.zeros(3)

        TARGET_XY += step_disp[0:2]
        self.WP[HOLD] += step_disp
        self.WP[LANDING] += step_disp

        if self.state in ["hold", "landing"] and not self._emergency:
            # TARGET_VEL e' la traiettoria SINTETICA (gia' applicata ai waypoint
            # con step_disp); _platform_vel e' la velocita' MISURATA della
            # piattaforma vera, il cui spostamento lo applica _track_platform.
            # Qui si sommano solo come riferimento di velocita' per l'MPC: non
            # vanno in step_disp, o il moto verrebbe contato due volte.
            current_target_vel = TARGET_VEL + self._platform_vel()
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

        R_d, R_t = self._yaw_frames()
        t0 = time.perf_counter()
        force, roll, pitch, yaw_rate, mode = self.ctrl.compute(
            pos, vel, p_LOS, target_yaw=0.0, target_vel=current_target_vel, ramp_ref_vel=v_ff,
            a_xy_lim=A_XY, final_pos=self.WP[LANDING], landing=landing,
            z_ground=(self._land_ground if self._land_ground is not None else self._z_ground),
            z_ground_ge=(self._ge_ground if self._ge_ground is not None
                         else (FLOOR_Z - GE_ROTOR_DZ)),
            cmd_saturated_prev=self._last_cmd_sat,
            R_drone=R_d, R_target=R_t,
            # rotazione del riferimento: attiva solo in hold/landing e solo in
            # modalita' "circle". Fuori da li' vale 0.0, che e' il ramo in cui
            # il riferimento e' identico bit per bit a quello di prima.
            tgt_omega=(self._platform_omega()
                       if (self.state in TGT_OMEGA_STATES and not self._emergency)
                       else 0.0))
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Nessun tetto qui: la rampa di spunto vive interamente nella fase
        # "spool" in anello aperto e si e' gia' conclusa col distacco.
        cmd_ctrl = force_to_cmd(force)
        cmd = cmd_ctrl
        self._last_cmd_sat = cmd >= 59999 or cmd <= 10001
        self.last_cmd_sent = int(cmd)   # usato dal rilevatore di contatto
        spool_ceil_log = float("nan")

        # terzo argomento = VELOCITA' di imbardata in deg/s, non un angolo
        self.cf.commander.send_setpoint(rad2deg(roll), rad2deg(pitch),
                                        rad2deg(yaw_rate), cmd)

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

        # --- terminazione del landing: CONTATTO, non piu' una soglia di quota ---
        # Prima si usciva a Z_LAND + 1 cm e si tagliavano i motori: con
        # LAND_CLEARANCE = 0.015 il drone veniva lasciato ~2.3 cm sopra
        # l'appoggio e quegli ultimi centimetri erano caduta libera (volo
        # 131434: vz controllata 0.094 m/s, impatto reale 0.68 m/s). Ora il
        # controllo arriva fino al contatto e la transizione la decide la
        # dinamica (picco di acc.z, inversione o azzeramento di vz).
        if self.state == "landing":
            touched, cause = self._contact_detected(pos, vel, time.perf_counter())
            if touched:
                self.state = "touchdown"
                self._touch_t0 = time.perf_counter()
                self._touch_cause = cause
                self.get_logger().info(
                    f"[CONTATTO] {cause} | vz={vel[2]:+.3f} m/s -> spegnimento in "
                    f"{TOUCHDOWN_T:.2f} s")
                return

        # Ultima rete, puramente verticale: se il mocap dice che siamo ben sotto
        # il riferimento di landing e il rilevatore di contatto non ha ancora
        # scattato, qualcosa non torna e si chiude comunque. La soglia sta sotto
        # il riferimento (gia' LAND_UNDERSHOOT sotto il contatto), quindi in un
        # volo normale questa riga non viene mai raggiunta: a decidere e' il
        # contatto. Non usa il gate xy+carrot delle transizioni di navigazione
        # perche' in coda dxy puo' rioscillare sopra stop_delta e lascerebbe il
        # drone appeso nello stato landing per secondi.
        # La soglia e' agganciata alla quota di CONTATTO, non al riferimento:
        # altrimenti cambiare LAND_UNDERSHOOT sposta anche la rete di sicurezza.
        # 4 cm sotto il contatto = dove stava con LAND_UNDERSHOOT = 0.02.
        _zc_net = (self._land_ground if self._land_ground is not None
                   else self._z_ground)
        if (self.state == "landing" and _zc_net is not None
                and pos[2] <= _zc_net - 0.04):
            self.get_logger().warn(
                f"[LANDING] chiusura di sicurezza a z={pos[2]:.3f} senza contatto rilevato")
            self.state = "idle"; self.old_wp_id, self.wp_counter = LANDING, IDLE
            return

        # --- B2: permanenza MASSIMA in hold (vedi PLAT_HOLD_MAX) -------------
        # Forza l'uscita dall'hold senza aspettare il gate di distanza+carota.
        # Senza questo la durata dell'hold la decide la geometria residua del
        # segmento e varia da 0.02 a 2.2 s fra un volo e l'altro, spostando il
        # punto di contatto di oltre 70 cm. Il landing parte da dove il drone si
        # trova: il vincolo di cono non dipende dal segmento, quindi resta
        # valutato correttamente comunque.
        if (self.state == "hold" and PLAT_HOLD_MAX > 0.0
                and PLAT_CHASE_ENABLED and USE_PLATFORM
                and self._hold_t0 is not None
                and time.perf_counter() - self._hold_t0 >= PLAT_HOLD_MAX
                and not (distance <= self.stop_delta and carrot_arrived)):
            v_rel = vel[0:2] - np.asarray(current_target_vel)[0:2]
            self.get_logger().info(
                f"[HOLD] uscita FORZATA a {PLAT_HOLD_MAX:.2f} s (distanza "
                f"{distance:.3f} m dal punto di hold), velocita' relativa "
                f"({v_rel[0]:+.3f}, {v_rel[1]:+.3f}) m/s")
            self.state = "landing"; self.old_wp_id, self.wp_counter, self.stop_delta = HOLD, LANDING, 0.05
            self.dynamic_p_start = (self.last_p_LOS.copy() if self.last_p_LOS is not None
                                     else self.WP[HOLD].copy())
            self.land_t0 = time.perf_counter()
            return

        if distance <= self.stop_delta and carrot_arrived:
            if self.state == "rising":
                self.state = "nav_to_wp"; self.old_wp_id, self.wp_counter, self.stop_delta = RISING, NAV, 0.3
            elif self.state == "hold":
                # --- B2: permanenza minima in hold (vedi PLAT_HOLD_MIN) -------
                if (PLAT_HOLD_MIN > 0.0 and PLAT_CHASE_ENABLED and USE_PLATFORM
                        and self._hold_t0 is not None
                        and time.perf_counter() - self._hold_t0 < PLAT_HOLD_MIN):
                    return
                if self._hold_t0 is not None:
                    v_rel = vel[0:2] - np.asarray(current_target_vel)[0:2]
                    self.get_logger().info(
                        f"[HOLD] uscita dopo {time.perf_counter() - self._hold_t0:.2f} s, "
                        f"velocita' relativa alla piattaforma "
                        f"({v_rel[0]:+.3f}, {v_rel[1]:+.3f}) m/s "
                        f"(|v_rel| = {float(np.linalg.norm(v_rel)):.3f})")
                self.state = "landing"; self.old_wp_id, self.wp_counter, self.stop_delta = HOLD, LANDING, 0.05
                self.dynamic_p_start = (self.last_p_LOS.copy() if self.last_p_LOS is not None
                                         else self.WP[HOLD].copy())
                self.land_t0 = time.perf_counter()
            elif self.state == "nav_to_wp":
                # --- B2: si resta fermi al nav finche' la piattaforma non passa ---
                if PLAT_CHASE_ENABLED and USE_PLATFORM:
                    tnow = time.perf_counter()
                    if self._nav_arrive_t is None:
                        self._nav_arrive_t = tnow
                        self.get_logger().info(
                            "[CHASE] fermo al nav, aspetto che la piattaforma prenda il distacco")
                    # distacco di rilascio: generatrice del cono, stessa formula
                    # del punto di hold statico (riga dist_xy_start piu' sotto)
                    gap_rel = ((Z_HOLD - Z_LAND) / ALPHA_CONE if PLAT_PASS_GAP is None
                               else float(PLAT_PASS_GAP))
                    # DISTACCO MISURATO LUNGO IL +x DEL FRAME DEL DRONE, non
                    # lungo il +x del mondo. Con il controllo di imbardata attivo
                    # in nav il drone si allinea alla piattaforma, quindi il suo
                    # asse longitudinale E' la direzione di marcia e il criterio
                    # vale qualunque verso abbia la corsa. Con il +x del mondo
                    # (volo 141841) la piattaforma andava in -x e il gap restava
                    # negativo: rilascio mai.
                    psi = float(getattr(self.ctrl, "yaw_drone", 0.0))
                    u_dir = np.array([math.cos(psi), math.sin(psi)])
                    gap = float(np.dot(TARGET_XY - pos[0:2], u_dir))
                    if self._plat_pass_t is None and gap >= PLAT_PASS_MARGIN:
                        self._plat_pass_t = tnow         # solo per diagnostica
                        self.get_logger().info(
                            f"[CHASE] sorpasso, gap={gap:+.3f} m (prua {math.degrees(psi):+.1f} deg); "
                            f"rilascio a {gap_rel:.3f} m")
                    scaduto = (tnow - self._nav_arrive_t) > PLAT_PASS_TIMEOUT
                    if gap < gap_rel and not scaduto:
                        return
                    if scaduto and gap < gap_rel:
                        self.get_logger().warn(
                            f"[CHASE] timeout di {PLAT_PASS_TIMEOUT:.0f} s con gap={gap:+.3f} m "
                            f"< {gap_rel:.3f} m: proseguo, VOLO DA SCARTARE")
                    self.get_logger().info(
                        f"[CHASE] rilascio: gap={gap:+.3f} m (bersaglio {gap_rel:.3f}, "
                        f"prua {math.degrees(psi):+.1f} deg), "
                        f"attesa al nav {tnow - self._nav_arrive_t:.2f} s")
                self.state = "hold"; self.old_wp_id, self.wp_counter, self.stop_delta = NAV, HOLD, 0.3
                self._hold_t0 = time.perf_counter()
                dist_xy_start = (Z_HOLD - Z_LAND) / ALPHA_CONE
                seg_nav = self.WP[HOLD][0:2] - self.WP[NAV][0:2]
                Lnav = np.linalg.norm(seg_nav)
                u_nav = seg_nav / Lnav if Lnav > 1e-6 else np.zeros(2)
                # TAGLIO DEL PUNTO DI HOLD ALLA POSIZIONE ATTUALE DEL DRONE.
                # L'hold sta sulla generatrice del cono, a (Z_HOLD-Z_LAND)/ALPHA_CONE
                # dal target: con ALPHA_CONE piccolo quella distanza cresce e il
                # punto finisce DIETRO al drone, che dovrebbe tornare indietro
                # uscendo dal volume mocap (perdita di posa, vedi ALPHA_CONE=0.5:
                # 3.14 m contro i 2.08 m a cui si trova il drone). Si taglia alla
                # posizione attuale, PROIETTATA sull'asse nav->target: usare la
                # distanza euclidea sposterebbe l'hold di lato quando il drone e'
                # fuori asse. max(0.0, ...) copre il caso in cui il drone abbia
                # gia' superato il target: l'hold finisce sul target e la discesa
                # e' verticale, invece di finire dall'altra parte.
                # Quando il taglio morde, il drone parte PIU' DENTRO al cono di
                # quanto previsto dalla generatrice: il margine al punto di hold
                # non vale piu' ALPHA_CONE*r_base ma (Z_HOLD-Z_LAND) - ALPHA_CONE*
                # (d_now - r_base). Va annotato nel registro del volo.
                d_now = float(np.dot(TARGET_XY - pos[0:2], u_nav))
                dist_cone = dist_xy_start
                dist_xy_start = min(dist_xy_start, max(0.0, d_now))
                if dist_xy_start < dist_cone - 1e-9:
                    self.get_logger().warn(
                        f"[HOLD] punto di hold tagliato: generatrice del cono a "
                        f"{dist_cone:.2f} m dal target, drone a {d_now:.2f} m -> "
                        f"hold a {dist_xy_start:.2f} m (si parte dentro al cono)")
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
        self.WP[LANDING] = np.array([pos[0], pos[1], z_em - LAND_UNDERSHOOT])
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
        stamp = save_and_plot(self.rows, ctrl=self.ctrl)
        try:
            # le righe "pose_lost" non hanno posizione: fuori dai grafici 3D
            plot_advanced_diagnostics(rows=[r for r in self.rows if r["state"] != "pose_lost"],
                                      stamp=stamp,
                                      # Il cono DISEGNATO deve essere quello IMPOSTO dalla
                                      # barriera, non la geometria del punto di hold: alpha,
                                      # z_cut e r_base si leggono dal controllore. Con i due
                                      # alpha disaccoppiati (hold nel main, barriera nel
                                      # controllore) passare ALPHA_CONE qui disegnava il cono
                                      # sbagliato proprio nei voli in cui la differenza conta.
                                      alpha_cone=getattr(self.ctrl, "alpha_cone", ALPHA_CONE),
                                      z_cut=getattr(self.ctrl, "z_cut", 1.5),
                                      r_base=getattr(self.ctrl, "r_base", 0.03))
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