"""
controller_deploy.py  —  il blocco "Controller" del tuo schema.

E' l'adattamento pybullet-free di MPCPIDHYControlDynamic. Riusa VERBATIM la
matematica (i due solve MPC, la CBF cono, il gate reachable-set, il PID di
reach) ma NON produce piu' RPM: si ferma a (forza di spinta, roll, pitch, yaw),
perche' assetto+rate+mixer li chiude il firmware del Crazyflie.

Differenze rispetto al controllore di sim, tutte necessarie per il deploy:
  - niente import pybullet, niente BaseControl;
  - lo stato in ingresso e' solo posizione + velocita' (quello che leggi dal
    SyncLogger): l'orientamento non serve piu', vedi nota su _pid_reach;
  - l'uscita e' forza [N] + assetto [rad], non RPM.

INDICE DELLE COSE AGGIUNTE PER LO YAW (per non cercarle):
  - yaw_* in __init__      yaw_enabled, yaw_kp, yaw_rate_max, yaw_deadband,
                           yaw_symmetry, yaw_rate_sign. Nessuna protezione
                           automatica: il controllo non si spegne da solo.
  - _yaw_error()           errore in angolo-asse da R_drone^T R_target.
  - _yaw_rate_cmd()        proporzionale + banda morta + saturazione.
  - _world_to_body()       ruota il comando orizzontale dal mondo al corpo.
  - log_yaw_err/rate       solo diagnostica.
"""

import numpy as np
import cvxpy as cp
import math
import time
from scipy.spatial.transform import Rotation


class HybridController:
    def __init__(self, g=9.81, dt=0.02, mass=0.0379,
                 polytope_path="reachable_polytope.npz"):
        self.g = g
        self.MPC_FREQ_DIVIDER = 2 # MPC running at 25 Hz
        self.dt = dt
        self.mpc_dt = self.MPC_FREQ_DIVIDER * self.dt
        self.M = mass
        self.GRAVITY = mass * g
        self.N = 40

        # ---- MPC prediction models (identici al sim) -----------------------
        # ORIZZONTALE: ritardo di accelerazione nel modello. Fra il comando di
        # assetto e l'accelerazione vera passano 140-160 ms, misurati per
        # cross-correlazione fra g*tan(pitch) e la derivata di vx sui voli
        # 143216, 143403 e 173605 (correlazione 0.95-0.97). Con il doppio
        # integratore l'MPC crede che l'accelerazione sia istantanea: pianifica
        # una frenata che arriva in ritardo e va lungo. Si aggiunge un terzo
        # livello di stato, l'accelerazione, con dinamica del primo ordine
        #
        #     a_{k+1} = (1 - lam) a_k + lam * g * u_k ,   lam = mpc_dt / att_tau
        #
        # Lo stato diventa [x, y, vx, vy, ax, ay]. Il costo resta sulle prime
        # QUATTRO componenti, quindi Q_hrz e R_hrz non cambiano significato e la
        # taratura attuale vale ancora. Con att_lag_enabled = False si torna
        # esattamente al modello di prima.
        self.att_lag_enabled = True
        self.att_tau = 0.15          # [s] costante del ritardo di accelerazione
        self._a_hrz = np.zeros(2)    # stato di accelerazione [ax, ay] del modello
        self._build_hrz_model()
        # VERTICALE: stessa struttura, ma costante di tempo SUA, non att_tau.
        # Fino al 9/10 il modello verticale era un doppio integratore puro, cioe'
        # l'MPC credeva che la spinta comandata arrivasse istantanea. Il terzo
        # stato va messo, ma con il ritardo giusto.
        #
        # ATTENZIONE, errore corretto il 9/10 sera. Il primo tentativo usava
        # att_tau = 0.15 anche qui, "per simmetria con l'orizzontale". Sbagliato:
        # i due assi non condividono la catena di attuazione. L'orizzontale
        # comanda un ANGOLO e deve aspettare che il drone ruoti (0.14-0.16 s,
        # misurati per cross-correlazione). Il verticale comanda la SPINTA
        # direttamente ai motori: nessuna rotazione di mezzo, solo il polo dei
        # motori. Identificazione di cmd/hover -> acc.z, modello del primo ordine
        # con ritardo puro, minimi quadrati su (D, lambda), 14 voli del 9/10 in
        # due finestre indipendenti:
        #   hold + discesa sopra 40 cm (n~300, escursione +-2%):
        #       ritardo puro 0.02-0.04 s, polo 0.013-0.025 s  ->  TOTALE 0.04-0.06
        #   salita (n~420, escursione 19-27% di hover):
        #       ritardo puro 0.00-0.02 s, polo 0.051-0.067 s  ->  TOTALE 0.05-0.07
        # Con il log a 50 Hz non si risolve meglio di ~0.04 s, ma 0.15 e' fuori
        # di un fattore tre. Il valore usato e' 0.06, estremo alto delle due
        # stime, per non sottostimare il ritardo.
        #
        # Cosa faceva il valore sbagliato: lam = 0.04/0.15 = 0.27 invece di 0.67,
        # cioe' il modello credeva che ogni comando rendesse il 27% della sua
        # accelerazione al primo passo invece del 67%. A parita' di R_vrt
        # un'azione che "rende meno" costa di piu' per unita' di accelerazione, e
        # l'ottimo si sposta su profili meno aggressivi: era una DETUNING
        # mascherata del verticale. Lo stallo spariva perche' lo stallo era
        # sovra-frenata, non perche' la fase fosse corretta. Con vrt_tau giusto
        # una parte di stallo puo' tornare: si toglie con R_vrt / Q_vrt, che sono
        # i parametri per farlo, non con una costante di tempo falsa.
        #
        #     a_{k+1} = (1 - lam) a_k + lam * u_k ,   lam = mpc_dt / vrt_tau
        #
        # Lo stato diventa [z, vz, az]. Differenza rispetto all'orizzontale: qui u
        # E' gia' un'accelerazione (m/s^2), non un angolo, quindi in B non c'e' g.
        # Effetto suolo e disturbo NON passano dal ritardo: sono accelerazioni
        # reali gia' in atto, quindi entrano direttamente nella riga della
        # velocita' tramite E_vrt (stessa scelta fatta per d_xy sull'orizzontale).
        # Il costo resta sulle prime DUE componenti, quindi Q_vrt non cambia
        # significato. Con vrt_lag_enabled = False si torna al modello di prima.
        self.vrt_lag_enabled = True
        self.vrt_tau = 0.085          # [s] ritardo comando di spinta -> acc.z.
                                     # NON e' att_tau: vedi il blocco sopra.
        self._a_vrt = 0.0            # stato di accelerazione az del modello
        self._build_vrt_model()

        # ---- DSL PID gains (reach) -----------------------------------------
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.0, .0, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])


        #self.P_COEFF_FOR = np.array([.45, .45, 1.25])
        #self.D_COEFF_FOR = np.array([.15, .15, .5])

        # ---- MPC weights ---------------------------------------------------

        # Baseline 
        self.Q_hrz = np.diag([3.0, 3.0, 4.0, 4.0])
        self.R_hrz = np.diag([7.0, 7.0])
        self.Q_vrt = np.diag([2.0, 1.5])
        self.R_vrt = np.diag([3.0])


        # VERTICAL

        # Conservative 
        #self.Q_hrz = np.diag([3.0, 3.0, 4.0, 4.0])
        #self.R_hrz = np.diag([7.0, 7.0])
        #self.Q_vrt = np.diag([1.0, 2.0])
        #self.R_vrt = np.diag([10.0])
        
        # Aggressive
        #self.Q_hrz = np.diag([3.0, 3.0, 4.0, 4.0])
        #self.R_hrz = np.diag([7.0, 7.0])
        #self.Q_vrt = np.diag([8.0, 0.75])
        #self.R_vrt = np.diag([1.5])

        # HORIZONTAL
        
        # Conservative 
        #self.Q_hrz = np.diag([2.0, 2.0, 5.0, 5.0])
        #self.R_hrz = np.diag([12.0, 12.0])
        #self.Q_vrt = np.diag([2.0, 1.5])
        #self.R_vrt = np.diag([3.0])
        
        # Aggressive
        #self.Q_hrz = np.diag([6.0, 6.0, 3.0, 3.0])
        #self.R_hrz = np.diag([5.0, 5.0])
        #self.Q_vrt = np.diag([2.0, 1.5])
        #self.R_vrt = np.diag([3.0])
        
        # ---- Cone CBF (glideslope) -----------------------------------------
        self.cbf_cone_enabled = True
        #self.alpha_cone = 1.72

        self.alpha_cone = 2.0
        self.v_max_hrz = 0.9          # limite velocita' orizzontale [m/s] (faccia del politopo imposta nell'MPC)
        # --- DHOCBF grado 2: due coefficienti di classe-K (uno per livello) ---
        # catena  h0 -> h1 -> h2 ,  vincolo su h2 >= 0 (rilassato con slack).
        # Inizializzati uguali cosi' si parte da un comportamento noto; tarabili
        # indipendentemente (e' il vantaggio della HOCBF).
        self.gamma_cbf = 0.5         # mantenuto per retro-compatibilita' / riferimento
        self.gamma1_cbf = 0.5         # livello 1
        self.gamma2_cbf = 0.5         # livello 2
        # ---- effetto suolo nel modello verticale ----------------------------
        # In prossimita' del suolo la stessa spinta comandata produce piu'
        # spinta reale: T_IGE/T_OGE = 1/(1-(R_eff/4h)^2)  (Cheeseman-Bennett).
        # R_eff = 40 mm (contro i 22.5 mm dell'elica singola) tiene conto dei
        # 4 rotori e del corpo: con questo valore il modello da' -1.8% di
        # comando necessario a h=7.5 cm e -0.6% a 12.5 cm, in linea con i
        # voli del 16-17/9 (-0.2 / -1.3% misurati tra 7 e 15 cm, nulli sopra
        # 0.3 m). Entra nel modello come accelerazione nota:
        #     z_{k+1} = A z_k + B (u_k + a_ge(h_k)),   a_ge = g*(k_IGE - 1)
        # valutata sulla traiettoria z predetta al solve precedente, quindi
        # il problema resta un QP (termine affine, parametro).
        # ATTENZIONE alla quota da usare. Cheeseman-Bennett vuole la distanza
        # PIANO ROTORI - SUPERFICIE. Fino al 21/9 il modello usava self.z_ground,
        # che il main riempie con "superficie + quota del drone appoggiato": e'
        # la quota che l'origine del corpo AVRA' a contatto avvenuto, quindi h=0
        # al contatto mentre i rotori sono ancora ~3.5 cm sopra la superficie.
        # Risultato: h sottostimata di una costante pari a quell'offset, e
        # siccome il clamp k_max scat4ta sotto 3.4 cm, il modello saturava a
        # 0.98 m/s^2 in OGNI atterraggio (voli 175138/175926/181036/181649),
        # con l'MPC che toglieva ~1 m/s^2 proprio nel flare. Da qui in poi il
        # modello usa un suolo suo, self.z_ground_ge, che il main riempie con
        # la superficie vera; self.z_ground resta la quota di contatto e serve
        # solo al congelamento dell'osservatore (dist_freeze_h).
        self.ge_enabled = True
        self.ge_r_eff = 0.075        # raggio efficace [m]. Rifit 1/10 sulla h
                                     # corretta, voli 125809 e 131434: RMS 1.6%
                                     # sui 5-36 cm (contro 2.0% con 65 mm e 1.8%
                                     # con 85). Il vecchio 41 mm veniva da un fit
                                     # con h sottostimata di 3.5 cm e spiegava
                                     # solo il 2.5% a 7 cm contro il 6% reale.
        self.ge_k_max = 1.25         # rete di sicurezza sul guadagno. Con 41 mm
                                     # bastava 1.10; con 75 mm quel valore
                                     # scatterebbe gia' a 6.2 cm, cioe' dentro la
                                     # zona utile. A 1.25 scatta sotto 4.2 cm.
        self.ge_h_min = 0.020        # [m] quota minima a cui valutare il modello:
                                     # Cheeseman ha un polo in R/4 = 10 mm ed e'
                                     # validato solo per h/R >~ 0.5. Sotto questa
                                     # soglia il valore viene congelato invece di
                                     # divergere (e la traiettoria predetta, che
                                     # con LAND_UNDERSHOOT punta sotto la
                                     # superficie, non genera piu' h negative).
        self.z_ground = 0.0          # quota di CONTATTO [m] (origine corpo appoggiata)
        self.z_ground_ge = 0.0       # quota della SUPERFICIE [m], per l'effetto suolo
        self.log_ge0 = float("nan")  # a_ge del primo passo [m/s^2], solo log
        self.log_ge_h0 = float("nan")# h usata al primo passo [m], solo log

        # ---- MPC offset-free: disturbo costante stimato sull'asse z ----------
        # Il modello verticale diventa
        #     z_{k+1} = A z_k + B (u_k + a_ge,k + d)
        # dove d [m/s^2] raccoglie tutto cio' che il modello non spiega (errore
        # residuo di HOVER_CMD, effetto suolo reale diverso dal modello, flusso
        # in discesa, batteria). d e' stimato con un osservatore a finestra:
        # su finestre di DIST_WIN s si confronta la variazione di vz misurata
        # con quella attesa dall'accelerazione comandata (+ effetto suolo), e
        # la differenza media viene filtrata passa-basso con costante dist_tau.
        # L'MPC usa d come termine noto costante sull'orizzonte: l'errore a
        # regime va a zero (Pannocchia & Rawlings 2003; Maeder et al. 2009).
        # Stima congelata vicino al suolo (la reazione del terreno non e' un
        # disturbo da compensare) e con comando saturo; |d| limitato.
        self.dist_enabled = True
        self.dist_tau = 0.3          # costante del filtro [s]
        self.dist_win = 0.10         # finestra di stima [s]
        self.dist_max = 1.5          # |d| massimo [m/s^2] (~15% di g)
        self.dist_freeze_h = 0.02    # sotto questa quota sul contatto la stima si
                                     # ferma [m]. Alzata da 0.005 il 13/10: sotto 2 cm
                                     # il modello di effetto suolo e' troppo ripido
                                     # (+120 m/s^2 per cm) e l'osservatore inseguiva
                                     # l'errore del modello, non il disturbo, con
                                     # retroazione positiva: piu' d_hat cresce, piu'
                                     # il drone frena, piu' resta nella zona ripida.
                                     # Correlazione tempo-in-banda / d_hat: r=+0.89;
                                     # anomalie 2/10 -> 0/10 e sigma della velocita'
                                     # al contatto da 0.050 a 0.013 m/s (13.6x).
        self.d_hat = 0.0
        # Stessa struttura sul piano orizzontale: disturbi dx, dy [m/s^2] che
        # raccolgono i bias costanti (baricentro non centrato, trim d'assetto,
        # offset dell'IMU) responsabili dell'errore fisso di 2-4 cm visto al
        # contatto (voli 175138, 175926). Costante piu' lunga dell'asse z per
        # non inseguire la resistenza aerodinamica, che dipende dalla velocita'.
        self.dist_xy_enabled = True
        self.dist_xy_tau = 2.5       # [s]
        self.dist_xy_max = 0.5       # |d_xy| massimo [m/s^2] (~3 deg di assetto)
        self.d_xy = np.zeros(2)
        self._obs_prev = None        # (a_cmd [ax, ay, az], a_ge, t) del tick precedente
        self._obs_v0 = None          # velocita' [vx, vy, vz] a inizio finestra
        self._obs_int = np.zeros(3)  # integrale di (a_cmd + a_ge) sulla finestra
        self._obs_T = 0.0            # durata della finestra
        self.log_d_hat = float("nan")
        self.log_d_xy = np.full(2, float("nan"))

        # ---- barriera di impatto (impact-aware) ------------------------------
        # Il cono dice DOVE si puo' stare, non QUANTO VELOCE ci si arriva: da
        # solo lascia toccare a 0.10-0.12 m/s. Si aggiunge una CBF sulla
        # velocita' relativa di avvicinamento:
        #
        #   h_v(k) = v_td + (vz_k - vz_piatt) + k_v (z_k - z_contatto) >= 0
        #
        # Alla quota di contatto il vincolo diventa vz_rel >= -v_td, cioe' E'
        # la specifica di touchdown; piu' in alto concede velocita' in modo
        # lineare con la quota residua. h_v dipende dalla velocita', quindi la
        # sua derivata contiene gia' l'ingresso: GRADO RELATIVO 1, basta un
        # livello (il cono ne richiede due perche' e' una barriera di posizione):
        #
        #   h_v(k+1) >= (1 - gamma_v) h_v(k) - eps_k
        #
        # affine nello stato e nell'ingresso -> il problema resta un QP.
        # Lontano dal suolo h_v e' largamente positiva e il vincolo e' inattivo
        # da solo: con k_v = 1.3 inizia a mordere negli ultimi ~30 cm.
        # Lo slack e' penalizzato molto piu' di quello del cono: la velocita' di
        # impatto e' la specifica fisica, il corridoio e' una preferenza.
        self.imp_enabled = True
        self.imp_vtd = 0.05          # velocita' di contatto ammessa [m/s]
        self.imp_kv = 1.3          # pendenza della barriera [1/s]: con vz di
                                     # crociera 0.5 m/s morde sotto ~32 cm
        self.imp_gamma = 0.2         # classe-K discreta (come il cono)
        self.rho_imp = 1e6           # peso dello slack (>> rho_slack del cono)
        self.log_hv0 = float("nan")  # valore della barriera al passo 0 [m/s]
        self.log_eps_imp = float("nan")

        # ---- controllo di imbardata ------------------------------------------
        # Finora lo yaw non era controllato (rate 0 a send_setpoint) e il
        # controllore assumeva implicitamente muso lungo world +x.
        #
        # ERRORE. Si prendono le due matrici di rotazione rispetto al mondo e si
        # guarda la rotazione relativa
        #     R_err = R_drone^T R_target
        # in rappresentazione ANGOLO-ASSE: la componente z del vettore rotazione
        # e' l'angolo da annullare. E' ben definita anche con il target
        # inclinato, dove sottrarre due yaw estratti separatamente darebbe
        # risultati diversi a seconda della convenzione di estrazione.
        #
        # USCITA: velocita' di imbardata, perche' il terzo argomento di
        # send_setpoint e' un rate e non un angolo.
        #
        # FRAME. Controllare lo yaw obbliga a ruotare il comando orizzontale:
        # gli angoli che escono sia dall'MPC (B_hrz: a_x = g u1, a_y = -g u0)
        # sia dal DSL-PID sono riferiti al MONDO, mentre il firmware interpreta
        # roll e pitch nel frame del DRONE. Finche' lo yaw restava a zero le due
        # cose coincidevano. La rotazione si fa in UN SOLO POSTO, in fondo a
        # compute(), su entrambi i rami.
        self.yaw_enabled = True
        self.yaw_kp = 2.0            # [1/s] costante di tempo 0.5 s
        self.yaw_rate_max = 1.05     # [rad/s] ~60 deg/s
        self.yaw_deadband = 0.035    # [rad] ~2 deg, anti-jitter
        self.yaw_symmetry = 2.0 * math.pi   # 2pi = allineamento vero;
                                     # pi/2 sfrutta la simmetria del telaio in X
        # Il commander legacy del firmware nega il campo yawrate del pacchetto
        # CRTP, quindi un valore positivo inviato a send_setpoint fa ruotare il
        # drone in senso ORARIO mentre l'errore e' in convenzione mondo
        # (antiorario positivo). Da verificare a terra.
        self.yaw_rate_sign = -1.0
        self.yaw_drone = 0.0         # imbardata corrente [rad], da R_drone
        self.log_yaw_err = float("nan")
        self.log_yaw_rate = float("nan")

        # ---- bersaglio che CURVA: rotazione del riferimento ------------------
        # Il riferimento orizzontale estrapola la piattaforma in linea retta su
        # tutto l'orizzonte: xref_k = wp + v*k*dt, con v costante. Su un arco
        # quella retta e' la TANGENTE, e si stacca dall'arco di
        #
        #     delta = R (1 - cos(v T / R))  ~  v^2 T^2 / (2 R)       [all'esterno]
        #
        # a fine orizzonte. Con R = 1.9 m, v = 0.30 m/s e T = 1.6 s sono 6 cm, e
        # con il costo terminale che pesa dieci volte quello di stadio quel punto
        # TIRA. Misurato sui tre voli in cerchio del 23/10: al contatto restavano
        # +3.6 e +3.9 cm di errore radiale verso l'ESTERNO, stesso segno della
        # previsione e circa il 60% del modulo, che e' quello che ci si aspetta
        # dato che il costo di stadio tira verso il riferimento a breve termine,
        # che invece e' giusto.
        #
        # L'errore scala come 1/R: stringendo il raggio peggiora. A R = 0.75 m e
        # 0.30 m/s diventa 15 cm, cioe' piu' del budget di errore laterale.
        #
        # tgt_omega e' la velocita' di rotazione del VETTORE VELOCITA' del
        # bersaglio [rad/s], che per un differenziale coincide con la sua
        # velocita' di imbardata. Con tgt_omega = 0 (default) il riferimento
        # resta la retta di sempre, bit per bit: chi non passa l'argomento non
        # vede nessuna differenza.
        self.tgt_omega = 0.0
        self.tgt_omega_max = 1.5     # [rad/s] tetto di sicurezza: una stima
                                     # sballata curverebbe il riferimento in un
                                     # cerchio stretto attorno al drone
        self.log_tgt_omega = 0.0
        self.a_cen_enabled = True           # feed-forward di centripeta ON/OFF.
                                            # Con False il controllore torna
                                            # identico a prima della patch
                                            # (rotazione del riferimento
                                            # compresa: cambia SOLO questo
                                            # termine), per l'A/B a parita' di
                                            # tutto il resto.
        self.a_cen_ff = 0.0                # [m/s^2] centripeta del riferimento
        self._tgt_traj = None        # cache per tick: la usano orizzontale E cono

        self.z_cut = 1.5
        self.r_base = 0.06
        self.rho_slack = 1e5        # peso penalita' slack del cono (grande = slack usato solo se necessario)

        # ---- multi-rate decimation -----------------------------------------
        self.control_counter = 0
        self.mpc_step = 0
        self.mpc_activated = False
        self.last_force = self.GRAVITY
        self.last_euler = np.zeros(3)

        # ---- diagnostica (solo log, non entra nel controllo) ---------------
        # log_az_mpc: primo ingresso del solve verticale [m/s^2], PRIMA del blend
        # log_eps0 / log_eps_max: slack del cono al primo passo e massimo
        # sull'orizzonte (0 = vincolo rispettato senza rilassamento).
        # NaN in modalita' PID; nei tick senza solve restano all'ultimo valore.
        self.log_az_mpc = float("nan")
        self.log_eps0 = float("nan")
        self.log_eps_max = float("nan")

        # ---- bumpless transfer PID -> MPC ------------------------------
        # Al gate le due leggi di controllo vengono scambiate istantaneamente:
        # PID e MPC calcolano forza/assetto in modo indipendente, senza continuita'
        # garantita, quindi il comando puo' avere un gradino netto al passaggio.
        # Si sfuma linearmente dall'uscita PID a quella MPC su blend_duration
        # secondi, poi MPC puro.
        #
        # ATTENZIONE, corretto il 14/10. Fino a quella data il blend partiva da
        # un'ANCORA: al fronte del gate si congelava last_pid_force e la si
        # pesava (1-alpha) per tutta la finestra. Sbagliato, perche' l'uscita del
        # PID NON e' un livello di trim ma una CORREZIONE funzione dello stato:
        #     target_thrust = P*e_pos + I*int + D*e_vel + [0,0,GRAVITY]
        # Se al gate il drone stava scendendo piu' in fretta del riferimento, il
        # termine D chiede spinta in piu' -- giusto per quel decimo di secondo.
        # Congelandola, quella richiesta resta applicata per 0.8 s A PRESCINDERE
        # dallo stato: diventa un comando in ANELLO APERTO su una grandezza nata
        # per essere in anello chiuso, e nessuno la ritira quando non serve piu'.
        #
        # Misurato sui voli del 13/10 sera (quattro voli, stessa configurazione):
        # ancora a +8.9% su hover -> vz da -0.305 a -0.041 m/s per mezzo secondo,
        # cioe' la discesa si FERMA mentre l'orizzontale continua; traiettoria
        # incurvata di 25.4 cm rispetto alla retta e durata 8.25 s. Gli altri tre
        # voli, ancora entro l'1% da hover, hanno dato 9.8-12.6 cm e 7.5-7.8 s.
        # Correlazione ancora-curvatura sui quattro voli: r = +0.996.
        # Il gate scatta sulla condizione del politopo e non sa nulla della fase
        # dell'oscillazione del PID: era una lotteria.
        #
        # Ora il PID viene RICALCOLATO a ogni tick della finestra e si fa il blend
        # sull'uscita viva. A alpha = 0 il valore coincide con quello che dava
        # l'ancora, quindi la continuita' al gate resta identica; da li' in poi la
        # correzione continua a dipendere dallo stato e si esaurisce da sola.
        #
        # Windup: durante la finestra il PID integra un errore che corregge solo
        # in parte. Qui e' innocuo e limitato per costruzione: integral_pos_e[2]
        # e' saturato a +-0.15 e I_COEFF_FOR[2] vale 0.05, quindi il contributo
        # massimo e' 0.0075 N su 0.372 di hover (2%), su una finestra di 0.8 s.
        # Non serve congelare l'integrale.
        self.blend_duration = 0.8
        self.blend_steps_total = max(1, int(round(self.blend_duration / self.dt)))
        self.blend_steps_left = 0
        # conservate solo come DIAGNOSTICA: non entrano piu' nel controllo, ma
        # restano la covariata con cui marcare i voli con transitorio anomalo
        # (scostamento da HOVER_FORCE oltre il 3%).
        self.blend_anchor_force = self.GRAVITY
        self.blend_anchor_euler = np.zeros(3)
        self.last_pid_force = self.GRAVITY
        self.last_pid_euler = np.zeros(3)
        self.integral_pos_e = np.zeros(3)

        # ---- reachable-set polytope ----------------------------------------
        _P = np.load(polytope_path)
        self.H_ax, self.h_ax = _P["H_ax"], _P["h_ax"]
        self.H_vz, self.h_vz = _P["H_vz"], _P["h_vz"]

        # ==================================================================== #
        #  CVXPY: Compilazione Parametrizzata (Eseguita 1 sola volta)          #
        # ==================================================================== #
        self._compile_mpc_problems()

    def _build_hrz_model(self):
        """Matrici del modello orizzontale, con o senza ritardo di accelerazione.

        Senza ritardo: stato [x, y, vx, vy], l'ingresso entra nella velocita'.
        Con ritardo:   stato [x, y, vx, vy, ax, ay], l'ingresso entra
        nell'accelerazione, che e' a sua volta uno stato con costante att_tau.
        In entrambi i casi il disturbo stimato d_xy entra nelle righe della
        velocita', quindi l'ingresso di equilibrio u_ss resta lo stesso.
        """
        dt, g = self.mpc_dt, self.g
        if self.att_lag_enabled:
            lam = float(np.clip(dt / max(self.att_tau, 1e-3), 0.0, 1.0))
            self.n_hrz = 6
            self.A_hrz = np.array([[1, 0, dt, 0,       0,       0],
                                   [0, 1, 0,  dt,      0,       0],
                                   [0, 0, 1,  0,      dt,       0],
                                   [0, 0, 0,  1,       0,      dt],
                                   [0, 0, 0,  0, 1 - lam,       0],
                                   [0, 0, 0,  0,       0, 1 - lam]], dtype=float)
            self.B_hrz = np.array([[0, 0], [0, 0], [0, 0], [0, 0],
                                   [0, lam * g], [-lam * g, 0]], dtype=float)
            self.E_hrz = np.array([[0, 0], [0, 0], [dt, 0], [0, dt],
                                   [0, 0], [0, 0]], dtype=float)
        else:
            self.n_hrz = 4
            self.A_hrz = np.array([[1, 0, dt, 0],
                                   [0, 1, 0, dt],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=float)
            self.B_hrz = np.array([[0, 0], [0, 0], [0, g * dt], [-g * dt, 0]],
                                  dtype=float)
            self.E_hrz = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]], dtype=float)

    def _build_vrt_model(self):
        """Matrici del modello verticale, con o senza ritardo di accelerazione.

        Senza ritardo: stato [z, vz], l'ingresso entra nella velocita'.
        Con ritardo:   stato [z, vz, az], l'ingresso entra nell'accelerazione,
        che e' a sua volta uno stato con costante vrt_tau (NON att_tau: la
        catena di attuazione verticale non passa dall'assetto).
        In entrambi i casi l'effetto suolo e il disturbo stimato entrano nella
        riga della velocita' (E_vrt): sono accelerazioni reali, non comandi, e
        non subiscono il ritardo d'attuazione.
        """
        dt = self.mpc_dt
        if self.vrt_lag_enabled:
            lam = float(np.clip(dt / max(self.vrt_tau, 1e-3), 0.0, 1.0))
            self.n_vrt = 3
            self.A_vrt = np.array([[1.0, dt, 0.0],
                                   [0.0, 1.0, dt],
                                   [0.0, 0.0, 1.0 - lam]], dtype=float)
            self.B_vrt = np.array([[0.0], [0.0], [lam]], dtype=float)
            self.E_vrt = np.array([0.0, dt, 0.0], dtype=float)
        else:
            self.n_vrt = 2
            self.A_vrt = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
            self.B_vrt = np.array([[0.0], [dt]], dtype=float)
            self.E_vrt = np.array([0.0, dt], dtype=float)

    def _compile_mpc_problems(self):
        # --- 1. HORIZONTAL PROBLEM ---
        self.x_hrz = cp.Variable((self.n_hrz, self.N + 1))
        self.u_hrz = cp.Variable((2, self.N))
        self.p_cur_hrz = cp.Parameter(self.n_hrz)
        self.p_xref_hrz = cp.Parameter((4, self.N + 1))
        self.p_axy_lim = cp.Parameter(nonneg=True)
        self.p_dxy = cp.Parameter(2)     # disturbo stimato [dx, dy] (offset-free)
        self.p_uss = cp.Parameter(2)     # ingresso di equilibrio [dy/g, -dx/g]
        E_hrz = self.E_hrz

        wQh=np.sqrt(np.diag(self.Q_hrz)); wRh=np.sqrt(np.diag(self.R_hrz)); wQhT=np.sqrt(10.0)*wQh
        cost_hrz = 0
        cons_hrz = [self.x_hrz[:, 0] == self.p_cur_hrz]
        for k in range(self.N):
            # il costo pesa solo [x, y, vx, vy]: con il modello a 6 stati
            # l'accelerazione e' uno stato interno, non un obiettivo
            cost_hrz += cp.sum_squares(cp.multiply(wQh, self.x_hrz[0:4, k] - self.p_xref_hrz[:, k]))
            # costo su u - u_ss: all'equilibrio serve l'assetto che compensa d
            cost_hrz += cp.sum_squares(cp.multiply(wRh, self.u_hrz[:, k] - self.p_uss))
            cons_hrz += [self.x_hrz[:, k + 1] == self.A_hrz @ self.x_hrz[:, k]
                         + self.B_hrz @ self.u_hrz[:, k] + E_hrz @ self.p_dxy]
            cons_hrz += [cp.abs(self.u_hrz[:, k]) <= self.p_axy_lim]
            #cons_hrz += [cp.abs(self.x_hrz[2, k]) <= self.v_max_hrz]   # |vx| <= v_max
            #cons_hrz += [cp.abs(self.x_hrz[3, k]) <= self.v_max_hrz]   # |vy| <= v_max
        cost_hrz += cp.sum_squares(cp.multiply(wQhT, self.x_hrz[0:4, self.N] - self.p_xref_hrz[:, self.N]))
        self.prob_hrz = cp.Problem(cp.Minimize(cost_hrz), cons_hrz)

        # --- 2. VERTICAL PROBLEM (Unificato con Cono CBF) ---
        self.x_vrt = cp.Variable((self.n_vrt, self.N + 1))
        self.u_vrt = cp.Variable((1, self.N))
        self.p_cur_vrt = cp.Parameter(self.n_vrt)
        self.p_xref_vrt = cp.Parameter(2)
        self.p_z_plat = cp.Parameter()
        self.p_r_cone = cp.Parameter(self.N + 1)
        self.p_ge = cp.Parameter(self.N)          # accelerazione da effetto suolo
        self.p_d = cp.Parameter()                 # disturbo stimato (offset-free)
        self.p_z_contact = cp.Parameter()         # quota di contatto (barriera impatto)
        self.p_vz_tgt = cp.Parameter()            # vz della piattaforma
        # disattivazione della barriera senza ricompilare: si rilassa il secondo
        # membro di una costante grande (stesso trucco di p_r_cone = -1000)
        self.p_imp_off = cp.Parameter(nonneg=True)
        self.eps_imp = cp.Variable(self.N, nonneg=True)
        # slack del cono: una variabile >=0 per ogni passo, penalizzata nel costo.
        # Rende il QP SEMPRE feasible: quando il cono e' impossibile da rispettare
        # (es. h->0 al vertice), il vincolo viene violato del minimo eps_k invece
        # di far collassare il solve. E' il rilassamento omega del paper DHOCBF.
        # un vincolo h2 per k = 0 .. N-2  ->  N-1 slack
        self.eps_cone = cp.Variable(self.N - 1, nonneg=True)

        wQv=np.sqrt(np.diag(self.Q_vrt)); wRv=np.sqrt(np.diag(self.R_vrt)); wQvT=np.sqrt(10.0)*wQv
        cost_vrt = 0
        cons_vrt = [self.x_vrt[:, 0] == self.p_cur_vrt]

        # --- barriera h0 su tutto l'orizzonte (helper, lineare in z) ---
        # h0_k = (z_k - z_plat) - r_k ,  r_k = p_r_cone[k] (parametro noto)
        h0 = [(self.x_vrt[0, k] - self.p_z_plat) - self.p_r_cone[k]
              for k in range(self.N + 1)]

        # --- livello 1:  h1_k = h0_{k+1} - (1-gamma1) h0_k ,  k = 0 .. N-1 ---
        h1 = [h0[k + 1] - (1.0 - self.gamma1_cbf) * h0[k] for k in range(self.N)]

        # --- barriera di impatto, grado relativo 1 (vedi __init__) ---
        hv = [self.imp_vtd + (self.x_vrt[1, k] - self.p_vz_tgt)
              + self.imp_kv * (self.x_vrt[0, k] - self.p_z_contact)
              for k in range(self.N + 1)]

        for k in range(self.N):
            # il costo pesa solo [z, vz]: con il modello a 3 stati l'accelerazione
            # e' uno stato interno, non un obiettivo (come sull'orizzontale)
            cost_vrt += cp.sum_squares(cp.multiply(wQv, self.x_vrt[0:2, k] - self.p_xref_vrt))
            # costo sull'accelerazione NETTA (u + effetto suolo + disturbo), non
            # su u: all'equilibrio sul target serve u = -(a_ge + d) != 0, e un
            # costo su u da solo farebbe preferire all'MPC un errore di quota
            # residuo pur di non spendere ingresso (plateau). Pesare la netta
            # equivale a pesare u - u_ss, come nella formulazione offset-free.
            cost_vrt += cp.sum_squares(cp.multiply(
                wRv, self.u_vrt[:, k] + self.p_ge[k] + self.p_d))
            # u passa dal ritardo (B_vrt), effetto suolo e disturbo no (E_vrt)
            cons_vrt += [self.x_vrt[:, k + 1] == self.A_vrt @ self.x_vrt[:, k]
                         + self.B_vrt @ self.u_vrt[:, k]
                         + cp.multiply(self.E_vrt, self.p_ge[k] + self.p_d)]
            cons_vrt += [cp.abs(self.u_vrt[:, k]) <= 9.0]

            # --- livello 2 (DHOCBF grado 2), rilassato con slack ---
            # h2_k = h1_{k+1} - (1-gamma2) h1_k >= -eps_k ,  valido per k = 0 .. N-2
            # (h2_k usa h0_{k+2} = z_{k+2}, l'ingresso az_k compare qui: grado rel. 2)
            if k <= self.N - 2:
                h2_k = h1[k + 1] - (1.0 - self.gamma2_cbf) * h1[k]
                cons_vrt += [h2_k >= -self.eps_cone[k]]

            # --- barriera di impatto: h_v(k+1) >= (1-gamma_v) h_v(k) ---
            cons_vrt += [hv[k + 1] >= (1.0 - self.imp_gamma) * hv[k]
                         - self.eps_imp[k] - self.p_imp_off]

        # penalita' quadratica sullo slack (feasibility, es. al vertice del cono)
        cost_vrt += self.rho_slack * cp.sum_squares(self.eps_cone)
        cost_vrt += self.rho_imp * cp.sum_squares(self.eps_imp)
        cost_vrt += cp.sum_squares(cp.multiply(wQvT, self.x_vrt[0:2, self.N] - self.p_xref_vrt))
        self.prob_vrt = cp.Problem(cp.Minimize(cost_vrt), cons_vrt)

        # --- AGGIUNGI QUESTO DUMMY SOLVE (WARM-UP) ---
        print("Warm-up MPC solver...")
        self.p_cur_hrz.value = np.zeros(self.n_hrz)
        self.p_xref_hrz.value = np.zeros((4, self.N + 1))
        self.p_axy_lim.value = 0.17
        self.p_dxy.value = np.zeros(2)
        self.p_uss.value = np.zeros(2)
        self.prob_hrz.solve(solver=cp.OSQP)
        
        self.p_cur_vrt.value = np.zeros(self.n_vrt)
        self.p_xref_vrt.value = np.zeros(2)
        self.p_z_plat.value = 0.0
        self.p_r_cone.value = np.full(self.N + 1, -100.0)
        self.p_ge.value = np.zeros(self.N)
        self.p_d.value = 0.0
        self.p_z_contact.value = -100.0
        self.p_vz_tgt.value = 0.0
        self.p_imp_off.value = 0.0
        self.prob_vrt.solve(solver=cp.OSQP)
        print("Warm-up completato.")

    # ==================================================================== #
    #  Entry point: stato (pos, vel) -> (forza, roll, pitch, yaw, modo)     #
    # ==================================================================== #
    def compute(self, cur_pos, cur_vel, target_pos, target_yaw=0.0,
                target_vel=None, a_xy_lim=0.17, final_pos=None, landing=False,
                ramp_ref_vel=None, z_ground=None, z_ground_ge=None,
                cmd_saturated_prev=False, R_drone=None, R_target=None,
                tgt_omega=None):
        # R_drone / R_target: matrici di rotazione rispetto al MONDO (3x3).
        # Se mancano, o se lo yaw e' disabilitato, non si ruota niente e il
        # comportamento e' identico a prima del controllo di imbardata.
        # velocita' di rotazione del bersaglio: se il main non la passa resta
        # zero e il riferimento e' la retta di sempre (vedi __init__)
        if tgt_omega is not None and np.isfinite(tgt_omega):
            self.tgt_omega = float(np.clip(float(tgt_omega),
                                           -self.tgt_omega_max, self.tgt_omega_max))
        self.log_tgt_omega = float(self.tgt_omega)
        self.control_counter += 1
        now = time.perf_counter()
        if R_drone is not None and self.yaw_enabled:
            Rd = np.asarray(R_drone, float)
            self.yaw_drone = math.atan2(Rd[1, 0], Rd[0, 0])
        elif not self.yaw_enabled:
            self.yaw_drone = 0.0
        yaw_rate = self._yaw_rate_cmd(R_drone, R_target, now)
        if z_ground is not None:
            self.z_ground = float(z_ground)
        # se il main non passa la superficie si ricade sulla quota di contatto
        # (comportamento vecchio, h sottostimata): serve solo per compatibilita'
        if z_ground_ge is not None:
            self.z_ground_ge = float(z_ground_ge)
        elif z_ground is not None:
            self.z_ground_ge = float(z_ground)
        self._observer_update(float(cur_pos[2]), cur_vel, now, cmd_saturated_prev)
        cur_pos = np.asarray(cur_pos, float)
        cur_vel = np.asarray(cur_vel, float)
        target_pos = np.asarray(target_pos, float)
        target_vel = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)
        ramp_ref_vel = np.zeros(3) if ramp_ref_vel is None else np.asarray(ramp_ref_vel, float)
        wp_final = np.asarray(final_pos, float) if final_pos is not None else target_pos

        in_set = self.is_in_reachable_set(cur_pos, cur_vel, wp_final, target_vel)

        gate_now = in_set and landing
        entering_mpc = gate_now and not self.mpc_activated

        if gate_now or self.mpc_activated:
            # ---------------- MPC MODE ----------------
            if entering_mpc:
                # fronte di salita del gate: apri la finestra di blend.
                # L'ancora viene registrata SOLO per diagnostica (vedi __init__):
                # il blend usa il PID vivo, non questo valore congelato.
                self.blend_anchor_force = self.last_pid_force
                self.blend_anchor_euler = self.last_pid_euler.copy()
                self.blend_steps_left = self.blend_steps_total

            self.mpc_activated = True
            run_mpc = (self.control_counter % self.MPC_FREQ_DIVIDER == 0
                       or self.mpc_step == 0)
            if run_mpc:
                self.mpc_step = 1
                force, euler = self._mpc_force_attitude(
                    cur_pos, cur_vel, wp_final, target_yaw, a_xy_lim, target_vel)
                self.last_force, self.last_euler = force, euler
            self.mpc_step += 1
            force, euler = self.last_force, self.last_euler

            if self.blend_steps_left > 0:
                # alpha: 0 appena entrati (tutto PID) -> 1 a fine finestra (tutto MPC)
                alpha = 1.0 - self.blend_steps_left / self.blend_steps_total
                # PID VIVO: stessi argomenti del ramo PID puro, cosi' la legge di
                # controllo e il riferimento sono gli stessi che aveva un tick
                # prima del gate. target_yaw resta 0 per la stessa ragione
                # spiegata nel ramo PID: la rotazione la fa _world_to_body.
                # Nota: qui il PID gira a 50 Hz mentre l'MPC resta a 25 Hz, ed e'
                # corretto -- il PID e' algebra, il suo costo e' trascurabile.
                pid_force, pid_euler = self._pid_reach(
                    cur_pos, cur_vel, target_pos, 0.0, target_vel + ramp_ref_vel)
                pid_euler = np.asarray(pid_euler, float)
                self.last_pid_force = pid_force
                self.last_pid_euler = pid_euler.copy()
                force = (1.0 - alpha) * pid_force + alpha * force
                euler = (1.0 - alpha) * pid_euler + alpha * np.asarray(euler, float)
                self.blend_steps_left -= 1

            mode = 1
        else:
            # ---------------- PID REACH MODE ----------------
            self.log_az_mpc = float("nan")
            self.log_eps0 = float("nan")
            self.log_eps_max = float("nan")
            self.log_ge0 = float("nan")
            self.log_ge_h0 = float("nan")
            self.log_hv0 = float("nan")
            self.log_eps_imp = float("nan")
            self.mpc_step = 0
            # target_yaw resta 0: i roll/pitch del DSL-PID sono riferiti al
            # MONDO e passargli lo yaw reale NON li ruota (as_euler('XYZ')
            # restituisce lo stesso pitch per qualunque yaw). La rotazione la
            # fa _world_to_body in fondo, per entrambi i rami.
            force, euler = self._pid_reach(
                cur_pos, cur_vel, target_pos, 0.0, target_vel + ramp_ref_vel)
            self.last_pid_force, self.last_pid_euler = force, np.asarray(euler, float).copy()
            mode = 0

        # MONDO -> CORPO, una volta sola per entrambi i rami. a_cmd esce di qui
        # gia' nel frame del mondo, che e' quello in cui lavora l'osservatore.
        phi_b, theta_b, a_cmd = self._world_to_body(
            float(euler[0]), float(euler[1]), force)
        euler = np.array([phi_b, theta_b, yaw_rate], dtype=float)
        ge_now = float(self._ge_accel(float(cur_pos[2]) - self.z_ground_ge)) if self.ge_enabled else 0.0
        self._obs_prev = (a_cmd, ge_now, now)
        # stato di accelerazione del modello con ritardo: e' il filtro del
        # primo ordine del comando, cioe' esattamente la dinamica che l'MPC
        # mette in A_hrz. Si aggiorna a ogni tick (dt), non solo quando gira
        # l'MPC, cosi' al latch del gate parte da un valore sensato.
        # Due costanti diverse: l'orizzontale aspetta la rotazione del drone,
        # il verticale no. Ogni filtro usa la stessa tau del modello che
        # alimenta, altrimenti lo stato iniziale passato al QP non e' lo stato
        # del modello del QP.
        if self.att_lag_enabled:
            lam_h = float(np.clip(self.dt / max(self.att_tau, 1e-3), 0.0, 1.0))
            self._a_hrz += lam_h * (np.asarray(a_cmd, float)[0:2] - self._a_hrz)
        if self.vrt_lag_enabled:
            # a_cmd[2] e' l'accelerazione verticale COMANDATA nel frame mondo
            # (gia' al netto dell'inclinazione): e' l'ingresso u che il modello
            # verticale filtra. Effetto suolo e disturbo non entrano qui, sono
            # accelerazioni reali e il QP li tratta a parte con E_vrt.
            lam_v = float(np.clip(self.dt / max(self.vrt_tau, 1e-3), 0.0, 1.0))
            self._a_vrt += lam_v * (float(np.asarray(a_cmd, float)[2]) - self._a_vrt)
        self.log_d_hat = self.d_hat
        self.log_d_xy = self.d_xy.copy()
        # il terzo valore e' la VELOCITA' di imbardata [rad/s], non un angolo
        return force, float(euler[0]), float(euler[1]), float(euler[2]), mode

    def _yaw_error(self, R_drone, R_target):
        """Errore di imbardata [rad] dalla rotazione relativa, in angolo-asse.

        R_err = R_drone^T R_target porta il frame del drone su quello del
        target; il logaritmo della matrice ne da' il vettore rotazione (asse per
        angolo) e la sua componente z e' l'imbardata da annullare.
        """
        R_err = np.asarray(R_drone, float).T @ np.asarray(R_target, float)
        e = float(Rotation.from_matrix(R_err).as_rotvec()[2])
        sym = float(self.yaw_symmetry)
        if sym > 0:                       # wrap in (-sym/2, +sym/2]
            e -= sym * math.floor(e / sym + 0.5)
        return e

    def _yaw_rate_cmd(self, R_drone, R_target, now=None):
        """Velocita' di imbardata comandata [rad/s]."""
        if not self.yaw_enabled or R_drone is None or R_target is None:
            self.log_yaw_err = float("nan")
            self.log_yaw_rate = 0.0
            return 0.0
        e = self._yaw_error(R_drone, R_target)
        self.log_yaw_err = e
        if abs(e) < self.yaw_deadband:
            self.log_yaw_rate = 0.0
            return 0.0
        r = float(np.clip(self.yaw_kp * e, -self.yaw_rate_max, self.yaw_rate_max))
        r *= self.yaw_rate_sign
        self.log_yaw_rate = r
        return r

    def _world_to_body(self, phi_w, theta_w, force):
        """(roll, pitch) dal frame MONDO a quello del DRONE.

        Esatta, non a piccoli angoli: si passa per l'accelerazione, si ruota di
        -psi e si torna agli angoli.

        ATTENZIONE AL RAMO. La coppia (roll, pitch) non e' unica: la stessa
        rotazione si scrive in due modi (doppia copertura), e il DSL-PID
        restituisce regolarmente quello "flippato", con roll vicino a +-180 e
        pitch di segno opposto. Succede ogni volta che target_thrust[2] diventa
        negativo, cioe' con il drone sopra il waypoint che scende: su stati
        realistici capita nel ~79% dei campioni. Ricostruire gli angoli con
        asin, che vive in [-90, +90], collassa quel ramo sull'altro e cambia il
        comando in modo drastico (roll 179 -> 1 deg: spinta in basso che diventa
        spinta in alto). Qui il ramo d'ingresso viene riconosciuto dal segno di
        cos(phi_w) e riprodotto in uscita, e il pitch si ricostruisce con atan2
        invece che con asin, cosi' resta corretto anche oltre i 90 gradi.
        """
        psi = self.yaw_drone
        fm = max(force / self.M, 1e-6)
        cphi_w = math.cos(phi_w)
        ax = fm * cphi_w * math.sin(theta_w)
        ay = -fm * math.sin(phi_w)
        azg = fm * cphi_w * math.cos(theta_w)          # = az + g
        a_world = np.array([ax, ay, azg - self.g])
        if abs(psi) < 1e-12:
            return phi_w, theta_w, a_world
        c, s_ = math.cos(psi), math.sin(psi)
        ax_b = c * ax + s_ * ay
        ay_b = -s_ * ax + c * ay
        sphi = float(np.clip(-ay_b / fm, -1.0, 1.0))
        if cphi_w >= 0.0:                               # ramo "normale"
            phi_b = math.asin(sphi)
            theta_b = math.atan2(ax_b, azg)
        else:                                           # ramo flippato
            phi_b = math.pi - math.asin(sphi)
            if phi_b > math.pi:
                phi_b -= 2.0 * math.pi
            theta_b = math.atan2(-ax_b, -azg)
        return phi_b, theta_b, a_world

    def _observer_reset(self):
        self._obs_v0 = None
        self._obs_int = np.zeros(3)
        self._obs_T = 0.0

    def _observer_update(self, z, vel, now, saturated):
        """Stima a finestra dei disturbi [dx, dy, dz] (vedi __init__)."""
        vel = np.asarray(vel, float)
        if not self.dist_enabled:
            self.d_hat = 0.0
        if not self.dist_xy_enabled:
            self.d_xy[:] = 0.0
        prev = self._obs_prev
        if prev is None or saturated:
            self._observer_reset()
            return
        a_prev, ge_prev, t_prev = prev
        dtr = now - t_prev
        if dtr <= 0.0 or dtr > 0.2:          # buco nel loop: finestra da rifare
            self._observer_reset()
            return
        if self._obs_v0 is None:             # inizio finestra
            self._obs_v0 = vel.copy()
            self._obs_int = np.zeros(3)
            self._obs_T = 0.0
            return
        self._obs_int += (a_prev + np.array([0.0, 0.0, ge_prev])) * dtr
        self._obs_T += dtr
        if self._obs_T < self.dist_win:
            return
        d_inst = (vel - self._obs_v0 - self._obs_int) / self._obs_T
        # vicino al suolo il contatto (reazione, attrito) non e' un disturbo
        if z - self.z_ground >= self.dist_freeze_h:
            if self.dist_enabled:
                alpha = min(1.0, self._obs_T / self.dist_tau)
                self.d_hat += alpha * (d_inst[2] - self.d_hat)
                self.d_hat = float(np.clip(self.d_hat, -self.dist_max, self.dist_max))
            if self.dist_xy_enabled:
                alpha = min(1.0, self._obs_T / self.dist_xy_tau)
                self.d_xy += alpha * (d_inst[0:2] - self.d_xy)
                self.d_xy = np.clip(self.d_xy, -self.dist_xy_max, self.dist_xy_max)
        self._obs_v0 = vel.copy()
        self._obs_int = np.zeros(3)
        self._obs_T = 0.0

    def _mpc_force_attitude(self, cur_pos, cur_vel, wp, target_yaw, a_xy_lim, target_vel):
        cx, cy, cz = cur_pos
        vx, vy, vz = cur_vel
        phi_cmd, theta_cmd, x_hrz = self._mpc_horizontal(
            cx, cy, vx, vy, wp, a_xy_lim, target_vel)

        if x_hrz is not None:
            az = self._mpc_vertical(cz, vz, wp, target_vel,
                                    x_hrz[0, :], x_hrz[1, :], apply_cone=True)
        else:
            az = self._mpc_vertical(cz, vz, wp, target_vel)

        # tilt-coupling correction
        theta_corr = math.atan((self.g / (self.g + az)) * math.tan(theta_cmd))
        phi_corr = math.atan((math.cos(theta_corr) / math.cos(theta_cmd))
                             * (self.g / (self.g + az)) * math.tan(phi_cmd))

        force = self.M * (self.g + az)
        
        # NB segni: se il drone si inclina al contrario, invertire qui (scipy vs cflib). Ora NON invertiti.
        return force, np.array([phi_corr, theta_corr, target_yaw])

    def _pid_reach(self, cur_pos, cur_vel, target_pos, target_yaw, target_vel):
        pos_ctrl_max = 0.4
        pos_e = target_pos - cur_pos
        norm = np.linalg.norm(pos_e)
        normalized = pos_e / norm if norm > 1e-3 else np.zeros(3)
        actual_pos_e = min(pos_ctrl_max, norm) * normalized

        vel_e = target_vel - cur_vel
        self.integral_pos_e = np.clip(self.integral_pos_e + pos_e * self.dt, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        target_thrust = (self.P_COEFF_FOR * actual_pos_e
                         + self.I_COEFF_FOR * self.integral_pos_e
                         + self.D_COEFF_FOR * vel_e
                         + np.array([0, 0, self.GRAVITY]))

        force = float(np.linalg.norm(target_thrust))

        z_ax = target_thrust / np.linalg.norm(target_thrust)
        x_c = np.array([math.cos(target_yaw), math.sin(target_yaw), 0])
        y_ax = np.cross(z_ax, x_c)
        y_ax = y_ax / np.linalg.norm(y_ax)
        x_ax = np.cross(y_ax, z_ax)
        R = np.vstack([x_ax, y_ax, z_ax]).T
        euler = Rotation.from_matrix(R).as_euler('XYZ', degrees=False)
        return force, euler

    def is_in_reachable_set(self, cur_pos, cur_vel, target_pos, target_vel=None):
        cur_pos = np.asarray(cur_pos, float)
        cur_vel = np.asarray(cur_vel, float)
        target_pos = np.asarray(target_pos, float)
        v_tgt = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)

        e_pos = cur_pos - target_pos
        e_vel = cur_vel - v_tgt

        def inside(H, h, x):
            return bool(np.all(H @ x <= h + 1e-9))

        in_x = inside(self.H_ax, self.h_ax, np.array([e_pos[0], e_vel[0]]))
        in_y = inside(self.H_ax, self.h_ax, np.array([e_pos[1], e_vel[1]]))
        in_z = inside(self.H_vz, self.h_vz, np.array([e_pos[2], e_vel[2]]))
        return in_x and in_y and in_z

    def _target_traj(self, wp, target_vel):
        """Posizione e velocita' del bersaglio sui passi 0..N dell'orizzonte.

        Ritorna (px, py, vx, vy), quattro vettori di lunghezza N+1.

        Con tgt_omega = 0 e' l'estrapolazione rettilinea di sempre:

            p_k = p_0 + v_0 k dt ,   v_k = v_0

        Con tgt_omega != 0 il vettore velocita' RUOTA alla stessa velocita'
        angolare del bersaglio, quindi il riferimento segue l'ARCO invece della
        tangente:

            v_k = R(phi_k) v_0
            p_k = p_0 + S(phi_k) v_0 ,     phi_k = omega k dt

                        1  [ sin phi    -(1 - cos phi) ]
            S(phi) = ----- [                            ]
                     omega [ 1 - cos phi     sin phi    ]

        S(phi) -> k dt * I per omega -> 0, quindi la formula DEGENERA DA SOLA
        nel caso rettilineo: quando la piattaforma smette di curvare non c'e'
        nessuna discontinuita' da gestire, e il ramo esplicito qui sotto serve
        solo a evitare la divisione per omega.

        Resta tutto affine nello stato e nell'ingresso: il riferimento e' un
        PARAMETRO del QP, quindi il problema non cambia natura e i tempi di
        solve nemmeno.
        """
        k = np.arange(self.N + 1, dtype=float)
        v0 = np.array([float(target_vel[0]), float(target_vel[1])])
        om = float(self.tgt_omega)
        if abs(om) < 1e-4:
            # stesso ORDINE di moltiplicazioni della formula di prima
            # (v*k)*dt e non v*(k*dt): in virgola mobile non e' associativa, e
            # cosi' il caso rettilineo resta identico bit per bit
            px = wp[0] + v0[0] * k * self.mpc_dt
            py = wp[1] + v0[1] * k * self.mpc_dt
            vx = np.full(self.N + 1, v0[0])
            vy = np.full(self.N + 1, v0[1])
        else:
            phi = om * k * self.mpc_dt
            s_, c_ = np.sin(phi), np.cos(phi)
            S11 = s_ / om
            S12 = -(1.0 - c_) / om
            S21 = (1.0 - c_) / om
            S22 = s_ / om
            px = wp[0] + S11 * v0[0] + S12 * v0[1]
            py = wp[1] + S21 * v0[0] + S22 * v0[1]
            vx = c_ * v0[0] - s_ * v0[1]
            vy = s_ * v0[0] + c_ * v0[1]
        return px, py, vx, vy

    def _mpc_horizontal(self, cur_x, cur_y, cur_vx, cur_vy, wp, a_xy_lim, target_vel):
        cur = np.array([cur_x, cur_y, cur_vx, cur_vy], dtype=float)
        if self.n_hrz == 6:
            # l'accelerazione non si misura: si usa lo stato del modello di
            # ritardo, aggiornato a ogni tick in compute() sul comando inviato
            cur = np.concatenate([cur, self._a_hrz])
        self.p_cur_hrz.value = cur
        self.p_axy_lim.value = a_xy_lim
        dxy = self.d_xy if self.dist_xy_enabled else np.zeros(2)
        self.p_dxy.value = dxy

        # ACCELERAZIONE CENTRIPETA DEL RIFERIMENTO
        # Su un arco il bersaglio non ha velocita' costante: la ruota, quindi ha
        # un'accelerazione a_c = omega x v, di modulo omega*|v| e diretta verso
        # il centro. Finche' questo termine non sta in u_ss, l'MPC lo deve
        # produrre pagandolo con il costo su (u - u_ss), e a regime resta un
        # errore verso l'ESTERNO pari a a_c diviso la rigidezza d'anello.
        # Misurato sui voli circolari del 26/10: a_c = omega*v = 0.043-0.048
        # m/s^2, e il disturbo stimato d_xy proiettato in radiale converge
        # proprio li' (+0.059 .. +0.114 verso fuori) -- cioe' l'osservatore
        # stava assorbendo la centripeta al posto del modello, con tau = 2.5 s,
        # troppo lento per una discesa di 5 s.
        om = float(self.tgt_omega) if self.a_cen_enabled else 0.0
        if abs(om) > 1e-4 and target_vel is not None:
            vtx = float(target_vel[0]); vty = float(target_vel[1])
            a_cen = np.array([-om * vty, om * vtx])   # omega x v
        else:
            a_cen = np.zeros(2)
        self.a_cen_ff = float(np.hypot(a_cen[0], a_cen[1]))

        # vx' = vx + dt (g*u1 + dx) ; vy' = vy + dt (-g*u0 + dy)
        # u_ss: ingresso "neutro", quello che il costo non penalizza. Deve
        # cancellare il disturbo E produrre la centripeta, altrimenti la curva
        # costa e l'MPC la fa a meta'.
        self.p_uss.value = np.clip(
            np.array([(dxy[1] - a_cen[1]) / self.g,
                      (a_cen[0] - dxy[0]) / self.g]),
            -a_xy_lim, a_xy_lim)

        # traiettoria del bersaglio: retta se tgt_omega = 0, arco altrimenti.
        # Messa in cache perche' la DEVE usare anche il cono, in _mpc_vertical:
        # se l'orizzontale inseguisse un arco e il cono misurasse il raggio
        # richiesto contro una retta, lo slack del cono sarebbe un artefatto.
        px, py, vx, vy = self._target_traj(wp, target_vel)
        self._tgt_traj = (px, py, vx, vy)
        xref_mat = np.vstack([px, py, vx, vy])
        self.p_xref_hrz.value = xref_mat

        # The max_iter value was 4000 before

        self.prob_hrz.solve(solver=cp.OSQP, warm_start=True, max_iter=5000, eps_abs=1e-3, eps_rel=1e-3)

        if self.u_hrz[:, 0].value is None:
            return 0.0, 0.0, None
        return float(self.u_hrz[0, 0].value), float(self.u_hrz[1, 0].value), self.x_hrz.value

    def _ge_accel(self, h):
        """a_ge(h) [m/s^2]: accelerazione in piu' a parita' di comando, in effetto suolo.

        h = distanza piano rotori - superficie. Congelata a ge_h_min: sotto il
        modello non e' validato e il polo in R/4 lo fa divergere.
        """
        h = np.maximum(np.asarray(h, float), self.ge_h_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 1.0 / np.maximum(1.0 - (self.ge_r_eff / (4.0 * h)) ** 2, 1e-3)
        k = np.clip(k, 1.0, self.ge_k_max)
        return self.g * (k - 1.0)

    def _ge_profile(self, cur_z, cur_vz):
        """a_ge sui passi 0..N-1.

        Il passo 0 usa la z MISURATA (che e' nota esattamente): prima usava la
        z predetta un passo avanti, il che introduceva un bias di |vz|*mpc_dt
        sul valore loggato e rendeva ge[0] non interpretabile a posteriori.
        I passi 1..N-1 usano la traiettoria predetta al solve precedente.
        Al primo solve dopo il latch non c'e' traiettoria valida: si restituisce
        zero invece di extrapolare a velocita' costante, che portava la z sotto
        la superficie e produceva un campione spurio a k_max (visto a z=1.55 m).
        """
        if not self.ge_enabled:
            return np.zeros(self.N)
        zp = self.x_vrt[0, :].value
        if zp is None or not np.all(np.isfinite(zp)):
            return np.zeros(self.N)          # primo solve: nessuna predizione
        zp = np.concatenate([zp[1:], zp[-1:]])   # soluzione precedente, shiftata
        zp = zp[:self.N].copy()
        zp[0] = cur_z                            # passo 0: quota misurata
        return self._ge_accel(zp - self.z_ground_ge)

    def _mpc_vertical(self, cur_z, cur_vz, wp, target_vel, x_pred=None, y_pred=None, apply_cone=False):
        use_cone = (apply_cone and self.cbf_cone_enabled and x_pred is not None and y_pred is not None)

        ge = self._ge_profile(cur_z, cur_vz)
        self.p_ge.value = ge
        self.p_d.value = float(self.d_hat) if self.dist_enabled else 0.0
        self.log_ge0 = float(ge[0])
        self.log_ge_h0 = float(max(cur_z - self.z_ground_ge, self.ge_h_min))

        # stato iniziale: con il ritardo c'e' una terza componente, l'accelerazione
        # verticale del modello, aggiornata a ogni tick in compute() (non si misura).
        if self.n_vrt == 3:
            self.p_cur_vrt.value = np.array([cur_z, cur_vz, self._a_vrt])
        else:
            self.p_cur_vrt.value = np.array([cur_z, cur_vz])
        self.p_xref_vrt.value = np.array([wp[2], 0.0])
        # Il pavimento del CONO deve stare SOTTO il bersaglio verticale, altrimenti
        # h0 -> 0 proprio al contatto e la barriera vieta di atterrare (provato il
        # 9/10: con p_z_plat = z_ground il margine scendeva a 0.1 cm e lo slack del
        # cono restava attivo per tutta la discesa). wp[2] e' gia' LAND_UNDERSHOOT
        # sotto il contatto: e' quello il margine che rende il landing ammissibile.
        self.p_z_plat.value = wp[2]

        # barriera di impatto: quota di contatto = self.z_ground (quota che
        # l'origine del corpo ha a contatto avvenuto), non la superficie.
        vz_tgt = float(target_vel[2]) if target_vel is not None else 0.0
        self.p_z_contact.value = float(self.z_ground)
        self.p_vz_tgt.value = vz_tgt
        self.p_imp_off.value = 0.0 if self.imp_enabled else 1e3
        self.log_hv0 = float(self.imp_vtd + (cur_vz - vz_tgt)
                             + self.imp_kv * (cur_z - self.z_ground))

        if use_cone:
            # STESSA traiettoria del bersaglio usata dall'orizzontale (cache
            # riempita in _mpc_horizontal). Se il solve orizzontale non e'
            # passato la cache non c'e' e si ricostruisce: costa quattro vettori.
            if self._tgt_traj is not None:
                _px, _py = self._tgt_traj[0], self._tgt_traj[1]
            else:
                _px, _py, _, _ = self._target_traj(wp, target_vel)
            ex = np.asarray(x_pred) - _px
            ey = np.asarray(y_pred) - _py
            r = self.alpha_cone * np.maximum(0, np.sqrt(ex ** 2 + ey ** 2) - self.r_base)
            self.p_r_cone.value = np.minimum(r, self.z_cut)
        else:
            # Vincolo CBF disattivato matematicamente
            self.p_r_cone.value = np.full(self.N + 1, -1000.0)

        # The max_iter value was 4000 before

        self.prob_vrt.solve(solver=cp.OSQP, warm_start=True, max_iter=5000, eps_abs=1e-3, eps_rel=1e-3)
        
        if self.u_vrt[0, 0].value is None:
            self.log_az_mpc = float("nan")
            self.log_eps0 = float("nan")
            self.log_eps_max = float("nan")
            self.log_eps_imp = float("nan")
            return 0.0
        az = float(self.u_vrt[0, 0].value)
        self.log_az_mpc = az
        eps = self.eps_cone.value
        if use_cone and eps is not None:
            eps = np.maximum(np.asarray(eps, float), 0.0)   # OSQP puo' dare -1e-6
            self.log_eps0 = float(eps[0])
            self.log_eps_max = float(eps.max())
        else:
            # cono disattivato (solve orizzontale fallito): slack senza significato
            self.log_eps0 = float("nan")
            self.log_eps_max = float("nan")
        ei = self.eps_imp.value
        self.log_eps_imp = (float(np.maximum(np.asarray(ei, float), 0.0).max())
                            if (self.imp_enabled and ei is not None) else float("nan"))
        return az