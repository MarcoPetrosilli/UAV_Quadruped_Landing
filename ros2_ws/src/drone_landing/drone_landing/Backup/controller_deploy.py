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

La piattaforma mobile e' ignorata: target statico, target_vel = 0.
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

        # ---- MPC prediction models -----------------------------------------
        # ORIZZONTALE: il modello originale assume che l'assetto comandato
        # diventi accelerazione ISTANTANEAMENTE (v_{k+1} = v_k + g dt u_k).
        # Il loop di assetto del firmware ci mette invece ~171 ms, misurati per
        # correlazione fra l'accelerazione comandata da roll/pitch e quella da
        # dv/dt su 4 voli e 2 assi (160-181 ms, corr 0.82-0.96). Un controllore
        # tarato ignorando quel ritardo risuona a 1/(2 pi tau) = 0.93 Hz, cioe'
        # periodo 1.07 s: ed e' esattamente l'oscillazione orizzontale di
        # +-1.5 cm con periodo 0.8-1.2 s presente in TUTTI i voli, anche quelli
        # del 21/9 senza offset-free orizzontale (l'osservatore la amplificava,
        # non la creava).
        # Con att_lag_enabled lo stato diventa [x, y, vx, vy, ax, ay] e
        # l'assetto entra attraverso un primo ordine:
        #     a_{k+1} = (1 - lam) a_k + lam * g u_k ,   lam = mpc_dt / att_tau
        #     v_{k+1} = v_k + mpc_dt (a_k + d)
        # Resta lineare, quindi resta un QP: cambia solo la dimensione (6 stati
        # invece di 4). L'MPC sa che la sua azione arriva in ritardo e smette di
        # sovracorreggere, senza dover detunare i pesi.
        self.att_lag_enabled = True
        self.att_tau = 0.17          # [s] costante del loop di assetto (misurata)
        self.q_acc_hrz = 0.5         # peso sugli stati di accelerazione
        self.a_lag = np.zeros(2)     # accelerazione orizzontale "realizzata"
        self.A_vrt = np.array([[1, self.mpc_dt], [0, 1]])
        self.B_vrt = np.array([[0], [self.mpc_dt]])
        self._build_hrz_model()

        # ---- DSL PID gains (reach) -----------------------------------------
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.0, .0, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])


        #self.P_COEFF_FOR = np.array([.45, .45, 1.25])
        #self.D_COEFF_FOR = np.array([.15, .15, .5])

        # ---- MPC weights ---------------------------------------------------
        #self.Q_hrz = np.diag([20.0, 20.0, 15.0, 15.0])
        #self.R_hrz = np.diag([20.0, 20.0])
        #self.Q_vrt = np.diag([20.0, 15.0])
        #self.R_vrt = np.diag([15.0])

        self.Q_hrz = np.diag([2.0, 2.0, 1.5, 1.5])
        self.R_hrz = np.diag([5.0, 5.0])
        self.Q_vrt = np.diag([2.0, 1.5])
        self.R_vrt = np.diag([5.0])
        
        # ---- Cone CBF (glideslope) -----------------------------------------
        self.cbf_cone_enabled = True
        #self.alpha_cone = 1.72

        self.alpha_cone = 1.0
        self.v_max_hrz = 0.9          # limite velocita' orizzontale [m/s] (faccia del politopo imposta nell'MPC)
        # --- DHOCBF grado 2: due coefficienti di classe-K (uno per livello) ---
        # catena  h0 -> h1 -> h2 ,  vincolo su h2 >= 0 (rilassato con slack).
        # Inizializzati uguali cosi' si parte da un comportamento noto; tarabili
        # indipendentemente (e' il vantaggio della HOCBF).
        self.gamma_cbf = 0.5          # mantenuto per retro-compatibilita' / riferimento
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
        # siccome il clamp k_max scatta sotto 3.4 cm, il modello saturava a
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
        self.dist_tau = 0.6          # costante del filtro [s]
        self.dist_win = 0.10         # finestra di stima [s]
        self.dist_max = 1.5          # |d| massimo [m/s^2] (~15% di g)
        self.dist_freeze_h = 0.03    # sotto questa quota sul contatto la stima si
                                     # ferma [m]. Alzata da 0.02 ora che il landing
                                     # prosegue FINO al contatto: il transitorio di
                                     # appoggio non deve entrare nel disturbo.
        self.d_hat = 0.0
        # Stessa struttura sul piano orizzontale: disturbi dx, dy [m/s^2] che
        # raccolgono i bias costanti (baricentro non centrato, trim d'assetto,
        # offset dell'IMU) responsabili dell'errore fisso di 2-4 cm visto al
        # contatto (voli 175138, 175926). Costante piu' lunga dell'asse z per
        # non inseguire la resistenza aerodinamica, che dipende dalla velocita'.
        self.dist_xy_enabled = True
        self.dist_xy_tau = 2.5       # [s]. Era 1.0, cioe' la stessa scala del
                                     # ciclo limite orizzontale misurato in coda
                                     # al landing (periodo ~1 s, +-1.5 cm, con
                                     # d_x in fase con vx a +-0.10 m/s^2, voli
                                     # 21/9 e 1/10): l'osservatore inseguiva la
                                     # propria azione di controllo invece del
                                     # disturbo. Il disturbo xy vero e' un bias
                                     # quasi costante (baricentro, trim), quindi
                                     # rallentare non costa nulla.
        self.dist_xy_max = 0.5       # |d_xy| massimo [m/s^2] (~3 deg di assetto)
        self.d_xy = np.zeros(2)
        self._obs_prev = None        # (a_cmd [ax, ay, az], a_ge, t) del tick precedente
        self._obs_v0 = None          # velocita' [vx, vy, vz] a inizio finestra
        self._obs_int = np.zeros(3)  # integrale di (a_cmd + a_ge) sulla finestra
        self._obs_T = 0.0            # durata della finestra
        self.log_d_hat = float("nan")
        self.log_d_xy = np.full(2, float("nan"))
        self.log_a_lag = np.zeros(2)   # accelerazione orizzontale realizzata (modello)

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
        self.imp_vtd = 0.08          # velocita' di contatto ammessa [m/s]
        self.imp_kv = 1.3            # pendenza della barriera [1/s]: con vz di
                                     # crociera 0.5 m/s morde sotto ~32 cm
        self.imp_gamma = 0.03        # classe-K discreta. ATTENZIONE alla scala:
                                     # h(k+1) >= (1-g) h(k) equivale in continuo
                                     # a h_dot >= -alpha h con alpha = g/mpc_dt.
                                     # Con mpc_dt = 0.04 s, g = 0.5 da' alpha =
                                     # 12.5 1/s (tau 80 ms): il margine puo'
                                     # collassare in un istante e il vincolo non
                                     # limita il RITMO di avvicinamento, solo il
                                     # segno. g = 0.03 -> alpha = 0.8 1/s: in
                                     # simulazione il margine minimo raddoppia
                                     # (+0.016 -> +0.039) e il contatto passa da
                                     # 0.063 a 0.039 m/s, costando 0.9 s.
                                     # NB: protegge dall'ottimismo della
                                     # predizione, NON dall'errore di modello;
                                     # per quello contano imp_vtd e imp_kv.
        self.rho_imp = 1e6           # peso dello slack (>> rho_slack del cono)
        self.log_hv0 = float("nan")  # valore della barriera al passo 0 [m/s]
        self.log_eps_imp = float("nan")

        # ---- controllo di imbardata -----------------------------------------
        # Finora lo yaw non era controllato: a send_setpoint andava rate 0 e il
        # controllore assumeva che il muso del drone fosse lungo world +x. Il
        # firmware pero' integra lo yaw dal giroscopio a partire dal reset, cioe'
        # "yaw = 0" significa "come era appoggiato", non "lungo +x". Il 1/10
        # questo e' costato un drone contro il muro: posato ruotato di ~90 gradi,
        # l'anello orizzontale si e' chiuso in retroazione POSITIVA.
        #
        # Errore: si prendono le due matrici di rotazione rispetto al mondo e si
        # guarda la rotazione relativa
        #     R_err = R_drone^T R_target
        # di cui si prende la rappresentazione ANGOLO-ASSE (log della matrice).
        # La componente z del vettore di rotazione e' l'angolo da annullare: e'
        # ben definita anche con la piattaforma inclinata, dove prendere la
        # differenza di due yaw estratti separatamente darebbe risultati diversi
        # a seconda della convenzione di estrazione.
        #
        # Uscita: VELOCITA' di imbardata, perche' il terzo argomento di
        # send_setpoint e' un rate, non un angolo.
        self.yaw_enabled = True
        self.yaw_kp = 2.0            # [1/s] guadagno (costante di tempo 0.5 s)
        self.yaw_rate_max = 1.57     # [rad/s] ~90 deg/s
        self.yaw_deadband = 0.035    # [rad] ~2 deg: sotto, rate zero (anti-jitter)
        # Simmetria del telaio: il Crazyflie in X e' indistinguibile a multipli
        # di 90 gradi, quindi per appoggiarsi basta allinearsi al multiplo piu'
        # vicino invece di fare mezzo giro. 360 = allineamento vero (default,
        # come richiesto); 90 = sfrutta la simmetria.
        self.yaw_symmetry = 2.0 * math.pi
        # SEGNO DEL RATE. L'errore e' calcolato in convenzione mondo (ENU,
        # z verso l'alto, yaw antiorario). Il commander legacy del firmware
        # Crazyflie nega il campo yawrate del pacchetto CRTP, quindi un valore
        # POSITIVO inviato a send_setpoint fa ruotare il drone in senso ORARIO.
        # Con -1 l'anello e' in retroazione negativa; se sul tuo firmware la
        # convenzione fosse l'altra, l'imbardata divergerebbe. Da verificare a
        # terra prima del primo volo (vedi nota nel main), e comunque sotto c'e'
        # la protezione automatica.
        self.yaw_rate_sign = -1.0
        # Protezione: se l'errore cresce in modulo per piu' di yaw_div_n
        # campioni consecutivi mentre si sta comandando un rate, il segno e'
        # sbagliato o il loop e' instabile. Si spegne il controllo di imbardata
        # invece di lasciare il drone avvitarsi.
        self.yaw_div_n = 40
        self._yaw_div = 0
        self._yaw_e_prev = None
        self.yaw_drone = 0.0         # imbardata corrente [rad], da R_drone
        self.log_yaw_err = float("nan")
        self.log_yaw_rate = float("nan")
        self.yaw_tripped = False     # True se la protezione e' scattata

        self.z_cut = 1.0
        self.r_base = 0.3
        self.rho_slack = 1e4        # peso penalita' slack del cono (grande = slack usato solo se necessario)

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
        # Si sfuma linearmente dall'ultima uscita PID (ancora) all'uscita MPC su
        # blend_duration secondi, poi MPC puro.
        self.blend_duration = 0.8
        self.blend_steps_total = max(1, int(round(self.blend_duration / self.dt)))
        self.blend_steps_left = 0
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
        """Matrici del modello orizzontale, con o senza ritardo di assetto."""
        dt, g = self.mpc_dt, self.g
        if self.att_lag_enabled:
            lam = float(np.clip(dt / max(self.att_tau, 1e-3), 0.0, 1.0))
            self.n_hrz = 6
            self.A_hrz = np.array([[1, 0, dt, 0,  0,  0],
                                   [0, 1, 0, dt,  0,  0],
                                   [0, 0, 1, 0,  dt,  0],
                                   [0, 0, 0, 1,   0, dt],
                                   [0, 0, 0, 0, 1 - lam, 0],
                                   [0, 0, 0, 0, 0, 1 - lam]], dtype=float)
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
            self.B_hrz = np.array([[0, 0], [0, 0], [0, g * dt], [-g * dt, 0]], dtype=float)
            self.E_hrz = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]], dtype=float)

    def _q_hrz_diag(self):
        """Diagonale di Q orizzontale, estesa agli stati di accelerazione."""
        q = np.diag(self.Q_hrz).astype(float)
        if self.att_lag_enabled:
            q = np.concatenate([q, np.full(2, self.q_acc_hrz)])
        return q

    def _compile_mpc_problems(self):
        # --- 1. HORIZONTAL PROBLEM ---
        self._build_hrz_model()
        nh = self.n_hrz
        self.x_hrz = cp.Variable((nh, self.N + 1))
        self.u_hrz = cp.Variable((2, self.N))
        self.p_cur_hrz = cp.Parameter(nh)
        self.p_xref_hrz = cp.Parameter((nh, self.N + 1))
        self.p_axy_lim = cp.Parameter(nonneg=True)
        self.p_dxy = cp.Parameter(2)     # disturbo stimato [dx, dy] (offset-free)
        self.p_uss = cp.Parameter(2)     # ingresso di equilibrio [dy/g, -dx/g]
        E_hrz = self.E_hrz

        wQh=np.sqrt(self._q_hrz_diag()); wRh=np.sqrt(np.diag(self.R_hrz)); wQhT=np.sqrt(10.0)*wQh
        cost_hrz = 0
        cons_hrz = [self.x_hrz[:, 0] == self.p_cur_hrz]
        for k in range(self.N):
            cost_hrz += cp.sum_squares(cp.multiply(wQh, self.x_hrz[:, k] - self.p_xref_hrz[:, k]))
            # costo su u - u_ss: all'equilibrio serve l'assetto che compensa d
            cost_hrz += cp.sum_squares(cp.multiply(wRh, self.u_hrz[:, k] - self.p_uss))
            cons_hrz += [self.x_hrz[:, k + 1] == self.A_hrz @ self.x_hrz[:, k]
                         + self.B_hrz @ self.u_hrz[:, k] + E_hrz @ self.p_dxy]
            cons_hrz += [cp.abs(self.u_hrz[:, k]) <= self.p_axy_lim]
            #cons_hrz += [cp.abs(self.x_hrz[2, k]) <= self.v_max_hrz]   # |vx| <= v_max
            #cons_hrz += [cp.abs(self.x_hrz[3, k]) <= self.v_max_hrz]   # |vy| <= v_max
        cost_hrz += cp.sum_squares(cp.multiply(wQhT, self.x_hrz[:, self.N] - self.p_xref_hrz[:, self.N]))
        self.prob_hrz = cp.Problem(cp.Minimize(cost_hrz), cons_hrz)

        # --- 2. VERTICAL PROBLEM (Unificato con Cono CBF) ---
        self.x_vrt = cp.Variable((2, self.N + 1))
        self.u_vrt = cp.Variable((1, self.N))
        self.p_cur_vrt = cp.Parameter(2)
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
            cost_vrt += cp.sum_squares(cp.multiply(wQv, self.x_vrt[:, k] - self.p_xref_vrt))
            # costo sull'accelerazione NETTA (u + effetto suolo + disturbo), non
            # su u: all'equilibrio sul target serve u = -(a_ge + d) != 0, e un
            # costo su u da solo farebbe preferire all'MPC un errore di quota
            # residuo pur di non spendere ingresso (plateau). Pesare la netta
            # equivale a pesare u - u_ss, come nella formulazione offset-free.
            cost_vrt += cp.sum_squares(cp.multiply(
                wRv, self.u_vrt[:, k] + self.p_ge[k] + self.p_d))
            cons_vrt += [self.x_vrt[:, k + 1] == self.A_vrt @ self.x_vrt[:, k]
                         + self.B_vrt @ (self.u_vrt[:, k] + self.p_ge[k] + self.p_d)]
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
        cost_vrt += cp.sum_squares(cp.multiply(wQvT, self.x_vrt[:, self.N] - self.p_xref_vrt))
        self.prob_vrt = cp.Problem(cp.Minimize(cost_vrt), cons_vrt)

        # --- AGGIUNGI QUESTO DUMMY SOLVE (WARM-UP) ---
        print("Warm-up MPC solver...")
        self.p_cur_hrz.value = np.zeros(self.n_hrz)
        self.p_xref_hrz.value = np.zeros((self.n_hrz, self.N + 1))
        self.p_axy_lim.value = 0.17
        self.p_dxy.value = np.zeros(2)
        self.p_uss.value = np.zeros(2)
        self.prob_hrz.solve(solver=cp.OSQP)
        
        self.p_cur_vrt.value = np.zeros(2)
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
                cmd_saturated_prev=False, R_drone=None, R_target=None):
        # R_drone / R_target: matrici di rotazione rispetto al MONDO (3x3).
        # Se mancano, lo yaw non viene controllato e si assume muso lungo +x,
        # cioe' esattamente il comportamento precedente.
        self.control_counter += 1
        if R_drone is not None:
            Rd = np.asarray(R_drone, float)
            self.yaw_drone = math.atan2(Rd[1, 0], Rd[0, 0])
        yaw_rate = self._yaw_rate_cmd(R_drone, R_target)
        if z_ground is not None:
            self.z_ground = float(z_ground)
        # se il main non passa la superficie si ricade sulla quota di contatto
        # (comportamento vecchio, h sottostimata): serve solo per compatibilita'
        if z_ground_ge is not None:
            self.z_ground_ge = float(z_ground_ge)
        elif z_ground is not None:
            self.z_ground_ge = float(z_ground)
        now = time.perf_counter()
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
                # fronte di salita del gate: congela l'ultima uscita PID come
                # ancora di partenza e apri la finestra di blend (vedi __init__)
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
                force = (1.0 - alpha) * self.blend_anchor_force + alpha * force
                euler = (1.0 - alpha) * self.blend_anchor_euler + alpha * euler
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
            # ATTENZIONE: i roll/pitch del DSL-PID sono riferiti al MONDO, non
            # al corpo. Passargli lo yaw reale NON li ruota: l'estrazione
            # as_euler('XYZ') restituisce lo stesso pitch per qualunque yaw
            # (verificato: errore verso +x -> pitch +0.312 a psi = 0, 30, 90 e
            # 180 gradi). Al primo volo con yaw attivo questo ha fatto accelerare
            # il drone in +y mentre chiedeva +x, cioe' retroazione positiva.
            # La rotazione si fa in fondo a compute(), per entrambi i rami.
            force, euler = self._pid_reach(
                cur_pos, cur_vel, target_pos, 0.0, target_vel + ramp_ref_vel)
            self.last_pid_force, self.last_pid_euler = force, np.asarray(euler, float).copy()
            mode = 0

        # Il terzo canale non e' piu' un angolo ma la VELOCITA' di imbardata,
        # che e' quello che si manda a send_setpoint. Viene scritto DOPO il
        # blend: il blend serve a non avere gradini di spinta e assetto al
        # passaggio PID -> MPC, ma l'anello di yaw e' lo stesso nelle due
        # modalita', quindi non c'e' niente da sfumare.
        euler = np.asarray(euler, float).copy()
        euler[2] = yaw_rate

        # --- MONDO -> CORPO, una volta sola per entrambi i rami ---------------
        # Fin qui roll e pitch esprimono la direzione della spinta nel frame del
        # mondo; il firmware li interpreta nel frame del drone. Si converte in
        # accelerazione, si ruota di -psi e si torna agli angoli: esatto, non
        # un'approssimazione a piccoli angoli.
        phi_w, theta_w = float(euler[0]), float(euler[1])
        fm_w = max(force / self.M, 1e-6)
        ax_w = fm_w * math.cos(phi_w) * math.sin(theta_w)
        ay_w = -fm_w * math.sin(phi_w)
        az_w = fm_w * math.cos(phi_w) * math.cos(theta_w) - self.g
        cpsi, spsi = math.cos(self.yaw_drone), math.sin(self.yaw_drone)
        ax_b = cpsi * ax_w + spsi * ay_w
        ay_b = -spsi * ax_w + cpsi * ay_w
        phi_b = math.asin(float(np.clip(-ay_b / fm_w, -0.999, 0.999)))
        theta_b = math.asin(float(np.clip(ax_b / (fm_w * max(math.cos(phi_b), 1e-6)),
                                          -0.999, 0.999)))
        euler[0], euler[1] = phi_b, theta_b

        # accelerazione comandata in questo tick, nel frame MONDO: e' gia'
        # stata calcolata sopra dagli angoli prima della rotazione
        a_cmd = np.array([ax_w, ay_w, az_w])
        ge_now = float(self._ge_accel(float(cur_pos[2]) - self.z_ground_ge)) if self.ge_enabled else 0.0
        # stato del ritardo di assetto: primo ordine sull'accelerazione
        # orizzontale comandata in QUESTO tick (PID o MPC, blend incluso), con
        # la costante del loop di assetto. E' la condizione iniziale degli stati
        # ax, ay del modello orizzontale.
        if self.att_lag_enabled:
            k_att = float(np.clip(self.dt / max(self.att_tau, 1e-3), 0.0, 1.0))
            self.a_lag += k_att * (a_cmd[0:2] - self.a_lag)
        self.log_a_lag = self.a_lag.copy()
        self._obs_prev = (a_cmd, ge_now, now)
        self.log_d_hat = self.d_hat
        self.log_d_xy = self.d_xy.copy()
        return force, float(euler[0]), float(euler[1]), float(euler[2]), mode

    def _yaw_error(self, R_drone, R_target):
        """Errore di imbardata [rad] dalla rotazione relativa, in angolo-asse.

        R_err = R_drone^T R_target porta il frame del drone su quello del
        target; il logaritmo della matrice ne da' il vettore rotazione (asse per
        angolo) e la sua componente z e' l'imbardata da annullare.
        """
        R_err = np.asarray(R_drone, float).T @ np.asarray(R_target, float)
        e = float(Rotation.from_matrix(R_err).as_rotvec()[2])
        # riporta nell'intervallo (-sym/2, +sym/2]: con sym = 2 pi e' il solito
        # wrap a +-180 gradi, con sym = pi/2 sfrutta la simmetria del telaio
        sym = float(self.yaw_symmetry)
        if sym > 0:
            e -= sym * math.floor(e / sym + 0.5)
        return e

    def _yaw_rate_cmd(self, R_drone, R_target):
        """Velocita' di imbardata comandata [rad/s]."""
        if not self.yaw_enabled or R_drone is None or R_target is None:
            self.log_yaw_err = float("nan")
            self.log_yaw_rate = 0.0
            return 0.0
        e = self._yaw_error(R_drone, R_target)
        self.log_yaw_err = e
        if abs(e) < self.yaw_deadband:
            self._yaw_div = 0
            self._yaw_e_prev = e
            self.log_yaw_rate = 0.0
            return 0.0
        # rilevamento di divergenza (segno sbagliato o loop instabile)
        if self._yaw_e_prev is not None:
            self._yaw_div = self._yaw_div + 1 if abs(e) > abs(self._yaw_e_prev) + 1e-4 else 0
            if self._yaw_div >= self.yaw_div_n:
                self.yaw_enabled = False
                self.yaw_tripped = True
                self.log_yaw_rate = 0.0
                return 0.0
        self._yaw_e_prev = e
        r = float(np.clip(self.yaw_kp * e, -self.yaw_rate_max, self.yaw_rate_max))
        r *= self.yaw_rate_sign
        self.log_yaw_rate = r
        return r

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
        u0, u1, x_hrz = self._mpc_horizontal(
            cx, cy, vx, vy, wp, a_xy_lim, target_vel)

        # angoli riferiti al MONDO (B_hrz: a_x = g u1, a_y = -g u0). La
        # rotazione nel frame del drone si fa una volta sola in compute().
        phi_cmd, theta_cmd = u0, u1

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

    def _mpc_horizontal(self, cur_x, cur_y, cur_vx, cur_vy, wp, a_xy_lim, target_vel):
        if self.att_lag_enabled:
            # lo stato del ritardo non si misura: e' una funzione nota dei
            # comandi passati, mantenuta in compute() filtrando l'assetto
            # applicato (quindi corretta anche dopo la fase PID e il blend)
            self.p_cur_hrz.value = np.array([cur_x, cur_y, cur_vx, cur_vy,
                                             self.a_lag[0], self.a_lag[1]])
        else:
            self.p_cur_hrz.value = np.array([cur_x, cur_y, cur_vx, cur_vy])
        self.p_axy_lim.value = a_xy_lim
        dxy = self.d_xy if self.dist_xy_enabled else np.zeros(2)
        self.p_dxy.value = dxy
        # all'equilibrio: a = -d  e  g*u1 = a_x, -g*u0 = a_y -> u = [dy/g, -dx/g]
        # (identico nei due modelli: il ritardo non sposta il punto di lavoro)
        self.p_uss.value = np.clip(np.array([dxy[1] / self.g, -dxy[0] / self.g]),
                                   -a_xy_lim, a_xy_lim)

        xref_mat = np.zeros((self.n_hrz, self.N + 1))
        for k in range(self.N + 1):
            xref_mat[0:4, k] = np.array([wp[0] + target_vel[0] * k * self.mpc_dt,
                                         wp[1] + target_vel[1] * k * self.mpc_dt,
                                         target_vel[0], target_vel[1]])
        if self.att_lag_enabled:
            # accelerazione di equilibrio: quella che cancella il disturbo
            xref_mat[4, :] = -dxy[0]
            xref_mat[5, :] = -dxy[1]
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

        self.p_cur_vrt.value = np.array([cur_z, cur_vz])
        self.p_xref_vrt.value = np.array([wp[2], 0.0])
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
            ex = np.asarray(x_pred) - (wp[0] + target_vel[0] * np.arange(self.N + 1) * self.mpc_dt)
            ey = np.asarray(y_pred) - (wp[1] + target_vel[1] * np.arange(self.N + 1) * self.mpc_dt)
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