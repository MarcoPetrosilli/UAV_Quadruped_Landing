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
        self.A_hrz = np.array([[1, 0, self.mpc_dt, 0],
                               [0, 1, 0, self.mpc_dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.B_hrz = np.array([[0, 0],
                               [0, 0],
                               [0, g * self.mpc_dt],
                               [-g * self.mpc_dt, 0]])
        self.A_vrt = np.array([[1, self.mpc_dt], [0, 1]])
        self.B_vrt = np.array([[0], [self.mpc_dt]])

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
        self.Q_vrt = np.diag([3.0, 2.5])
        self.R_vrt = np.diag([8.0])
        
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
        self.ge_enabled = True
        self.ge_r_eff = 0.045        # raggio efficace [m]
        #self.ge_k_max = 1.1         # saturazione del guadagno di spinta
        self.ge_k_max = 1.1
        self.z_ground = 0.0          # quota del suolo [m] (da mocap, la passa il main)
        self.log_ge0 = float("nan")  # a_ge del primo passo [m/s^2], solo log

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

    def _compile_mpc_problems(self):
        # --- 1. HORIZONTAL PROBLEM ---
        self.x_hrz = cp.Variable((4, self.N + 1))
        self.u_hrz = cp.Variable((2, self.N))
        self.p_cur_hrz = cp.Parameter(4)
        self.p_xref_hrz = cp.Parameter((4, self.N + 1))
        self.p_axy_lim = cp.Parameter(nonneg=True)

        wQh=np.sqrt(np.diag(self.Q_hrz)); wRh=np.sqrt(np.diag(self.R_hrz)); wQhT=np.sqrt(10.0)*wQh
        cost_hrz = 0
        cons_hrz = [self.x_hrz[:, 0] == self.p_cur_hrz]
        for k in range(self.N):
            cost_hrz += cp.sum_squares(cp.multiply(wQh, self.x_hrz[:, k] - self.p_xref_hrz[:, k]))
            cost_hrz += cp.sum_squares(cp.multiply(wRh, self.u_hrz[:, k]))
            cons_hrz += [self.x_hrz[:, k + 1] == self.A_hrz @ self.x_hrz[:, k] + self.B_hrz @ self.u_hrz[:, k]]
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

        for k in range(self.N):
            cost_vrt += cp.sum_squares(cp.multiply(wQv, self.x_vrt[:, k] - self.p_xref_vrt))
            cost_vrt += cp.sum_squares(cp.multiply(wRv, self.u_vrt[:, k]))
            cons_vrt += [self.x_vrt[:, k + 1] == self.A_vrt @ self.x_vrt[:, k]
                         + self.B_vrt @ (self.u_vrt[:, k] + self.p_ge[k])]
            cons_vrt += [cp.abs(self.u_vrt[:, k]) <= 9.0]

            # --- livello 2 (DHOCBF grado 2), rilassato con slack ---
            # h2_k = h1_{k+1} - (1-gamma2) h1_k >= -eps_k ,  valido per k = 0 .. N-2
            # (h2_k usa h0_{k+2} = z_{k+2}, l'ingresso az_k compare qui: grado rel. 2)
            if k <= self.N - 2:
                h2_k = h1[k + 1] - (1.0 - self.gamma2_cbf) * h1[k]
                cons_vrt += [h2_k >= -self.eps_cone[k]]

        # penalita' quadratica sullo slack (feasibility, es. al vertice del cono)
        cost_vrt += self.rho_slack * cp.sum_squares(self.eps_cone)
        cost_vrt += cp.sum_squares(cp.multiply(wQvT, self.x_vrt[:, self.N] - self.p_xref_vrt))
        self.prob_vrt = cp.Problem(cp.Minimize(cost_vrt), cons_vrt)

        # --- AGGIUNGI QUESTO DUMMY SOLVE (WARM-UP) ---
        print("Warm-up MPC solver...")
        self.p_cur_hrz.value = np.zeros(4)
        self.p_xref_hrz.value = np.zeros((4, self.N + 1))
        self.p_axy_lim.value = 0.17
        self.prob_hrz.solve(solver=cp.OSQP)
        
        self.p_cur_vrt.value = np.zeros(2)
        self.p_xref_vrt.value = np.zeros(2)
        self.p_z_plat.value = 0.0
        self.p_r_cone.value = np.full(self.N + 1, -100.0)
        self.p_ge.value = np.zeros(self.N)
        self.prob_vrt.solve(solver=cp.OSQP)
        print("Warm-up completato.")

    # ==================================================================== #
    #  Entry point: stato (pos, vel) -> (forza, roll, pitch, yaw, modo)     #
    # ==================================================================== #
    def compute(self, cur_pos, cur_vel, target_pos, target_yaw=0.0,
                target_vel=None, a_xy_lim=0.17, final_pos=None, landing=False,
                ramp_ref_vel=None, z_ground=None):
        self.control_counter += 1
        if z_ground is not None:
            self.z_ground = float(z_ground)
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
            self.mpc_step = 0
            force, euler = self._pid_reach(
                cur_pos, cur_vel, target_pos, target_yaw, target_vel + ramp_ref_vel)
            self.last_pid_force, self.last_pid_euler = force, np.asarray(euler, float).copy()
            mode = 0

        return force, float(euler[0]), float(euler[1]), float(euler[2]), mode

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

    def _mpc_horizontal(self, cur_x, cur_y, cur_vx, cur_vy, wp, a_xy_lim, target_vel):
        self.p_cur_hrz.value = np.array([cur_x, cur_y, cur_vx, cur_vy])
        self.p_axy_lim.value = a_xy_lim

        xref_mat = np.zeros((4, self.N + 1))
        for k in range(self.N + 1):
            xref_mat[:, k] = np.array([wp[0] + target_vel[0] * k * self.mpc_dt,
                                       wp[1] + target_vel[1] * k * self.mpc_dt,
                                       target_vel[0], target_vel[1]])
        self.p_xref_hrz.value = xref_mat

        # The max_iter value was 4000 before

        self.prob_hrz.solve(solver=cp.OSQP, warm_start=True, max_iter=5000, eps_abs=1e-3, eps_rel=1e-3)

        if self.u_hrz[:, 0].value is None:
            return 0.0, 0.0, None
        return float(self.u_hrz[0, 0].value), float(self.u_hrz[1, 0].value), self.x_hrz.value

    def _ge_accel(self, h):
        """a_ge(h) [m/s^2]: accelerazione in piu' a parita' di comando, in effetto suolo."""
        h = np.maximum(np.asarray(h, float), 1e-3)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 1.0 / np.maximum(1.0 - (self.ge_r_eff / (4.0 * h)) ** 2, 1e-3)
        k = np.clip(k, 1.0, self.ge_k_max)
        return self.g * (k - 1.0)

    def _ge_profile(self, cur_z, cur_vz):
        """a_ge sui passi 0..N-1, valutata sulla z predetta al solve precedente."""
        if not self.ge_enabled:
            return np.zeros(self.N)
        zp = self.x_vrt[0, :].value
        if zp is None or not np.all(np.isfinite(zp)):
            # primo solve: propagazione a velocita' costante
            zp = cur_z + cur_vz * self.mpc_dt * np.arange(self.N + 1)
        else:
            # la soluzione precedente e' shiftata di un passo
            zp = np.concatenate([zp[1:], zp[-1:]])
        return self._ge_accel(zp[:self.N] - self.z_ground)

    def _mpc_vertical(self, cur_z, cur_vz, wp, target_vel, x_pred=None, y_pred=None, apply_cone=False):
        use_cone = (apply_cone and self.cbf_cone_enabled and x_pred is not None and y_pred is not None)

        ge = self._ge_profile(cur_z, cur_vz)
        self.p_ge.value = ge
        self.log_ge0 = float(ge[0])

        self.p_cur_vrt.value = np.array([cur_z, cur_vz])
        self.p_xref_vrt.value = np.array([wp[2], 0.0])
        self.p_z_plat.value = wp[2]

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
        return az