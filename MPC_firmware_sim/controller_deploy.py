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
        self.dt = dt
        self.M = mass
        self.GRAVITY = mass * g
        self.N = 20

        # ---- MPC prediction models (identici al sim) -----------------------
        self.A_hrz = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.B_hrz = np.array([[0, 0],
                               [0, 0],
                               [0, g * dt],
                               [-g * dt, 0]])
        self.A_vrt = np.array([[1, dt], [0, 1]])
        self.B_vrt = np.array([[0], [dt]])

        # ---- DSL PID gains (reach) -----------------------------------------
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.0, .0, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])

        # ---- MPC weights ---------------------------------------------------
        self.Q_hrz = np.diag([20.0, 20.0, 10.0, 10.0])
        self.R_hrz = np.diag([10.0, 10.0])
        self.Q_vrt = np.diag([30.0, 15.0])
        self.R_vrt = np.diag([15.0])

        # ---- Cone CBF (glideslope) -----------------------------------------
        self.cbf_cone_enabled = True
        self.alpha_cone = 1.72
        self.gamma_cbf = 0.1
        self.z_cut = 1.0
        self.r_base = 0.1

        # ---- multi-rate decimation -----------------------------------------
        self.MPC_FREQ_DIVIDER = 1
        self.control_counter = 0
        self.mpc_step = 0
        self.mpc_activated = False
        self.last_force = self.GRAVITY
        self.last_euler = np.zeros(3)
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

        cost_hrz = 0
        cons_hrz = [self.x_hrz[:, 0] == self.p_cur_hrz]
        for k in range(self.N):
            cost_hrz += cp.quad_form(self.x_hrz[:, k] - self.p_xref_hrz[:, k], self.Q_hrz)
            cost_hrz += cp.quad_form(self.u_hrz[:, k], self.R_hrz)
            cons_hrz += [self.x_hrz[:, k + 1] == self.A_hrz @ self.x_hrz[:, k] + self.B_hrz @ self.u_hrz[:, k]]
            cons_hrz += [cp.abs(self.u_hrz[:, k]) <= self.p_axy_lim]
        cost_hrz += 10.0 * cp.quad_form(self.x_hrz[:, self.N] - self.p_xref_hrz[:, self.N], self.Q_hrz)
        self.prob_hrz = cp.Problem(cp.Minimize(cost_hrz), cons_hrz)

        # --- 2. VERTICAL PROBLEM (Unificato con Cono CBF) ---
        self.x_vrt = cp.Variable((2, self.N + 1))
        self.u_vrt = cp.Variable((1, self.N))
        self.p_cur_vrt = cp.Parameter(2)
        self.p_xref_vrt = cp.Parameter(2)
        self.p_z_plat = cp.Parameter()
        self.p_r_cone = cp.Parameter(self.N + 1)

        cost_vrt = 0
        cons_vrt = [self.x_vrt[:, 0] == self.p_cur_vrt]
        for k in range(self.N):
            cost_vrt += cp.quad_form(self.x_vrt[:, k] - self.p_xref_vrt, self.Q_vrt)
            cost_vrt += cp.quad_form(self.u_vrt[:, k], self.R_vrt)
            cons_vrt += [self.x_vrt[:, k + 1] == self.A_vrt @ self.x_vrt[:, k] + self.B_vrt @ self.u_vrt[:, k]]
            cons_vrt += [cp.abs(self.u_vrt[:, k]) <= 9.0]
            
            # Vincolo CBF (disattivato dinamicamente passando raggi negativi enormi)
            h_k = (self.x_vrt[0, k] - self.p_z_plat) - self.p_r_cone[k]
            h_k1 = (self.x_vrt[0, k + 1] - self.p_z_plat) - self.p_r_cone[k + 1]
            cons_vrt += [h_k1 >= (1.0 - self.gamma_cbf) * h_k]
            
        cost_vrt += 10.0 * cp.quad_form(self.x_vrt[:, self.N] - self.p_xref_vrt, self.Q_vrt)
        self.prob_vrt = cp.Problem(cp.Minimize(cost_vrt), cons_vrt)

    # ==================================================================== #
    #  Entry point: stato (pos, vel) -> (forza, roll, pitch, yaw, modo)     #
    # ==================================================================== #
    def compute(self, cur_pos, cur_vel, target_pos, target_yaw=0.0,
                target_vel=None, a_xy_lim=0.17, final_pos=None, landing=False):
        self.control_counter += 1
        cur_pos = np.asarray(cur_pos, float)
        cur_vel = np.asarray(cur_vel, float)
        target_pos = np.asarray(target_pos, float)
        target_vel = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)
        wp_final = np.asarray(final_pos, float) if final_pos is not None else target_pos

        in_set = self.is_in_reachable_set(cur_pos, cur_vel, wp_final, target_vel)

        if (in_set and landing) or self.mpc_activated:
            # ---------------- MPC MODE ----------------
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
            mode = 1
        else:
            # ---------------- PID REACH MODE ----------------
            self.mpc_step = 0
            force, euler = self._pid_reach(
                cur_pos, cur_vel, target_pos, target_yaw, target_vel)
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
        
        # Invertito theta_corr per adattamento al firmware Crazyflie
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
            xref_mat[:, k] = np.array([wp[0] + target_vel[0] * k * self.dt,
                                       wp[1] + target_vel[1] * k * self.dt,
                                       target_vel[0], target_vel[1]])
        self.p_xref_hrz.value = xref_mat

        self.prob_hrz.solve(solver=cp.OSQP, warm_start=True, max_iter=4000, eps_abs=1e-4, eps_rel=1e-4)

        if self.u_hrz[:, 0].value is None:
            return 0.0, 0.0, None
        return float(self.u_hrz[0, 0].value), float(self.u_hrz[1, 0].value), self.x_hrz.value

    def _mpc_vertical(self, cur_z, cur_vz, wp, target_vel, x_pred=None, y_pred=None, apply_cone=False):
        use_cone = (apply_cone and self.cbf_cone_enabled and x_pred is not None and y_pred is not None)

        self.p_cur_vrt.value = np.array([cur_z, cur_vz])
        self.p_xref_vrt.value = np.array([wp[2], 0.0])
        self.p_z_plat.value = wp[2]

        if use_cone:
            ex = np.asarray(x_pred) - (wp[0] + target_vel[0] * np.arange(self.N + 1) * self.dt)
            ey = np.asarray(y_pred) - (wp[1] + target_vel[1] * np.arange(self.N + 1) * self.dt)
            r = self.alpha_cone * np.maximum(0, np.sqrt(ex ** 2 + ey ** 2) - self.r_base)
            self.p_r_cone.value = np.minimum(r, self.z_cut)
        else:
            # Vincolo CBF disattivato matematicamente
            self.p_r_cone.value = np.full(self.N + 1, -1000.0)

        self.prob_vrt.solve(solver=cp.OSQP, warm_start=True, max_iter=4000, eps_abs=1e-4, eps_rel=1e-4)
        
        if self.u_vrt[0, 0].value is None:
            return 0.0
        return float(self.u_vrt[0, 0].value)