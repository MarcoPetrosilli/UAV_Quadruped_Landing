import numpy as np
import cvxpy as cp
import math
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from scipy.spatial.transform import Rotation
import pybullet as p


class MPCPIDHYControlDynamic(BaseControl):
    """Hybrid controller: DSL-PID (reach phase) + MPC (final maneuver).

    Two-mode structure inspired by Persson (KTH, 2019):

      - OUTSIDE the backward-reachable set: the original DSL PID position +
        attitude controller drives the drone toward the maneuver-initiation
        set. Stock gym-pybullet-drones DSLPIDControl law, unchanged (xyz
        handled jointly, no axis separation, no tilt-coupling corrections).

      - INSIDE the reachable set: the MPC takes over and minimises the
        position error with a ZERO velocity reference and a soft terminal
        constraint toward the waypoint.

    The paper's offline polytopic backward-reachable set is replaced by an
    online kinematic feasibility gate (is_in_reachable_set).
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8, dt=0.02):
        super().__init__(drone_model=drone_model, g=g)
        self.dt = dt
        self.N = 80

        # ---- MPC prediction models -----------------------------------------
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

        # ---- Mixer / motor constants ---------------------------------------
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([[-.5, -.5, -1],
                                          [-.5, .5, 1],
                                          [.5, .5, -1],
                                          [.5, -.5, 1]])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([[0, -1, -1],
                                          [+1, 0, 1],
                                          [0, 1, -1],
                                          [-1, 0, 1]])

        # ---- DSL PID gains (stock DSLPIDControl) ---------------------------
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.0, .0, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])

        # ---- MPC weights (position error, zero velocity ref) ---------------
        self.Q_hrz = np.diag([20.0, 20.0, 10.0, 10.0])
        self.R_hrz = np.diag([10.0, 10.0])
        self.Q_vrt = np.diag([30.0, 15.0])
        self.R_vrt = np.diag([15.0])

        # ---- Kinematic reachable-set parameters ----------------------------
        self.a_reach_xy = 1.6     # ~ g*tan(a_xy_lim); tune to widen/shrink set
        self.a_reach_z = 9.0
        self.v_max = 1.0
        self.reach_margin = 0.85

        # ---- Multi-rate decimation + FOH (MPC mode only) -------------------
        self.MPC_FREQ_DIVIDER = 8
        self.last_mpc_thrust = 0.0
        self.last_mpc_euler = np.zeros(3)
        self.prev_mpc_thrust = 0.0
        self.prev_mpc_euler = np.zeros(3)
        self.mpc_step = 0
        self.pos_e = np.zeros(3)
        
        self.mpc_activated = False
        
        _P = np.load("reachable_polytope.npz")
        self.H_ax, self.h_ax = _P["H_ax"], _P["h_ax"]   # horizontal per-axis (2D)
        self.H_vz, self.h_vz = _P["H_vz"], _P["h_vz"]    # vertical (2D)

        self.reset()

    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    # ------------------------------------------------------------------ #
    #  Kinematic backward-reachable-set gate                             #
    # ------------------------------------------------------------------ #
    def is_in_reachable_set(self, cur_pos, cur_vel, target_pos, target_vel=None):
        cur_pos = np.asarray(cur_pos, float)
        cur_vel = np.asarray(cur_vel, float)
        target_pos = np.asarray(target_pos, float)
        v_tgt = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)

        # relative state (drone minus platform)
        e_pos = cur_pos - target_pos
        e_vel = cur_vel - v_tgt

        def inside(H, h, x):
            return bool(np.all(H @ x <= h + 1e-9))

        # horizontal: one 2-D test per axis (state [e, ev])
        in_x = inside(self.H_ax, self.h_ax, np.array([e_pos[0], e_vel[0]]))
        in_y = inside(self.H_ax, self.h_ax, np.array([e_pos[1], e_vel[1]]))
        # vertical
        in_z = inside(self.H_vz, self.h_vz, np.array([e_pos[2], e_vel[2]]))

        return in_x and in_y and in_z

    # ------------------------------------------------------------------ #
    #  Main entry point                                                  #
    # ------------------------------------------------------------------ #
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       a_xy_lim=0.17,
                       final_pos = None,
                       landing = False
                       ):
        self.control_counter += 1
        wp = np.array(target_pos, dtype=float)
        wp_final = np.array(final_pos, dtype=float) if final_pos is not None else wp
        in_set = self.is_in_reachable_set(cur_pos, cur_vel, wp_final, target_vel)

        if (in_set and landing) or self.mpc_activated:
        #if in_set and landing:
            
            self.mpc_activated = True
            # ---------------- MPC MODE ----------------
            run_mpc = (self.control_counter % self.MPC_FREQ_DIVIDER == 0
                       or self.mpc_step == 0)
            if run_mpc:
                self.prev_mpc_thrust = self.last_mpc_thrust
                self.prev_mpc_euler = self.last_mpc_euler.copy()
                self.mpc_step = 1
                thrust_new, euler_new, pos_e = self._mpc_position_control(
                    cur_pos, cur_vel, wp_final, target_rpy, a_xy_lim, target_vel)
                self.last_mpc_thrust = thrust_new
                self.last_mpc_euler = euler_new
                self.pos_e = pos_e
            else:
                pos_e = self.pos_e

            #alpha = self.mpc_step / self.MPC_FREQ_DIVIDER
            #thrust = (1 - alpha) * self.prev_mpc_thrust + alpha * self.last_mpc_thrust
            #computed_target_rpy = ((1 - alpha) * self.prev_mpc_euler
            #                       + alpha * self.last_mpc_euler)
            thrust = self.last_mpc_thrust
            computed_target_rpy = self.last_mpc_euler
            self.mpc_step += 1

            rpm = self._dslPIDAttitudeControl(
                control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates)
            cur_rpy = p.getEulerFromQuaternion(cur_quat)
            return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2], 1
        else:
            # ---------------- DSL PID REACH MODE ----------------
            self.mpc_step = 0  # force fresh MPC solve when we re-enter the set
            thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(
                control_timestep, cur_pos, cur_quat, cur_vel, wp, target_rpy, target_vel)
            rpm = self._dslPIDAttitudeControl(
                control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates)
            cur_rpy = p.getEulerFromQuaternion(cur_quat)
            return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2], 0

    # ------------------------------------------------------------------ #
    #  MPC position control (position error, zero velocity ref)          #
    # ------------------------------------------------------------------ #
    def _mpc_position_control(self, cur_pos, cur_vel, wp, target_rpy, a_xy_lim, target_vel):
        cur_x, cur_y, cur_z = cur_pos
        vx, vy, vz = cur_vel

        phi_cmd, theta_cmd = self._mpc_horizontal(cur_x, cur_y, vx, vy, wp, a_xy_lim, target_vel)
        az = self._mpc_vertical(cur_z, vz, wp)

        theta_corr = math.atan((9.81 / (9.81 + az)) * math.tan(theta_cmd))
        phi_corr = math.atan((math.cos(theta_corr) / math.cos(theta_cmd))
                             * (9.81 / (9.81 + az)) * math.tan(phi_cmd))

        m = 0.027
        scalar_thrust = m * (9.81 + az)
        thrust = (math.sqrt(max(0, scalar_thrust) / (4 * self.KF))
                  - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_euler = np.array([phi_corr, theta_corr, float(target_rpy[2])])
        pos_e = wp - np.array(cur_pos)
        return thrust, target_euler, pos_e

    def _mpc_horizontal(self, cur_x, cur_y, cur_vx, cur_vy, wp, a_xy_lim, target_vel):
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        xref = np.array([wp[0], wp[1], target_vel[0], target_vel[1]])
        cost = 0
        cons = [x[:, 0] == [cur_x, cur_y, cur_vx, cur_vy]]
        for k in range(self.N):
            xref = np.array([wp[0]+target_vel[0]*k*self.dt, wp[1]+target_vel[1]*k*self.dt, target_vel[0], target_vel[1]])
            cost += cp.quad_form(x[:, k] - xref, self.Q_hrz)
            cost += cp.quad_form(u[:, k], self.R_hrz)
            cons += [x[:, k + 1] == self.A_hrz @ x[:, k] + self.B_hrz @ u[:, k]]
            
            #cons += [x[2, k]<=self.v_max]
            #cons += [x[2, k]>=-self.v_max]
            #cons += [x[3, k]<=self.v_max]
            #cons += [x[3, k]>=-self.v_max]
            
            cons += [cp.abs(u[:, k]) <= a_xy_lim]
        cost += 10.0 * cp.quad_form(x[:, self.N] - xref, self.Q_hrz)
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)
        if u[:, 0].value is None:
            return 0.0, 0.0
        return float(u[0, 0].value), float(u[1, 0].value)

    def _mpc_vertical(self, cur_z, cur_vz, wp):
        x = cp.Variable((2, self.N + 1))
        u = cp.Variable((1, self.N))
        xref = np.array([wp[2], 0.0])
        cost = 0
        cons = [x[:, 0] == [cur_z, cur_vz]]
        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - xref, self.Q_vrt)
            cost += cp.quad_form(u[:, k], self.R_vrt)
            cons += [x[:, k + 1] == self.A_vrt @ x[:, k] + self.B_vrt @ u[:, k]]
            
            #cons += [x[1, k]<=self.v_max]
            #cons += [x[1, k]>=-self.v_max]
            
            cons += [cp.abs(u[:, k]) <= 9.0]
        cost += 10.0 * cp.quad_form(x[:, self.N] - xref, self.Q_vrt)
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.OSQP, warm_start=True)
        if u[0, 0].value is None:
            return 0.0
        return float(u[0, 0].value)

    # ------------------------------------------------------------------ #
    #  DSL PID position control (STOCK, unchanged from DSLPIDControl)    #
    # ------------------------------------------------------------------ #
    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat,
                               cur_vel, target_pos, target_rpy, target_vel):
        pos_ctrl_max = 0.4
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos

        if np.linalg.norm(pos_e) > 1e-03:
            normalized_pos_e = pos_e / np.linalg.norm(pos_e)
        else:
            normalized_pos_e = np.zeros(3)
        min_pos_e = min(pos_ctrl_max, np.linalg.norm(pos_e))
        actual_pos_e = min_pos_e * normalized_pos_e

        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        target_thrust = (np.multiply(self.P_COEFF_FOR, actual_pos_e)
                         + np.multiply(self.I_COEFF_FOR, self.integral_pos_e)
                         + np.multiply(self.D_COEFF_FOR, vel_e)
                         + np.array([0, 0, self.GRAVITY]))
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF))
                  - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter,
                  "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e

    # ------------------------------------------------------------------ #
    #  DSL PID attitude control (STOCK, shared by both modes)            #
    # ------------------------------------------------------------------ #
    def _dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat,
                               target_euler, target_rpy_rates):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = (np.dot(target_rotation.transpose(), cur_rotation)
                        - np.dot(cur_rotation.transpose(), target_rotation))
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        target_torques = (- np.multiply(self.P_COEFF_TOR, rot_e)
                          + np.multiply(self.D_COEFF_TOR, rpy_rates_e)
                          + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e))
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
