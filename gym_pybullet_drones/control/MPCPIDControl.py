import numpy as np
import cvxpy as cp
import math
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from scipy.spatial.transform import Rotation
import pybullet as p

class MPCPIDControl(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float=9.8, dt=0.02):
        super().__init__(drone_model=drone_model, g=g)
        self.dt = dt
        self.N = 20  # Orizzonte ridotto per mantenere i 240Hz
        
        # Inizializza matrici modello [cite: 1121-1125]
        self.A_hrz = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B_hrz = np.array([[0, 0], [0, 0], [0, g * dt], [-g * dt, 0]])
        self.A_vrt = np.array([[1, dt], [0, 1]])
        self.B_vrt = np.array([[0], [dt]])
        
        # Carica mixer e altre costanti esistenti
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.MIXER_MATRIX = np.array([[-.5, -.5, -1], [-.5, .5, 1], [.5, .5, -1], [.5, -.5, 1]])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.g = [0.0, 0.0, -9.81]
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        
        self.last_mpc_thrust = 0.0
        self.last_mpc_euler = np.zeros(3)
        self.MPC_FREQ_DIVIDER = 24 # Se Pybullet va a 240Hz, 240/12 = 20Hz per l'MPC
    
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
                       a_xy_lim = 0.043
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        
        # === INIZIO DECIMATION MPC ===
        # Calcoliamo l'MPC solo ogni 12 step (es. 20 volte al secondo)
        if self.control_counter % self.MPC_FREQ_DIVIDER == 0 or self.control_counter == 1:
            thrust, computed_target_rpy, pos_e = self._dslMPCPositionControl(
                control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel, a_xy_lim
            )
            # Salviamo i risultati per i frame successivi
            self.last_mpc_thrust = thrust
            self.last_mpc_euler = computed_target_rpy
            self.pos_e = pos_e # Salviamo l'errore per il return
            
        else:
            # Nei frame intermedi, RICICLIAMO l'ultimo comando calcolato dall'MPC
            thrust = self.last_mpc_thrust
            computed_target_rpy = self.last_mpc_euler
            pos_e = self.pos_e
        # === FINE DECIMATION ===

        # Il PID Attitude invece continua a girare ad OGNI FRAME (240Hz)
        rpm = self._dslPIDAttitudeControl(
            control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates
        )
        
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
        
    def _mpc_horizontal(self, cur_x, cur_y, cur_vx, cur_vy, target_pos, a_xy_lim):
        x = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        cost = 0
       
        Q = np.diag([24.0, 24.0, 6.0, 6.0]) # Penalizza molto l'errore di posizione, meno quello di velocità
        R = np.diag([1.0, 1.0])
        
        constraints = [x[:, 0] == [cur_x, cur_y, cur_vx, cur_vy]]
        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - [target_pos[k][0], target_pos[k][1], 0, 0], Q)
            cost += cp.quad_form(u[:, k], R)
            constraints += [x[:, k+1] == self.A_hrz @ x[:, k] + self.B_hrz @ u[:, k]]
            #constraints += [cp.abs(u[:, k]) <= 0.044]
            #constraints += [cp.abs(u[:, k]) <= 0.087]
            constraints += [cp.abs(u[:, k]) <= a_xy_lim]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        return u[:, 0].value # [phi, theta]

    def _mpc_vertical(self, cur_z, cur_vz, target_pos):
        x = cp.Variable((2, self.N + 1))
        u = cp.Variable((1, self.N))
        
        Q = np.diag([20.0, 5.0]) # Penalizza molto l'errore di posizione, meno quello di velocità
        R = np.diag([1.0])
        
        cost = 0
        constraints = [x[:, 0] == [cur_z, cur_vz]]
        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - [target_pos[k][2], 0], Q)
            cost += cp.quad_form(u[:, k], R)
            constraints += [x[:, k+1] == self.A_vrt @ x[:, k] + self.B_vrt @ u[:, k]]
            #constraints += [cp.abs(u[:, k]) <= 6.0] # Limite az
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        return u[:, 0].value[0] # az

    def _dslMPCPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel, a_xy_lim):
        # 1. Estrarre stati
        cur_x, cur_y, cur_z = cur_pos
        vx, vy, vz = cur_vel
        
        # 2. Ottenere comandi MPC (garantiamo che siano float scalari con .item())
        res_hrz = self._mpc_horizontal(cur_x, cur_y, vx, vy, target_pos, a_xy_lim)
        phi_cmd = float(res_hrz[0])
        theta_cmd = float(res_hrz[1])
        
        az = float(self._mpc_vertical(cur_z, vz, target_pos))
        
        # 3. Termine di accoppiamento (Calcolo sicuro con scalari)
        # Nota: tan e arctan su float puri non creano array NumPy
        theta_corr = math.atan((9.81 / (9.81 + az)) * math.tan(theta_cmd))
        phi_corr = math.atan((math.cos(theta_corr) / math.cos(theta_cmd)) * (9.81 / (9.81 + az)) * math.tan(phi_cmd))
        
        # 4. Calcolo Thrust corretto
        m = 0.027 
        scalar_thrust = m * (9.81 + az) 
        thrust = (math.sqrt(max(0, scalar_thrust) / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # 5. Creazione array di destinazione (tutti float!)
        target_euler = np.array([phi_corr, theta_corr, float(target_rpy[2])])
        
        return thrust, target_euler, target_pos[0][:] - cur_pos
        
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST       
        
    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4//DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4//DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
            
    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4//DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4//DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()      
        
