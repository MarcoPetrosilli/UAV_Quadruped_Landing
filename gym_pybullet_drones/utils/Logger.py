import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str="results",
                 num_drones: int=1,
                 duration_sec: int=0,
                 colab: bool=False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.COLAB = colab
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_drones, 19, duration_sec*self.LOGGING_FREQ_HZ)) #### 16 states: pos_x,
                                                                                      # pos_y,
                                                                                      # pos_z,
                                                                                      # vel_x,
                                                                                      # vel_y,
                                                                                      # vel_z,
                                                                                      # roll,
                                                                                      # pitch,
                                                                                      # yaw,
                                                                                      # ang_vel_x,
                                                                                      # ang_vel_y,
                                                                                      # ang_vel_z,
                                                                                      # rpm0,
                                                                                      # rpm1,
                                                                                      # rpm2,
                                                                                      # rpm3
        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                               # pos_y,
                                                                                                               # pos_z,
                                                                                                               # vel_x, 
                                                                                                               # vel_y,
                                                                                                               # vel_z,
                                                                                                               # roll,
                                                                                                               # pitch,
                                                                                                               # yaw,
                                                                                                               # ang_vel_x,
                                                                                                               # ang_vel_y,
                                                                                                               # ang_vel_z
        self.control_type = np.zeros((1, duration_sec*self.LOGGING_FREQ_HZ))                                                                                                          

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12),
            pos_e=np.zeros(3),
            p_LOS=np.zeros(3),
            v_LOS=np.zeros(3),
            control_type = 0
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 19, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
            self.control_type = np.concatenate((self.control_type, np.zeros((self.NUM_DRONES, 1))), axis=1)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], v_LOS, pos_e, state[16:20], p_LOS])
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1
        
        self.control_type[drone, current_counter] = control_type

    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(os.path.join(self.OUTPUT_FOLDER, "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy"), 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str=""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        csv_dir = os.path.join(self.OUTPUT_FOLDER, "save-flight-"+comment+"-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir+'/')
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        for i in range(self.NUM_DRONES):
            with open(csv_dir+"/x"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0, :]])), delimiter=",")
            with open(csv_dir+"/y"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 1, :]])), delimiter=",")
            with open(csv_dir+"/z"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 2, :]])), delimiter=",")
            ####
            with open(csv_dir+"/r"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 6, :]])), delimiter=",")
            with open(csv_dir+"/p"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 7, :]])), delimiter=",")
            with open(csv_dir+"/ya"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 8, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rr"+str(i)+".csv", 'wb') as out_file:
                rdot = np.hstack([0, (self.states[i, 6, 1:] - self.states[i, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, rdot])), delimiter=",")
            with open(csv_dir+"/pr"+str(i)+".csv", 'wb') as out_file:
                pdot = np.hstack([0, (self.states[i, 7, 1:] - self.states[i, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, pdot])), delimiter=",")
            with open(csv_dir+"/yar"+str(i)+".csv", 'wb') as out_file:
                ydot = np.hstack([0, (self.states[i, 8, 1:] - self.states[i, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, ydot])), delimiter=",")
            ###
            with open(csv_dir+"/vx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3, :]])), delimiter=",")
            with open(csv_dir+"/vy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 4, :]])), delimiter=",")
            with open(csv_dir+"/vz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 5, :]])), delimiter=",")
            ####
            with open(csv_dir+"/wx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9, :]])), delimiter=",")
            with open(csv_dir+"/wy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 10, :]])), delimiter=",")
            with open(csv_dir+"/wz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 11, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rpm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12, :]])), delimiter=",")
            with open(csv_dir+"/rpm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 13, :]])), delimiter=",")
            with open(csv_dir+"/rpm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 14, :]])), delimiter=",")
            with open(csv_dir+"/rpm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 15, :]])), delimiter=",")
            ####
            with open(csv_dir+"/pwm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 12, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 13, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 14, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 15, :] - 4070.3) / 0.2685])), delimiter=",")

    ################################################################################
    
    def plot(self, pwm=False):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('v_LOS_x')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('v_LOS_y')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('v_LOS_z')

        #### Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('e_x')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('e_y')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('e_z')

        #### Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            #rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, self.control_type[j, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('control_type')
        row = 4
        for j in range(self.NUM_DRONES):
            pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, pdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        row = 5
        for j in range(self.NUM_DRONES):
            ydot = np.hstack([0, (self.states[j, 8, 1:] - self.states[j, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, ydot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (10):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                                 frameon=True
                                 )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        #### PLOT 3D DELLA CAROTA LOS VS TRAIETTORIA REALE #########
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        fig2d = plt.figure(figsize=(10, 8))
        ax2d = fig2d.add_subplot(111)

        for j in range(self.NUM_DRONES):
            # Estrai coordinate reali
            act_x = self.states[j, 0, :]
            act_y = self.states[j, 1, :]
            act_z = self.states[j, 2, :]
            
            # Estrai coordinate carota LOS
            los_x = self.states[j, 16, :]
            los_y = self.states[j, 17, :]
            los_z = self.states[j, 18, :]

            # Traiettoria drone
            ax3d.plot(act_x, act_y, act_z, label=f"Drone {j} Trajectory", color='b', linewidth=2)
            # Traiettoria target
            ax3d.plot(los_x, los_y, los_z, label=f"Drone {j} LOS Target Trajectory", color='g', linestyle='--', linewidth=1.5)
            
            step_size = max(1, int(len(act_x) / 50)) 

            ax3d.quiver(act_x[::step_size], act_y[::step_size], act_z[::step_size], 
                        los_x[::step_size] - act_x[::step_size],                    
                        los_y[::step_size] - act_y[::step_size],                    
                        los_z[::step_size] - act_z[::step_size],                    
                        color='r', 
                        alpha=0.6,               
                        arrow_length_ratio=0.15, 
                        linewidth=1.5,
                        label=f"Drone {j} Error Vector" if j == 0 else "" 
                        )
                        
            ax2d.plot(act_x, act_y, label=f"Drone {j} Trajectory", color='b', linewidth=2)
            ax2d.plot(los_x, los_y, label=f"Drone {j} LOS Target Trajectory", color='g', linestyle='--', linewidth=1.5)
            
            ax2d.quiver(act_x[::step_size], act_y[::step_size], 
                        los_x[::step_size] - act_x[::step_size],                    
                        los_y[::step_size] - act_y[::step_size], 
                        angles='xy', scale_units='xy', scale=1,
                        color='r', 
                        alpha=0.6,               
                        width=0.003,
                        label=f"Drone {j} Error Vector" if j == 0 else "" 
                        )

        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('3D Tracking: Actual Position vs LOS Target')
        ax3d.legend()
        
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        ax2d.set_title('2D Top-Down View: XY Tracking')
        ax2d.grid(True)
        ax2d.axis('equal')
        ax2d.legend()
        ############################################################
        if self.COLAB: 
            plt.savefig(os.path.join('results', 'output_figure.png'))
        else:
            plt.show()

    ################################################################################

    def plot_reachable_entry(self, final_pos, target_vel=None,
                             npz_path="reachable_polytope.npz", drone=0,
                             landing_mask=None, tol=1e-6, save=None):
        """Stato-errore sul set controllabile piu esterno, ristretto alla fase
        di landing, con marcatura dell'ingresso nel gate congiunto (set prodotto)."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        d = np.load(npz_path)
        H_ax, h_ax, H_vz, h_vz = d["H_ax"], d["h_ax"], d["H_vz"], d["h_vz"]
        V_ax, V_vz = d["V_ax"], d["V_vz"]

        pos = self.states[drone, 0:3, :]          # <-- righe posizione: verifica layout
        vel = self.states[drone, 3:6, :]          # <-- righe velocita:  verifica layout
        fp  = np.asarray(final_pos, float); fp = fp[:, None] if fp.ndim == 1 else fp
        tv  = (np.zeros_like(pos) if target_vel is None
               else (np.asarray(target_vel, float)[:, None]
                     if np.ndim(target_vel) == 1 else np.asarray(target_vel, float)))
        ep, ev = pos - fp, vel - tv
        tt = self.timestamps[drone]

        if landing_mask is not None:              # <-- SOLO la fase di landing
            m = np.asarray(landing_mask, bool)
            ep, ev, tt = ep[:, m], ev[:, m], tt[m]
        Np = ep.shape[1]

        def inside(H, h, e, v):
            return np.all(H @ np.array([e, v]) <= h + tol)
        in_x = np.array([inside(H_ax, h_ax, ep[0, k], ev[0, k]) for k in range(Np)])
        in_y = np.array([inside(H_ax, h_ax, ep[1, k], ev[1, k]) for k in range(Np)])
        in_z = np.array([inside(H_vz, h_vz, ep[2, k], ev[2, k]) for k in range(Np)])
        in_all = in_x & in_y & in_z
        entry = int(np.argmax(in_all)) if in_all.any() else None

        panels = [("asse X", ep[0], ev[0], V_ax, "e_x [m]", "e_vx [m/s]"),
                  ("asse Y", ep[1], ev[1], V_ax, "e_y [m]", "e_vy [m/s]"),
                  ("asse Z", ep[2], ev[2], V_vz, "e_z [m]", "e_vz [m/s]")]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        for ax, (name, e, v, V, xl, yl) in zip(axes, panels):
            Vc = np.vstack([V, V[0]])
            ax.fill(Vc[:, 0], Vc[:, 1], color="tab:green", alpha=0.10, zorder=0)
            ax.plot(Vc[:, 0], Vc[:, 1], color="tab:green", lw=1.6, zorder=1,
                    label="set controllabile")
            pts  = np.array([e, v]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="plasma", zorder=2)
            lc.set_array(tt[:-1]); lc.set_linewidth(2.0); ax.add_collection(lc)
            ax.scatter(e[0], v[0], c="k", s=55, zorder=4, label="inizio landing")
            if entry is not None:
                ax.scatter(e[entry], v[entry], marker="*", s=280, c="red",
                           edgecolor="k", zorder=5, label=f"gate t={tt[entry]:.2f}s")
            xs = np.r_[Vc[:, 0], e]; ys = np.r_[Vc[:, 1], v]     # autoscale con la traiettoria
            mx, my = 0.1*np.ptp(xs), 0.1*np.ptp(ys)
            ax.set_xlim(xs.min()-mx, xs.max()+mx); ax.set_ylim(ys.min()-my, ys.max()+my)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(name)
            ax.grid(alpha=.3); ax.axhline(0, color='k', lw=.4); ax.axvline(0, color='k', lw=.4)
        # 1. Crea la colorbar (leggermente più staccata dai grafici con pad=0.02)
        fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02).set_label("tempo [s] (landing)")
        
        # 2. Imposta il titolo generale
        msg = f"gate a t={tt[entry]:.2f}s" if entry is not None else "gate mai attivo"
        fig.suptitle(f"Stato-errore sul set controllabile — solo fase landing — {msg}")
        
        # 3. Recupera la legenda solo dal primo asse per evitare le triplicazioni
        handles, labels = axes[0].get_legend_handles_labels()
        
        # 4. Aggiusta i margini globali: 
        # left=0.15 lascia spazio alla legenda, right=0.88 lascia spazio alla colorbar
        fig.subplots_adjust(left=0.15, right=0.88) 
        
        # 5. Disegna la legenda nello spazio vuoto a sinistra
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.01, 0.5), frameon=True, fontsize=9)
        if save: plt.savefig(save, dpi=110, bbox_inches="tight")
        else:    plt.show()
        return entry
        
        
        
        
        
        
        
