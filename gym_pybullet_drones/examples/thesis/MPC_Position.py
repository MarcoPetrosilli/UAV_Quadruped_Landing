"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.MPCPIDControl import MPCPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False




def initialize_state(num_drones, states, wp_counters):   
            
    for j in range(num_drones):
            
        states[j] = "rising"
        wp_counters[j] = 1 
    return states, wp_counters
                                 
def update_state(pos_e, num_drones, states, wp_counters, old_wp_id, stop_delta):

    for j in range(num_drones):

        distance = np.linalg.norm(pos_e[j])
        
        
        if distance < stop_delta:
            if states[j] == "rising":
                states[j] = "nav_to_wp"
                old_wp_id = 1
                wp_counters[j] = 2
                stop_delta = 0.3
            elif states[j] == "nav_to_wp":
                states[j] = "landing"
                old_wp_id = 2
                wp_counters[j] = 3
                stop_delta = 0.1
            else:
            	states[j] = "idle"
            	old_wp_id = 3
            	wp_counters[j] = 0
    return states, wp_counters, old_wp_id, stop_delta


def LOS_wp(p_actual, p_start, p_end, delta, N):
    p_actual = np.array(p_actual)
    p_start = np.array(p_start)
    p_end = np.array(p_end)

    path_vector = p_end - p_start
    path_length = np.linalg.norm(path_vector)

    if path_length < 1e-6:
        p_LOS = [p_end for _ in range(N)]
        return p_LOS, True

    u = path_vector / path_length

    p_LOS = []
    reached_end = False

    v = p_actual - p_start
    s = np.dot(v, u)  # ✅ proiezione calcolata UNA volta, fuori dal loop

    for i in range(N):
        s_lookahead = s + delta * (i + 1)  # ogni punto guarda più avanti

        if s_lookahead <= 0:
            p_LOS.append(p_start.copy())
        elif s_lookahead >= path_length:
            p_LOS.append(p_end.copy())
            reached_end = True
        else:
            p_LOS.append(p_start + s_lookahead * u)  # ✅ formula corretta
            reached_end = False

    return p_LOS, reached_end

############################################################
#### RUN Function ##########################################
############################################################

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
        
############################################################
#### Initialize the simulation #############################
############################################################

    H = .1
    H_STEP = .05
    R = .3
    
    starting_pos = R*np.cos((0/6)*2*np.pi+np.pi/2), R*np.sin((0/6)*2*np.pi+np.pi/2)-R, H+0*H_STEP
    
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

############################################################
#### Initialize waypoints ##################################
############################################################

    v_plat = [0.0, 0.0, 0.0] 
    w_plat = [0.0, 0.0, 0.0]
    
    target_v = [0.0, 0.0, 0.0]
    
    WP_MISSION = np.array([
        starting_pos,                               # IDLE
        [starting_pos[0], starting_pos[1], 1.2],    # RISING
        [3.5, 3.5, 1.8],                            # AIR TARGET
        [3.5, 3.5, 0.15]                            # LANDING
    ])
    
    NUM_WP = 4
    
    stop_delta = 0.3 # hardcoded stop_delta = 0.1 for the landing phase
    
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    
    old_wp_id = 0
    
    
    states = ["idle" for i in range(num_drones)]
    
    [states, wp_counters] = initialize_state(num_drones, states, wp_counters)  
    

    
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )
                        
############################################################
#### PyBullet Client ID - Logger - Controller ##############
############################################################

    PYB_CLIENT = env.getPyBulletClient()

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [MPCPIDControl(drone_model=drone) for i in range(num_drones)]
        
        

############################################################
#### MAIN LOOP #############################################
############################################################


    action = np.zeros((num_drones,4))
    pos_e = np.zeros((num_drones,3))
    
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        
        #### Step the landing platform #############################
        p.resetBaseVelocity(
            env.PLATFORM_ID, 
            linearVelocity=v_plat, 
            angularVelocity=w_plat, 
            physicsClientId=PYB_CLIENT
        )
        
        plat_pos, plat_quat = p.getBasePositionAndOrientation(
            env.PLATFORM_ID, 
            physicsClientId=PYB_CLIENT
        )
        
        WP_MISSION[2] = [plat_pos[0], plat_pos[1], plat_pos[2] + 1.0]
        
        WP_MISSION[3] = [plat_pos[0], plat_pos[1], plat_pos[2] + 0.05] 
         

        #### Compute control for the current way point #############
        for j in range(num_drones):
        
       	
            actual_pt = obs[j][0:3]
            
            #WP_MISSION[0] = actual_pt
            
            WP_MISSION[0] = WP_MISSION[3]
            
            if states[j]=="idle":  
                pos_e_plot = np.zeros(3)
                
            else:
                pos_e_plot = WP_MISSION[3] - actual_pt
                
                if wp_counters[j] == 3 or wp_counters[j] == 2:
                    target_v = v_plat
                else:
                    target_v = [0.0, 0.0, 0.0]
            
            [p_LOS, reached_end] = LOS_wp(actual_pt, WP_MISSION[old_wp_id], WP_MISSION[wp_counters[j]], delta=0.1, N=20)
            
            action[j,:], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    #target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                    target_pos=p_LOS,
                                                                    
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :],
                                                                    target_vel=target_v
                                                                    )
            pos_e[j] = WP_MISSION[wp_counters[j]] - obs[j][0:3]
            if old_wp_id==3 and wp_counters[j]==0:
                action[j,:] = np.zeros(4) 
                
                                                                        
        
        
        [states, wp_counters, old_wp_id, stop_delta] = update_state(pos_e, num_drones, states, wp_counters, old_wp_id, stop_delta)        
           

                
                
                
                
############################################################
#### Log the simulation ####################################
############################################################


        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       #control=np.hstack([WP_MISSION[wp_counters[j],0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)]),
                       control=np.hstack([p_LOS[0][0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)]),
                       pos_e = pos_e_plot,
                       p_LOS = p_LOS[0][:]
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
