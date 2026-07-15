"""Hybrid MPC+PID landing script (kinematic reachable-set gating).

The line-of-sight (LOS) guidance produces a single carrot waypoint along the
current path segment (no velocity reference). The controller
(MPCPIDControl) internally decides, via a kinematic backward-reachable-set
gate, whether to drive toward the set with a PD law or to run the MPC that
minimises the position error with zero velocity reference.

Run:
    $ python pid.py
"""
import os
import time
import argparse
from datetime import datetime
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.MPCPIDHYControl import MPCPIDHYControl
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
DEFAULT_DURATION_SEC = 20
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


def LOS_wp(p_actual, p_start, p_end, delta, stop_delta):
    """Single-point LOS carrot with progressive look-ahead contraction.

    The look-ahead distance shrinks as the drone approaches the end of the
    segment, so the carrot collapses onto the final waypoint instead of
    saturating at `delta` ahead. This makes the position error seen by the
    PID decay to zero, so the drone reaches the waypoint-switch threshold
    almost at rest: no residual cruise velocity, no inertia-driven overshoot
    at the switch. Purely geometric braking (no velocity reference).
    """
    p_actual = np.array(p_actual)
    p_start = np.array(p_start)
    p_end = np.array(p_end)

    path_vector = p_end - p_start
    path_length = np.linalg.norm(path_vector)

    if path_length < 1e-6:
        return p_end, True

    u = path_vector / path_length
    v = p_actual - p_start
    s = np.dot(v, u)

    # residual distance along the path
    dist_to_end = path_length - s

    # --- progressive contraction of the look-ahead ---
    # far from the end  -> delta_eff = delta (full look-ahead, cruise)
    # near the end      -> delta_eff -> 0 (carrot collapses on the waypoint)
    # the blending starts one `delta` before the switch threshold, so that at
    # s = path_length - stop_delta the drone is already slowing down.
    brake_len = 2.0 * delta            # length of the deceleration zone
    if dist_to_end <= stop_delta:
        delta_eff = 0.0
    elif dist_to_end >= stop_delta + brake_len:
        delta_eff = delta
    else:
        delta_eff = delta * (dist_to_end - stop_delta) / brake_len

    delta_eff = min(delta_eff, max(0.0, dist_to_end))

    reached_end = False
    if (s + delta_eff) <= 0:
        p_LOS = p_start
    elif (s + delta_eff) >= path_length:
        p_LOS = p_end
        reached_end = True
    else:
        p_LOS = p_start + (s + delta_eff) * u

    return p_LOS, reached_end


############################################################
#### RUN Function ##########################################
############################################################

def run(drone=DEFAULT_DRONES,
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
        colab=DEFAULT_COLAB):

    ####################################################################
    #### Initialize the simulation #####################################
    ####################################################################
    H = .1
    H_STEP = .05
    R = .3

    starting_pos = (R*np.cos((0/6)*2*np.pi+np.pi/2),
                    R*np.sin((0/6)*2*np.pi+np.pi/2)-R,
                    H+0*H_STEP)

    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2),
                           R*np.sin((i/6)*2*np.pi+np.pi/2)-R,
                           H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, i * (np.pi/2)/num_drones]
                          for i in range(num_drones)])

    ####################################################################
    #### Waypoints #####################################################
    ####################################################################
    v_plat = [0.0, 0.0, 0.0]
    w_plat = [0.0, 0.0, 0.0]

    WP_MISSION = np.array([
        starting_pos,                             # IDLE
        [starting_pos[0], starting_pos[1], 1.8],  # RISING
        [3.5, 3.5, 1.8],                          # AIR TARGET
        [3.5, 3.5, 0.15]                          # LANDING
    ])

    NUM_WP = 4
    stop_delta = 0.1
    wp_counters = np.array([int((i*NUM_WP/6) % NUM_WP) for i in range(num_drones)])
    old_wp_id = 0
    states = ["idle" for _ in range(num_drones)]
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
                     user_debug_gui=user_debug_gui)

    PYB_CLIENT = env.getPyBulletClient()
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab)

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [MPCPIDHYControl(drone_model=drone) for _ in range(num_drones)]

    ####################################################################
    #### MAIN LOOP #####################################################
    ####################################################################
    action = np.zeros((num_drones, 4))
    pos_e = np.zeros((num_drones, 3))

    # per-phase look-ahead and tilt authority
    LOS_DELTA = 0.3

    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        #### Step the (optionally moving) landing platform #############
        p.resetBaseVelocity(env.PLATFORM_ID,
                            linearVelocity=v_plat,
                            angularVelocity=w_plat,
                            physicsClientId=PYB_CLIENT)
        plat_pos, plat_quat = p.getBasePositionAndOrientation(
            env.PLATFORM_ID, physicsClientId=PYB_CLIENT)

        WP_MISSION[3] = [plat_pos[0], plat_pos[1], plat_pos[2] + 0.05]

        #### Compute control ###########################################
        for j in range(num_drones):
            actual_pt = obs[j][0:3]

            if states[j] == "idle":
                pos_e_plot = np.zeros(3)
                p_LOS = WP_MISSION[3]
            else:
                pos_e_plot = WP_MISSION[3] - actual_pt

                # per-phase tilt authority (kept from your tuning)
                if wp_counters[j] == 3:      # landing
                    a_xy = 0.17
                    los_delta = 0.15
                else:                        # rising / nav_to_wp
                    a_xy = 0.17
                    los_delta = 0.3

                # classic single-point LOS carrot (no velocity reference)
                p_LOS, reached_end = LOS_wp(actual_pt,
                                            WP_MISSION[old_wp_id],
                                            WP_MISSION[wp_counters[j]],
                                            delta=los_delta,
                                            stop_delta = stop_delta)

                action[j, :], _, _, control_type = ctrl[j].computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=obs[j],
                    target_pos=p_LOS,
                    target_rpy=INIT_RPYS[j, :],
                    target_vel=np.zeros(3),          # zero velocity reference
                    target_rpy_rates=np.zeros(3),
                    a_xy_lim=a_xy)

            pos_e[j] = WP_MISSION[wp_counters[j]] - obs[j][0:3]

            if old_wp_id == 3 and wp_counters[j] == 0:
                WP_MISSION[0] = WP_MISSION[3]
                action[j, :] = np.zeros(4)

        [states, wp_counters, old_wp_id, stop_delta] = update_state(
            pos_e, num_drones, states, wp_counters, old_wp_id, stop_delta)

        #### Log #######################################################
        for j in range(num_drones):
            p_los_arr = np.array(p_LOS)
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([p_los_arr[0:2], INIT_XYZS[j, 2],
                                          INIT_RPYS[j, :], np.zeros(6)]),
                       pos_e=pos_e_plot,
                       p_LOS=p_los_arr[:],
                       control_type = control_type)

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("pid")
    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Hybrid MPC+PID landing script (kinematic reachable-set gating)')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int,
                        help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                        help='Whether to add obstacles (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 30)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is run by a notebook (default: False)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
