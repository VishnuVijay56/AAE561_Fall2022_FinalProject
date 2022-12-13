"""
AAE561_project_functions.py: implementing primal-dual algorithms
                             model-known & model-free
    - Author: Vishnu Vijay
    - Created: 11/12/22
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import solve, MatrixSymbol, Matrix
import traceback
from scipy.signal import place_poles, cont2discrete
import time
from copy import copy

from mav import MAV
from mav_dynamics import MAV_Dynamics
from trim import compute_trim
from compute_models import compute_model, euler_state
from wind_simulation import WindSimulation
from signals import Signals
from helper import EulerToQuaternion, QuaternionToRotationMatrix

from mav_state import MAV_State
from delta_state import Delta_State
from autopilot_cmds import AutopilotCmds

import model_coef as M
import MB_feedback_gains as MBF
import MF_feedback_gains as MFF
import initial_states as ICs



## Write gains passed to function to feedback_gains.py ##
def write_gains(case, F_opt_lon, F_opt_lat):
    if (case == 1):
        file = open('MB_feedback_gains.py', 'w')
        file.write('import numpy as np\n')
        file.write('F_opt_lon = np.array([\n    '
                    '[%f, %f, %f, %f, %f],\n    '
                    '[%f, %f, %f, %f, %f]])\n' %
            (F_opt_lon[0][0], F_opt_lon[0][1],
            F_opt_lon[0][2], F_opt_lon[0][3],
            F_opt_lon[0][4], F_opt_lon[1][0],
            F_opt_lon[1][1], F_opt_lon[1][2],
            F_opt_lon[1][3], F_opt_lon[1][4],))
        file.write('F_opt_lat = np.array([\n    '
                    '[%f, %f, %f, %f, %f],\n    '
                    '[%f, %f, %f, %f, %f]])\n' %
            (F_opt_lat[0][0], F_opt_lat[0][1],
            F_opt_lat[0][2], F_opt_lat[0][3],
            F_opt_lat[0][4], F_opt_lat[1][0],
            F_opt_lat[1][1], F_opt_lat[1][2],
            F_opt_lat[1][3], F_opt_lat[1][4],))
        file.close()
    else:
        file = open('MF_feedback_gains.py', 'w')
        file.write('import numpy as np\n')
        file.write('F_lon = np.array([\n    '
                    '[%f, %f, %f, %f, %f],\n    '
                    '[%f, %f, %f, %f, %f]])\n' %
            (F_opt_lon[0][0], F_opt_lon[0][1],
            F_opt_lon[0][2], F_opt_lon[0][3],
            F_opt_lon[0][4], F_opt_lon[1][0],
            F_opt_lon[1][1], F_opt_lon[1][2],
            F_opt_lon[1][3], F_opt_lon[1][4],))
        file.write('F_lat = np.array([\n    '
                    '[%f, %f, %f, %f, %f],\n    '
                    '[%f, %f, %f, %f, %f]])\n' %
            (F_opt_lat[0][0], F_opt_lat[0][1],
            F_opt_lat[0][2], F_opt_lat[0][3],
            F_opt_lat[0][4], F_opt_lat[1][0],
            F_opt_lat[1][1], F_opt_lat[1][2],
            F_opt_lat[1][3], F_opt_lat[1][4],))
        file.close()


## Generate mav state time history ##
def run_sim(case, initial_state, display_graphs):
    # Create instance of MAV_Dynamics
    Ts = 0.01
    mav_dynamics = MAV_Dynamics(time_step=Ts, init_state=initial_state) # time step in seconds
    mav_state = mav_dynamics.mav_state
    # Create instance of MAV object using MAV_State object
    fullscreen = False
    view_sim = False
    this_mav = MAV(mav_state, fullscreen, view_sim)
    # Create instance of wind simulation
    wind_sim = WindSimulation(Ts)
    # Autopilot message
    commands = AutopilotCmds()
    Va_command = Signals(dc_offset=25.0,
                        amplitude=3.0,
                        start_time=2.0,
                        frequency=0.01)
    altitude_command = Signals(dc_offset=0.0,
                            amplitude=15.0,
                            start_time=0.0,
                            frequency=0.02)
    course_command = Signals(dc_offset=np.radians(0),
                            amplitude=np.radians(45),
                            start_time=5.0,
                            frequency=0.015)


    # # Find trim state
    # Va = 25
    # gamma_deg = 0
    # gamma_rad = gamma_deg * np.pi/180
    # trim_state, trim_input = compute_trim(mav_dynamics, Va, gamma_rad)
    # mav_dynamics.internal_state = trim_state
    # delta_trim = trim_input
    # #delta_trim.print()

    # # Compute state space model linearized about trim
    # compute_model(mav_dynamics, trim_state, trim_input)

    # Create instance of autopilot
    from autopilot_PD_LQR import Autopilot

    if (case == 1):
        F_lon = MBF.F_opt_lon
        F_lat = MBF.F_opt_lat
    else:
        F_lon = MFF.F_lon
        F_lat = MFF.F_lat

    autopilot = Autopilot(Ts, F_lon, F_lat)

    # Run Simulation
    curr_time = 0
    end_time = 20 # seconds
    sim_real_time = False

    # # Print Control Parameters
    # import control_parameters
    # control_parameters.print_coefficients()

    extra_elem = 0

    steps = int(end_time / Ts) + extra_elem
    lon_state_history = np.zeros((7, steps))
    lat_state_history = np.zeros((7, steps))

    if (display_graphs):
        time_arr = np.zeros(int(end_time / Ts) + extra_elem)
        
        north_history = np.zeros(int(end_time / Ts) + extra_elem)
        east_history = np.zeros(int(end_time / Ts) + extra_elem)

        alt_history = np.zeros(int(end_time / Ts) + extra_elem)
        alt_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
        
        airspeed_history = np.zeros(int(end_time / Ts) + extra_elem)
        airspeed_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
        
        phi_history = np.zeros(int(end_time / Ts) + extra_elem)
        theta_history = np.zeros(int(end_time / Ts) + extra_elem)
        psi_history = np.zeros(int(end_time / Ts) + extra_elem)

        chi_history = np.zeros(int(end_time / Ts) + extra_elem)
        chi_cmd_history = np.zeros(int(end_time / Ts) + extra_elem)
        
        d_e_history = np.zeros(int(end_time / Ts) + extra_elem)
        d_a_history = np.zeros(int(end_time / Ts) + extra_elem)
        d_r_history = np.zeros(int(end_time / Ts) + extra_elem)
        d_t_history = np.zeros(int(end_time / Ts) + extra_elem)
        
    
    ind = 0
    while (curr_time <= end_time):
        step_start = time.time()
        #print("\nTime: " + str(round(curr_time, 2)) + " ", end=" -> \n")

        # autopilot commands
        commands.airspeed_command = Va_command.square(curr_time)
        commands.course_command = course_command.square(curr_time)
        commands.altitude_command = altitude_command.square(curr_time)
        
        # autopilot
        estimated_state = mav_dynamics.mav_state #this is the actual mav state
        delta, commanded_state = autopilot.update(case, commands, estimated_state)        
        
        # wind sim
        wind_steady_gust = wind_sim.update() # np.zeros((6,1)) #

        # Update MAV dynamic state
        mav_dynamics.iterate(delta, wind_steady_gust)

        # Update MAV mesh for viewing
        this_mav.set_mav_state(mav_dynamics.mav_state)
        this_mav.update_mav_state()
        if (view_sim):
            this_mav.update_render()
        
        # DEBUGGING - Print Vehicle's state
        if (display_graphs):
            time_arr[ind] = curr_time
            
            north_history[ind] = estimated_state.north
            east_history[ind] = estimated_state.east

            alt_history[ind] = estimated_state.altitude
            alt_cmd_history[ind] = commands.altitude_command

            airspeed_history[ind] = estimated_state.Va
            airspeed_cmd_history[ind] = commands.airspeed_command

            chi_history[ind] = estimated_state.chi * 180 / np.pi
            chi_cmd_history[ind] = commands.course_command * 180 / np.pi
            
            phi_history[ind] = estimated_state.phi * 180 / np.pi
            theta_history[ind] = estimated_state.theta * 180 / np.pi
            psi_history[ind] = estimated_state.psi * 180 / np.pi

            d_e_history[ind] = delta.elevator_deflection * 180 / np.pi
            d_a_history[ind] = delta.aileron_deflection * 180 / np.pi
            d_r_history[ind] = delta.rudder_deflection * 180 / np.pi
            d_t_history[ind] = delta.throttle_level


        #mav_dynamics.mav_state.print()
        #commands.print()
        #delta.print()
        #print("Wind: ", mav_dynamics.wind_i.T)
        #print("COMMANDED STATE:", end=" -> ")
        #commanded_state.print()

        # # Wait for 5 seconds before continuing
        # if (ind == 0):
        #     time.sleep(5.)

        vk_lon, vk_lat = get_aug_states(estimated_state, delta, commands)
        lon_state_history[:, ind] = vk_lon[:, 0]
        lat_state_history[:, ind] = vk_lat[:, 0]

        # Update time
        step_end = time.time()
        if ((sim_real_time) and ((step_end - step_start) < mav_dynamics.time_step)):
            time.sleep(step_end - step_start)
        curr_time += Ts
        ind += 1


    if (display_graphs):
        # Main State tracker
        fig1, axes = plt.subplots(1, 3)
        
        axes[0].plot(time_arr, alt_history)
        axes[0].plot(time_arr, alt_cmd_history)
        axes[0].legend(["True", "Command"])
        axes[0].set_title("ALTITUDE")
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Altitude (meters)")

        axes[1].plot(time_arr, airspeed_history)
        axes[1].plot(time_arr, airspeed_cmd_history)
        axes[1].legend(["True", "Command"])
        axes[1].set_title("Va")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Airspeed (meters/second)")

        axes[2].plot(time_arr, chi_history)
        axes[2].plot(time_arr, chi_cmd_history)
        axes[2].legend(["True", "Command"])
        axes[2].set_title("CHI")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("Course Heading (degrees)")

        # Inputs
        fig2, axes2 = plt.subplots(2,2)

        axes2[0,0].plot(time_arr, d_e_history)
        axes2[0,0].set_title("ELEVATOR")
        axes2[0,0].set_xlabel("Time (seconds)")
        axes2[0,0].set_ylabel("Deflection (degrees)")

        axes2[1,0].plot(time_arr, d_a_history)
        axes2[1,0].set_title("AILERON")
        axes2[1,0].set_xlabel("Time (seconds)")
        axes2[1,0].set_ylabel("Deflection (degrees)")

        axes2[0,1].plot(time_arr, d_t_history)
        axes2[0,1].set_title("THROTTLE")
        axes2[0,1].set_xlabel("Time (seconds)")
        axes2[0,1].set_ylabel("Level (percent)")

        axes2[1,1].plot(time_arr, d_r_history)
        axes2[1,1].set_title("RUDDER")
        axes2[1,1].set_xlabel("Time (seconds)")
        axes2[1,1].set_ylabel("Deflection (degrees)")


        # Plane tracking
        fig2 = plt.figure()

        ax = fig2.add_subplot(2, 2, (1,4), projection='3d')
        ax.plot3D(east_history, north_history, alt_history)
        ax.set_title("MAV POSITION TRACK")
        ax.set_xlabel("East Position (meters)")
        ax.set_ylabel("North Position (meters)")
        ax.set_zlabel("Altitude (meters)")

        # Show plots
        plt.show()
    
    return lon_state_history, lat_state_history


## Create augmented state vector v_k given x_k and u_k ##
def get_aug_states(state, input, cmd):
    # Longitudinal System
    x_lon = state.get_lon_state(cmd)
    u_lon = input.get_lon_state()
    v_lon = np.concatenate((x_lon, u_lon), axis=0)

    # Latitudinal System
    x_lat = state.get_lat_state(cmd)
    u_lat = input.get_lat_state()
    v_lat = np.concatenate((x_lat, u_lat), axis=0)

    # Return
    return v_lon, v_lat


## Check if spectral radius of A is less than 1 ##
def check_stability(A):
    eigvals, eigvecs = np.linalg.eig(A)

    for eig in eigvals:
        if (abs(eig) >= 1):
            return False
    
    return True


## Compute optimal state gain matrix F given A and B ##
def pd_model_known_algo(F0, eps, A, B, Q, R, disp_conv):
    
    s = 0
    F_list = [F0]
    P_list = []
    Q_size = Q[0].size
    R_size = R[0].size

    n = A[0].size
    m = B[0].size

    Lambda = np.concatenate((np.concatenate((Q, np.zeros((Q_size, R_size))), axis = 1),
                             np.concatenate((np.zeros((R_size, Q_size)), R), axis = 1)), axis = 0)
    
    stop_flag = False
    while (not stop_flag):
        print("\nITERATION NUMBER: ", s)
        F_t = F_list[s]

        # Primal Update

        A_F = np.concatenate((np.concatenate((A, B), axis = 1),
                              np.concatenate((F_t @ A, F_t @ B), axis = 1)), axis = 0)

        # Dual Update
        
        P = Matrix(MatrixSymbol('P', (n+m), (n+m)))

        # P_21 = P_12^T
        for i in range(n):
            for j in range(n, n+m):
                P_ij = P[i, j]
                P[j, i] = P_ij

        expr = A_F.T @ P @ A_F + Lambda - P

        P_t_dict = solve(expr, P)

        P_t = np.zeros((n+m, n+m))
        for i in range(n+m):
            for j in range(n+m):
                try:
                    P_t[i,j] = P_t_dict.get(P[i,j])
                except Exception:
                    print("i,j: ", i, ", ", j)
                    traceback.print_exc()
                    exit()

        # P_21 = P_12^T
        for i in range(n):
            for j in range(n, n+m):
                P_ij = P_t[i, j]
                P_t[j, i] = P_ij

        P_list.append(P_t)

        P_22 = P_t[n:(n+m), n:(n+m)]
        P_11 = P_t[0:n, 0:n]
        P_12 = P_t[0:n, n:(n+m)]

        F_t1 = -np.linalg.inv(P_22) @ P_12.T

        # print(" -> State Gain")
        # print(F_t1)

        F_list.append(F_t1)
        s = s + 1

        norm_change = np.linalg.norm(F_list[-2] - F_list[-1])
        print(" -> norm(diff) = ", norm_change)
        stop_flag = (norm_change <= eps)

    # Display how F_t converges to optimal F*
    if (disp_conv):
        visualize_convergence(F_list)
    
    # Return optimal F feedback gain
    return F_list[-1]


## Compute optimal state gain matrices F for MAV system ##
def imp_MK_algo(eps, disp_conv, case):
    ## Get initial gains ##

    F0_lon, F0_lat = get_stabilizing_gains()
    
    ## Longitudinal System ##

    # Q and R
    q_u = 1e1
    q_w = 1e1
    q_q = 1e-2
    q_theta = 1e-1
    q_h = 1e3
    Q_lon = np.diag([q_u, q_w, q_q, q_theta, q_h])
    r_e = 1e0
    r_t = 1e0
    R_lon = np.diag([r_e, r_t])

    # Calc optimal gain
    A_lon, B_lon = discretize_AB(M.A_lon, M.B_lon)
    F_opt_lon = pd_model_known_algo(F0_lon, eps, A_lon, B_lon, Q_lon, R_lon, disp_conv)
    print("\nF_lon: \n", F_opt_lon)

    ## Latitudinal System ##

    # Q and R
    q_v = 1e-1
    q_p = 1e0
    q_r = 1e-1
    q_phi = 1e0
    q_chi = 1e1
    Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi])
    r_a = 1e1
    r_r = 1e0
    R_lat = np.diag([r_a, r_r])

    # Calc optimal gain
    A_lat, B_lat = discretize_AB(M.A_lat, M.B_lat)
    F_opt_lat = pd_model_known_algo(F0_lat, eps, A_lat, B_lat, Q_lat, R_lat, disp_conv)
    print("\nF_lat: \n", F_opt_lat)

    ## Write to file ##
    write_gains(1, F_opt_lon, F_opt_lat)


## Compute optimal state gain matrix F without model ##
def pd_model_free_algo(F_list, P_list, Q, R, gamma, sys):
    
    n = Q[0].size
    m = R[0].size

    Lambda = np.concatenate((np.concatenate((Q, np.zeros((n, m))), axis = 1),
                             np.concatenate((np.zeros((m, n)), R), axis = 1)), axis = 0)

    # Primal Update
    S_tilde = S_approx(sys)
    W_tilde = W_approx(sys)

    # Dual Update
    P = Matrix(MatrixSymbol('P', (n+m), (n+m)))

    # P_21 = P_12^T
    for i in range(n):
        for j in range(n, n+m):
            P_ij = P[i, j]
            P[j, i] = P_ij

    expr = S_tilde.T @ P @ S_tilde - gamma * W_tilde.T @ P @ W_tilde - S_tilde.T @ Lambda @ S_tilde

    P_t_dict = solve(expr, P, dict=True)

    P_t = np.zeros((n+m, n+m))
    for i in range(n+m):
        for j in range(n+m):
            try:
                P_t[i,j] = P_t_dict[0].get(P[i,j])
            except Exception:
                print("\n\n(i, j): (", i, ", ", j, ")")
                print("\nEXPRESSION BELOW")
                print(expr)
                # print("\nMATRIX BELOW")
                # print(P)
                print("\nDICTIONARY BELOW")
                print(P_t_dict)
                traceback.print_exc()
                exit()

    # P_21 = P_12^T
    for i in range(n):
        for j in range(n, n+m):
            P_ij = P_t[i, j]
            P_t[j, i] = P_ij

    P_list.append(P_t)

    P_22 = P_t[n:(n+m), n:(n+m)]
    P_11 = P_t[0:n, 0:n]
    P_12 = P_t[0:n, n:(n+m)]

    # Primal Update

    F_t1 = -np.linalg.inv(P_22) @ P_12.T

    # rows, cols = F_t1.shape
    
    # # for i in range(rows):
    # #     for j in range(cols):
    #         if (F_t1[i, j] < -200):
    #             F_t1[i, j] = -200
    #         elif (F_t1[i, j] > 200):
    #             F_t1[i, j] = 200

    # print(" -> State Gain")
    # print(F_t1)

    F_list.append(F_t1)
    
    return F_list, P_list


## Compute optimal state gain matrices F for MAV system ##
def imp_MF_algo(eps, disp_conv):

    # Q and R matrices - LON
    q_u = 1e1
    q_w = 1e1
    q_q = 1e-2
    q_theta = 1e-1
    q_h = 1e3
    Q_lon = np.diag([q_u, q_w, q_q, q_theta, q_h])
    r_e = 1e0
    r_t = 1e0
    R_lon = np.diag([r_e, r_t])

    # Q and R matrices - LAT
    q_v = 1e-1
    q_p = 1e0
    q_r = 1e-1
    q_phi = 1e0
    q_chi = 1e1
    Q_lat = np.diag([q_v, q_p, q_r, q_phi, q_chi])
    r_a = 1e1
    r_r = 1e0
    R_lat = np.diag([r_a, r_r])

    # Find initial gains
    #F0_lon, F0_lat = get_near_optimal_gain()
    F0_lon, F0_lat = get_stabilizing_gains()
    write_gains(2, F0_lon, F0_lat)

    # F_list and P_list
    F_lon_list = [F0_lon]
    F_lat_list = [F0_lat]
    P_lon_list = []
    P_lat_list = []

    # Discretize
    A_lon, B_lon = discretize_AB(M.A_lon, M.B_lon)
    A_lat, B_lat = discretize_AB(M.A_lat, M.B_lat)

    # Loop!

    stop_flag = False
    s = 0
    while (not stop_flag):
        print("\nITERATION: ", s)
        from autopilot_PD_LQR import Autopilot
        
        stabilizability_lon = check_stability(A_lon + B_lon @ F0_lon)
        print("\nLongitudinal System is Stabilizable: ", stabilizability_lon)
        stabilizability_lat = check_stability(A_lat + B_lat @ F0_lat)
        print("\nLatitude System is Stabilizable: ", stabilizability_lat)

        autopilot = Autopilot(M.Ts, F_lon_list[-1], F_lat_list[-1])

        # Gather Data
        for i in range(ICs.num_ICs):
            IC = copy(ICs.IC_list[i])
            try:
                #lon_state_history, lat_state_history = run_sim(2, IC, False)
                lon_state_history, lat_state_history = linear_sys_sim(10, M.Ts, autopilot, IC)
            except Exception:
                print("\n\n")
                traceback.print_exc()
                print("\n\nEXCEPTION at IC #", i, ": ")
                IC.print()
                exit()
            
            file_name = ".\data_folder\data" + str(i)
            np.savez(file_name, lon_state_history=lon_state_history, lat_state_history=lat_state_history)
        
        # Compute guess of F
        print("lon")
        F_lon_list, P_lon_list = pd_model_free_algo(F_lon_list, P_lon_list, Q_lon, R_lon, 1, 1)
        norm_change_lon = np.linalg.norm(F_lon_list[-2] - F_lon_list[-1])
        print("\n -> norm(diff)_lon = ", norm_change_lon)
        
        print("lat")
        F_lat_list, P_lat_list = pd_model_free_algo(F_lat_list, P_lat_list, Q_lat, R_lat, 1, 2)
        norm_change_lat = np.linalg.norm(F_lat_list[-2] - F_lat_list[-1])
        print(" -> norm(diff)_lat = ", norm_change_lat)

        # Write Gains to File
        write_gains(2, F_lon_list[-1], F_lat_list[-1])

        # Check convergence
        stop_flag = ((norm_change_lon <= eps) and (norm_change_lat <= eps))

        s = s + 1
    
    F_opt_lon = F_lon_list[-1]
    F_opt_lat = F_lat_list[-1]


    return F_opt_lon, F_opt_lat


## Discretizes the continuous time matrices A and B ##
def discretize_AB(A, B):
    n = A[0].size
    m = B[0].size
    C = np.zeros((1, n))
    D = np.zeros((1, m))
    dsys = cont2discrete((A, B, C, D), M.Ts, method='bilinear')
    Ad, Bd, *idc = dsys

    return Ad, Bd


## Returns a set of stabilizing gains for the lon and lat systems ##        
def get_stabilizing_gains():
    # Longitudinal System
    
    #Discretize
    A_lon, B_lon = discretize_AB(M.A_lon, M.B_lon)
    #Place poles
    des_poles_lon = np.array([complex((1/2), (1/2)),
                              complex((1/2), -1*(1/2)),
                              complex(0, 5/6),
                              complex(0, -5/6),
                              5/6])
    F0_lon = -1*place_poles(A_lon, B_lon, des_poles_lon).gain_matrix
    #Check stability
    stabilizability_lon = check_stability(A_lon + B_lon @ F0_lon)
    print("\nLongitudinal System is Stabilizable: ", stabilizability_lon)
    

    # Latitudinal System

    #Discretize
    A_lat, B_lat = discretize_AB(M.A_lat, M.B_lat)
    #Place poles
    des_poles_lat = np.array([complex((-1/2), (1/2)),
                              complex((-1/2), -1*(1/2)),
                              complex(0, 5/6),
                              complex(0, -5/6),
                              5/6])
    F0_lat = -1*place_poles(A_lat, B_lat, des_poles_lat).gain_matrix
    #Check stability
    stabilizability_lat = check_stability(A_lat + B_lat @ F0_lat)
    print("\nLatitude System is Stabilizable: ", stabilizability_lat)

    # Return
    return F0_lon, F0_lat


## Returns a set of near optimal gains for lon and lat systems ##
def get_near_optimal_gain():
    F_lon = MBF.F_opt_lon
    F_lat = MBF.F_opt_lat

    rows, cols = F_lon.shape

    for i in range(rows):
        for j in range(cols):
            error = np.random.normal(0, 10)
            F_lon[i, j] = F_lon[i, j] + error
            F_lat[i, j] = F_lat[i, j] + error
    
    # Return
    return F_lon, F_lat


## Returns an approximation of S according to equation in paper ##
def S_approx(sys):
    # Initialization
    S_tilde = np.zeros((7, 7))

    # over all ICs
    for i in range(ICs.num_ICs):
        file_name = ".\data_folder\data" + str(i) + ".npz"
        data = np.load(file_name)

        if (sys == 1):
            vk_history = data['lon_state_history']
        else:
            vk_history = data['lat_state_history']
        
        states, steps = vk_history.shape

        #print("steps: ", steps)

        # over time horizon of sim
        for j in range(steps):
            vk = vk_history[:, j]
            vk = np.reshape(vk, (-1, 1))

            S = vk @ vk.T
            S_tilde = S_tilde + S

    # Normalize over IC cases
    S_tilde = S_tilde / ICs.num_ICs

    # print("\nS_tilde")
    # print(S_tilde)

    return S_tilde


## Returns an approximation of W according to equation in paper ##
def W_approx(sys):
    # Initialization
    W_tilde = np.zeros((7, 7))

    # over all ICs
    for i in range(ICs.num_ICs):
        file_name = ".\data_folder\data" + str(i) + ".npz"
        data = np.load(file_name)

        if (sys == 1):
            vk_history = data['lon_state_history']
        else:
            vk_history = data['lat_state_history']
        
        states, steps = vk_history.shape

        #print("steps: ", steps)

        # over time horizon of sim
        for j in range(steps - 1):
            vk_1 = vk_history[:, (j+1)]
            vk_1 = np.reshape(vk_1, (-1, 1))

            vk = vk_history[:, j]
            vk = np.reshape(vk, (-1, 1))

            W = vk_1 @ vk.T
            W_tilde = W_tilde + W

    # Normalize over IC cases
    W_tilde = W_tilde / ICs.num_ICs

    # print("\nW_tilde")
    # print(W_tilde)

    return W_tilde


## Displays graph of how F converges to F_opt ##
def visualize_convergence(F_list):
    # print("\n\n VISUALIZE CONVERGENCE TIME")
    t = len(F_list)
    # print("length of F_list: ", len(F_list))

    F_abs_diff = []
    for i in range(t):
        F_abs_diff.append(np.linalg.norm(F_list[i] - F_list[-1]))
    
    # print("abs diff list: ")
    # print(F_abs_diff)

    x_axis_ticks = range(t)

    plt.figure()
    plt.plot(range(t), F_abs_diff)
    plt.xticks(x_axis_ticks)
    plt.xlim(0, t)
    plt.title("Convergence of Feedback Gain F_t")
    plt.xlabel("Iteration number t")
    plt.ylabel("||F_t - F*||")
    plt.grid(True)



## Generate time history of system using linear system assumption ##
def linear_sys_sim(t_end, Ts, autopilot, init_state):
    # Gains
    F_lon = autopilot.F_lon
    F_lat = autopilot.F_lat

    # Extract initial state
    e_quat = EulerToQuaternion(init_state.phi, init_state.theta, init_state.psi)
    pdot_d = -1*init_state.Va * np.sin(init_state.theta)
    pdot_e = init_state.Va * np.cos(init_state.theta) * np.sin(init_state.psi)
    pdot_n = init_state.Va * np.cos(init_state.theta) * np.cos(init_state.psi)
    tmp = QuaternionToRotationMatrix(e_quat).T @ np.array([pdot_n, pdot_e, pdot_d])

    # Trim state
    x_euler = euler_state(M.x_trim)
    u_star = x_euler.item(3)
    v_star = x_euler.item(4)
    w_star = x_euler.item(5)
    phi_star = x_euler.item(6)
    theta_star = x_euler.item(7)
    psi_star = x_euler.item(8)
    
    x_lon = np.array([[tmp.item(0) - u_star],
                       [tmp.item(2) - w_star],
                       [init_state.q],
                       [init_state.theta - theta_star],
                       [init_state.altitude]])
    
    x_lat = np.array([[tmp.item(1) - v_star],
                       [init_state.p],
                       [init_state.r],
                       [init_state.phi - phi_star],
                       [init_state.psi - psi_star]])

    
    # Initialize state history arrays
    extra_elem = 1
    steps = int(t_end / Ts) + extra_elem
    lon_state_history = np.zeros((7, steps))
    lat_state_history = np.zeros((7, steps))

    # Discretize
    A_lon, B_lon = discretize_AB(M.A_lon, M.B_lon)
    A_lat, B_lat = discretize_AB(M.A_lat, M.B_lat)

    # Augmented System
    AF_lon = np.concatenate((np.concatenate((A_lon, B_lon), axis=1), 
                             np.concatenate((F_lon @ A_lon, F_lon @ B_lon), axis=1)), axis=0)
    AF_lat = np.concatenate((np.concatenate((A_lat, B_lat), axis=1), 
                             np.concatenate((F_lat @ A_lat, F_lat @ B_lat), axis=1)), axis=0)

    for k in range(steps):
        v_lon = np.concatenate((x_lon, F_lon @ x_lon), axis=0)
        v_lat = np.concatenate((x_lat, F_lat @ x_lat), axis=0)

        lon_state_history[:, k] = v_lon[:, 0]
        lat_state_history[:, k] = v_lat[:, 0]

        #x_lon, x_lat, v_lon, v_lat = f_LTI(k, x_lon, x_lat, AF_lon, AF_lat)

        v_lon = AF_lon @ v_lon
        v_lat = AF_lat @ v_lat

    lon_state_history[:, k] = v_lon[:, 0]
    lat_state_history[:, k] = v_lat[:, 0]

    return lon_state_history, lat_state_history


## Compute x_k+1 based on current state x_k and A and B matrices
def f_LTI(k, x_lon, x_lat, A_lon, B_lon, A_lat, B_lat, commands, 
            Va_command, altitude_command, course_command, autopilot):
    # Extract
    # x_lon = x[0:5] # u, w, q, theta, h
    # x_lat = x[5:11] # v, p, r, phi, psi
    t = k * M.Ts

    # Calculate necessary states
    ms = MAV_State()
    ms.Va = np.sqrt(x_lon[0]**2 + x_lat[0]**2 + x_lon[1]**2)
    ms.alpha = np.arctan2(x_lon[1], x_lon[0])
    ms.beta = np.arcsin(x_lat[0] / ms.Va)

    ms.altitude = x_lon[4]
    ms.q = x_lon[2]
    ms.theta = x_lon[3]

    ms.p = x_lat[1]
    ms.r = x_lat[2]
    ms.phi = x_lat[3]
    ms.psi = x_lat[4]
    ms.chi = ms.phi
    
    
    # "autopilot"
    delta, commanded_state = autopilot.update(2, commands, ms)

    u_lon = delta.get_lon_state()
    u_lat = delta.get_lat_state()

    # Dynamics
    x_1_lon = A_lon @ x_lon + B_lon @ u_lon
    x_1_lat = A_lat @ x_lat + B_lat @ u_lat
    
    # Augmented states
    v_lon = np.concatenate((x_lon, u_lon), axis=0)
    v_lat = np.concatenate((x_lat, u_lat), axis=0)

    # Return
    return x_1_lon, x_1_lat, v_lon, v_lat
