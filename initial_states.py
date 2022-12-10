"""
initial_states.py: stores all the ICs to be used in the MF PD Algorithm
                   to estimate S and W matrices
    - Author: Vishnu Vijay
    - Created: 11/25/22
"""

from mav_state import MAV_State

alt_opt = [0, 5, 10]
phi_opt = [0]
theta_opt = [0, 10]
psi_opt = [-10, 0, 10]
p_opt = [0]
q_opt = [0]
r_opt = [0]
Va_opt = [18, 20, 25]

IC_list = []

for o1 in range(len(alt_opt)):
    for o2 in range(len(phi_opt)):
        for o3 in range(len(theta_opt)):
            for o4 in range(len(psi_opt)):
                for o5 in range(len(p_opt)):
                    for o6 in range(len(q_opt)):
                        for o7 in range(len(r_opt)):
                            for o8 in range(len(Va_opt)):
                                ms = MAV_State(alt_opt[o1],
                                               phi_opt[o2] * 3.14 / 180,
                                               theta_opt[o3] * 3.14 / 180,
                                               psi_opt[o4] * 3.14 / 180,
                                               p_opt[o5] * 3.14 / 180,
                                               q_opt[o6] * 3.14 / 180,
                                               r_opt[o7] * 3.14 / 180,
                                               Va_opt[o8])
                                IC_list.append(ms)

num_ICs = len(IC_list)