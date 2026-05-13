"""
Multistate reaction network search example.
Reactant: pyruvic acid
State: T1
Method: spGFN2-xTB
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import GraphBasedMolecularDynamics
from GraphBasedMolecularDynamics import *

GraphBasedMolecularDynamics.print_level = 1
GraphBasedMolecularDynamics.__min_ediff = 3
GraphBasedMolecularDynamics.__au2kcalmol = 627.5095
GraphBasedMolecularDynamics.__MAX_BARRIER_IN_SELECT_NEXT_REACTANT = 30
GraphBasedMolecularDynamics.__MAX_BARRIER_IN_MD = 30
GraphBasedMolecularDynamics.__MAX_ITERATION = 50

GraphBasedMolecularDynamics.__HALF_STEP_WINDOW = 0.1  # Angs
GraphBasedMolecularDynamics.__DE_THRESHOLD_FILTER = 1  # kcal/mol
GraphBasedMolecularDynamics.__GMD_STOP_AT_FIRST_MINIMUM = False

GraphBasedMolecularDynamics.__BREAKING_NUM = 2
GraphBasedMolecularDynamics.__FORMATION_NUM = 2
GraphBasedMolecularDynamics.__MAX_FRAG = -1
GraphBasedMolecularDynamics.__MAX_RADICAL = 0

# result coordinates
GraphBasedMolecularDynamics.result_coord_filename = 'result_coord.xyz'
# result hash + SPRINT coordinate
GraphBasedMolecularDynamics.result_hash_filename = 'result_hash.log'
# unique hash + SPRINT coordinate
GraphBasedMolecularDynamics.unique_hash_filename = 'unique_hash.log'
# corresponding unique coordinates
GraphBasedMolecularDynamics.unique_coord_filename = 'unique_coord.xyz'
# reaction network file
GraphBasedMolecularDynamics.reaction_net_filename = 'net.log'
# expected Hash
GraphBasedMolecularDynamics.expected_hash_filename = 'expected_hash.log'
# reaction matrix
GraphBasedMolecularDynamics.reaction_matrix_filename = 'reaction_mat.log'
# path energies
GraphBasedMolecularDynamics.path_energy_filename = 'path_energy.log'
# decay strucutre filename
GraphBasedMolecularDynamics.decay_traj_filename = 'decay_coord.xyz'
# job filename head
GraphBasedMolecularDynamics.job_filename_head = 'test_'
GraphBasedMolecularDynamics.opt_result_filename = 'opt.xyz'
GraphBasedMolecularDynamics.log_filename = 'run'
# GraphBasedMolecularDynamics.interface_type = 'OM2MRCI_S1S0'
GraphBasedMolecularDynamics.interface_type = 'xTB_T1S0'


def run_auto_search(copy_filename_list=[]):
    # start_time = time.time()
    cwd = os.getcwd()
    generate_logging(log_filename)
    n_proc = 20
    input_file = r'input.xyz'
    ele, p_list, _ = read_xyz(input_file)
    p = p_list[0]
    atom_index = [i for i in range(len(p.reshape(-1, 3)))]
    state_list = [(3, 0)]
    state_name_list = ['T1']
    for i in range(len(state_list)):
        mult, state = state_list[i]
        os.chdir(cwd)
        # check if reaction net exists
        state_filename = 'state_' + state_name_list[i]
        cwd_state = os.path.join(cwd, state_filename)
        logging.info(f'{state_name_list[i]} is ready.')
        logging.shutdown()
        quant_num = get_quant_num(state)
        cwd_state = os.path.join(cwd, state_filename)
        if not os.path.exists(cwd_state):
            os.mkdir(state_filename)
        os.chdir(state_filename)
        continue_flag = True
        iter_num = 0
        while continue_flag:
            iter_num += 1
            continue_flag = run_network_expansion(cwd, cwd_state, mult, state, state_name_list[i], ele, p, quant_num, n_proc, copy_filename_list=copy_filename_list, qm_atom_index=atom_index, freeze_index=[],
                                                  iter=iter_num, max_iter=GraphBasedMolecularDynamics.__MAX_ITERATION)
            if iter_num >= GraphBasedMolecularDynamics.__MAX_ITERATION:
                continue_flag = False
            # end_time = time.time()
            # print('time:', end_time-start_time)


if __name__  == "__main__":
    run_auto_search()