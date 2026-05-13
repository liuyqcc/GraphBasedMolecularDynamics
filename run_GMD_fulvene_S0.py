"""
GMD example.
Reactant: fulvene
State: S0
Method: GFN2-xTB
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
GraphBasedMolecularDynamics.__MAX_BARRIER_IN_SELECT_NEXT_REACTANT = 100
GraphBasedMolecularDynamics.__MAX_BARRIER_IN_MD = 100
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
# interface_type = 'OM2MRCI_S1S0'
# interface_type = 'xTB_S0'
GraphBasedMolecularDynamics.interface_type = 'xTB_S0'



def run_GMD(copy_filename_list=[]):
    # start_time = time.time()
    cwd = os.getcwd()
    generate_logging(log_filename)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    n_proc = 1
    input_file = r'input.xyz'
    ele, p_list, _ = read_xyz(input_file)
    p = p_list[0]
    ele_total = ele
    p_total = p
    # S1 S0 T1
    state_name_list = ['S0']
    state_name = state_name_list[0]
    mult, state = (1, 0)
    # check if reaction net exists
    state_filename = 'state_' + state_name_list[0]
    cwd_state = os.path.join(cwd, state_filename)
    logging.info(f'{state_name_list[0]} is ready.')
    logging.shutdown()
    cwd_state = os.path.join(cwd, state_filename)
    if not os.path.exists(cwd_state):
        os.mkdir(state_filename)
    os.chdir(state_filename)
    qm_atom_index = [i for i in range(len(ele))]
    freeze_index = []

    os.chdir(cwd_state)
    interface_list, main_interface_index, state_name_list = init_interface(state, mult, qm_atom_index, ele_total)
    tmp_interface = interface_list[main_interface_index]
    # count MD file
    all_file_list = os.listdir(cwd_state)
    file_start_index = 0
    for filename in all_file_list:
        if filename[:len(job_filename_head)] == job_filename_head:
            file_start_index += 1
    if file_start_index == 0:
        prepare_analy_file()
        # this is the first structure, run optimize first
        opt_path = os.path.join(cwd_state, job_filename_head + '0')
        if not os.path.exists(opt_path):
            os.mkdir(job_filename_head + '0')
        run_init_minimization(opt_path, file_index=0, p=p_total, ele=ele_total, mult=mult, state=state, qm_atom_index=qm_atom_index, freeze_index=freeze_index)
        os.chdir(cwd_state)
        # analy_result_file(filename_index=[0], interface=tmp_interface, reactant_index=None, write_mode='w', write_net=False, log_info=False)
        analy_result_file(filename_index=[0], state_index=0, reactant_index=None, write_mode='w', write_net=False, log_info=False)
    os.chdir(cwd)
    init_logging('run')
    logging.info(f'=============== Current State: {state_name} ===============')
    os.chdir(cwd_state)
    if file_start_index == 0:
        next_reactant_info = select_next_reactant(type='init')
        file_start_index += 1
    else:
        next_reactant_info = select_next_reactant(type='normal')

    if next_reactant_info is not None:
        next_info, _ = next_reactant_info
        p_total = next_info['coord']
    else:
        # no next structure, end
        return False
    p_reshape = p_total.reshape(-1, 3)
    ele_qm = [ele_total[index] for index in qm_atom_index]
    p_qm = p_reshape[qm_atom_index].flatten()

    # read hash list
    react_adj = generate_adj_mat(ele_qm, p_qm)
    logging.info(f"Prepare reactant, index: {next_info['result_index']}")
    # generate product matrix
    unique_info_list = read_unique_coord(unique_coord_filename, unique_hash_filename)
    hash_list = [info_dict['hash_str'] for info_dict in unique_info_list]
    logging.info('Generate product matrix.')

    # styrene
    # bond_list = [[[3, 4, 1], [1, 3, 0]]]
    # fulvene
    bond_list = [[[11, 9, 1], [11, 10, 0], [10, 5, 1], [9, 5, 0]]]
    # butadiene
    # bond_list = [[[1, 8, 1]]]
    product_adj_list, product_hash_list = graph_generator_with_input(ele, react_adj, hash_list, bond_list)
    hash_list = hash_list + product_hash_list
    logging.info(f'Product number: {len(product_hash_list)}.')
    # write_hash(expected_hash_filename, hash_list, mode='w')
    random_state = np.random.RandomState()
    random_seed = random_state.randint(0, 2**32-1, len(product_hash_list))
    ### loop every product
    logging.info(f'Start dynamics.')
    try:
        if len(product_adj_list) > 0:
            work_pool = Pool(processes=n_proc)
            for i in range(len(product_adj_list)):
                job_index = i + file_start_index
                job_filename = job_filename_head + str(job_index)
                work_pool.apply_async(func=work_flow, args=(cwd_state, job_filename, job_index,  copy_filename_list, random_seed[i], ele_total, p_total,
                                                            state, mult, react_adj, product_adj_list[i], qm_atom_index, freeze_index))
                # work_flow(cwd_state, job_filename, job_index, copy_filename_list, random_seed[i], ele, p, state, react_adj, product_adj_list[i])
            logging.info(f'Add job:' + ' '.join([str(i+file_start_index) for i in range(len(product_adj_list))]))
            logging.info(f'Waitting job finish...')
            work_pool.close()
            work_pool.join()
            logging.info(f'{state_name} All job finished.')
        # analysis result
        filename_index = [i+file_start_index for i in range(len(product_adj_list))]
        analy_result_file(filename_index=filename_index, state_index=0, reactant_index=next_info['result_index'], write_mode='a')
    except KeyboardInterrupt:
        work_pool.terminate()
        work_pool.join()
        logging.info('Keyboard interrupt, abort.')
        exit()
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.info('Error interrupt, abort.')
        logging.info(' '.join(err_info))
        work_pool.terminate()
        work_pool.join()
        exit()
    os.chdir(cwd_state)
    logging.shutdown()
    return True


if __name__ == "__main__":
    run_GMD()