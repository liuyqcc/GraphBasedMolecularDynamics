import os
import logging
import numpy as np
from optimizer import Optimizer
from bond_connect import get_target_distance
from potential.IOfunction import read_xyz, write_xyz
from post_process import analy_traj, read_data
from graph_tools import get_bond_from_mat
from GMD_fileIO import read_reaction_mat, write_path_energy, write_opt_result

# path energies
__path_energy_filename = 'path_energy.log'
__opt_result_filename = 'opt.xyz'
__data_filename = 'data.out'
__traj_filename = 'traj.xyz'
__k_au2kcalmol = 627.5095
__min_ediff = 3


def get_start_structure_from_traj_file(cwd, job_filename, quant_num):
    # analysis trajectory
    data_filepath = os.path.join(cwd, job_filename, __data_filename)
    state_name_list, state_index_list, all_energy_list, geom_list = read_data(data_filepath, quant_num=quant_num)
    ele, coord_list, second_line_list = read_xyz(os.path.join(cwd, job_filename, __traj_filename), return_all_elements=False)
    react_mat, product_mat = read_reaction_mat(os.path.join(cwd, job_filename))

    # find bond index and bond type
    bond_index, bond_type = get_bond_from_mat(product_mat-react_mat)
    # reference bond length
    target_distance = get_target_distance(ele, bond_index, type_list=bond_type)
    # calculate 1-D geometry coordinate
    geom_coord = np.zeros_like(geom_list[0])
    for i in range(len(geom_list)):
        if bond_type[i] == 1:
            geom_coord = geom_coord + geom_list[i]
        elif bond_type[i] == -1:
            tmp_geom = target_distance[i] - geom_list[i]
            geom_coord = geom_coord + np.where(tmp_geom > 0, tmp_geom, 0)
    state_index = state_index_list[0]
    return (ele, coord_list, state_index, state_name_list, all_energy_list, geom_coord)


def get_CI_index(state_index, state_name_list, all_energy_list_time, geom_sort_index):
    CI_coord_list_time = []
    if state_index > 0:
        if state_name_list[state_index-1][0] == state_name_list[state_index][0]:
            e_diff_time = np.abs(all_energy_list_time[state_index] - all_energy_list_time[state_index-1]) * __k_au2kcalmol
            e_diff_geom = e_diff_time[geom_sort_index]
            CI_coord_list_time = np.where(e_diff_geom <= __min_ediff)[0]
    return CI_coord_list_time


def get_CP_index(state_index, state_name_list, all_energy_list_time, geom_sort_index):
    # S1-T1 or T1-S0
    CP_coord_list_time = []
    if state_name_list[state_index] == 'S1':
        # S1-T1
        if 'T1' in state_name_list:
            T_index = state_name_list.index('T1')
            e_diff_time = np.abs(all_energy_list_time[state_index] - all_energy_list_time[T_index]) * __k_au2kcalmol
            e_diff_geom = e_diff_time[geom_sort_index]
            CP_coord_list_time = np.where(e_diff_geom <= __min_ediff)[0]
    elif state_name_list[state_index] == 'T1':
        # S0-T1
        if 'S0' in state_name_list:
            S_index = state_name_list.index('S0')
            e_diff_time = np.abs(all_energy_list_time[state_index] - all_energy_list_time[S_index]) * __k_au2kcalmol
            e_diff_geom = e_diff_time[geom_sort_index]
            CP_coord_list_time = np.where(e_diff_geom <= __min_ediff)[0]
    return CP_coord_list_time


def perpare_opt_strctures(cwd, job_filename, file_index, quant_num, half_step_window, de_threshold):
    """
    half_step_window: in Angs
    """
    ##########  Analysis Trajectory ##########
    # analysis the trajectory and do optimization
    ele, coord_list_time, state_index, state_name_list, all_energy_list_time, geom_coord = get_start_structure_from_traj_file(cwd, job_filename, quant_num)
    # collect coinical intersections
    geom_sort_index = np.argsort(geom_coord)[::-1]
    CI_coord_list_time = get_CI_index(state_index, state_name_list, all_energy_list_time, geom_sort_index)
    CP_coord_list_time = get_CP_index(state_index, state_name_list, all_energy_list_time, geom_sort_index)
    e_list_time = all_energy_list_time[state_index]
    analy_info = analy_traj(e_list_time[geom_sort_index], CI_coord_list_time, CP_coord_list_time, geom_coord[geom_sort_index], sort_index=geom_sort_index, half_time_window=half_step_window, de_threshold=de_threshold)
    state_name = state_name_list[state_index]
    state = int(state_name[1:])
    if state_name[0] != 'S':
        state = state - 1
    start_structure_info = []
    for i in range(len(analy_info)):
        index_time, index_geom, species_name = analy_info[i]
        info_dict = {}
        info_dict['ele'] =  ele
        info_dict['coord'] = coord_list_time[index_time]
        info_dict['energy_list'] = all_energy_list_time.T[index_time]

        info_dict['file_index'] = file_index
        info_dict['species_name'] = species_name
        info_dict['finished_flag'] = 1
        info_dict['state'] = state
        info_dict['path_index'] = i
        info_dict['index_time'] = index_time
        info_dict['index_geom'] = index_geom
        start_structure_info.append(info_dict)
    react_energy = all_energy_list_time[state_index][0]
    return (start_structure_info, react_energy)


def run_optimization(interface_opt, structure_info_list, react_energy, state_index, interface_MECP=None, write_opt_result_flag=True, write_path_flag=True, log_flag=True):
    ########## init interface ##########
    if len(structure_info_list) == 0:
        logging.info('Nothing to optimize----------')
        return
    logging.info('Run Optimization ----------')
    state = structure_info_list[0]['state']
    # state_index = interface_opt.get_state_index(state)
    opt = Optimizer(state=state)
    opt.set_interface(interface_opt)
    opt.set_min_ediff(min_ediff=__min_ediff)
    species_energy_list = []

    for i, info_dict in enumerate(structure_info_list):
        result_coord = info_dict['coord']
        result_energy_list = info_dict['energy_list']
        species_name = info_dict['species_name']
        result_coord_list = []
        if species_name == 'Min':
            # minimization
            result = opt.opt_min(result_coord, state, info_dict['ele'])
            result_coord, finished_flag, result_energy_list = result
            if finished_flag in [0, 1]:
                # normal finished or failed
                pass
            elif finished_flag == 2:
                # turn to find MECI
                species_name = 'MinCI'
                info_dict['species_name'] = species_name
        elif species_name == 'TS':
            # TODO: TS optimization
            finished_flag = 1
        elif species_name == 'MECP':
            if interface_MECP is not None:
                result = opt.opt_MECP(result_coord, interface_opt, interface_MECP,state_A=1, state_B=1)
                if result is not None:
                    finished_flag = 0
                    result_coord, result_energy_list = result
                else:
                    # failed
                    finished_flag = 1
            else:
                finished_flag = 1

        if species_name == 'MECI' or species_name == 'MinCI':
            # search MECI
            result = opt.opt_MECI(result_coord)
            if result is not None:
                finished_flag = 0
                result_coord, result_energy_list = result
            else:
                # failed
                finished_flag = 1
        index_time = info_dict['index_time']
        if len(result_energy_list) == 1:
            species_energy_list.append(result_energy_list[0])
        else:
            species_energy_list.append(result_energy_list[state_index])
        result_coord_list.append(result_coord)
        info_dict['coord'] = result_coord
        info_dict['energy_list'] = result_energy_list
        info_dict['finished_flag'] = finished_flag

    ########## Write result ##########
    if write_opt_result_flag:
        write_opt_result(__opt_result_filename, structure_info_list)
    if write_path_flag:
        write_path_energy(__path_energy_filename,
                          [info_dict['species_name'] for info_dict in structure_info_list],  # species_name
                          [info_dict['index_time'] for info_dict in structure_info_list],  # index_time
                          species_energy_list,
                          react_energy)

    ########## Done ##########
    if log_flag:
        logging.info('Optimization done.')
        logging.info(f'-------------------------------------------------')
        for i, item in enumerate(structure_info_list):
            if i == 0:
                logging.info(f'| Name | Index |   Energy   | InTraj | Finished |')
            species_name = item['species_name']
            index_time = item['index_time']
            finished_flag = item['finished_flag']
            index = i + 1
            energy = species_energy_list[i]
            logging.info(f'| {species_name:<5s} | {index:<5d} | {energy:<10.6f} | {index_time:<6d} | {finished_flag:<8d} |')
        logging.info(f'-------------------------------------------------')
