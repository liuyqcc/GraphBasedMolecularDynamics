"""
Graph-based Molecular Dynamics (GMD)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import sys
import shutil
import logging
import traceback
import copy
from multiprocessing import Pool
from PropagationClassical import PropagationClassical
from potential.interfaceMNDO import InterfaceMNDO
from potential.interfaceXTB import InterfaceXTB
from potential.IOfunction import read_xyz, write_xyz
from potential.atomic_mass import get_mass
from potential.Constants import Constants
from FormatOutput import FormatOutput
from BondGuidedBias import BondGuidedBias
from bond_connect import generate_adj_mat, get_target_distance
from graph_tools import gen_graph_hash, graph_generator, get_reaction_direction, graph_generator_with_input
from post_process import analy_traj, analy_traj_on_the_fly, analy_MECI_traj, read_data
from optimizer import Optimizer
from multiprocessing.pool import Pool
from molecular_dynamics_tools import scale_velocity, init_velocity
from GMD_tools import get_bond_length_vec, check_energy
from GMD_fileIO import init_logging, generate_logging, \
                        write_reaction_mat, read_reaction_mat,\
                        write_hash, read_hash,\
                        write_reaction_net, read_reaction_net,\
                        write_path_energy, read_path_energy,\
                        read_opt_result,\
                        read_result_coord, write_result_coord,\
                        read_unique_coord, write_unique_coord

from GMD_optimization import run_optimization, perpare_opt_strctures

print_level = 1
__min_ediff = 5
__au2kcalmol = 627.5095
__MAX_BARRIER_IN_SELECT_NEXT_REACTANT = 30
__MAX_BARRIER_IN_MD = 30
__MAX_ITERATION = 50

__HALF_STEP_WINDOW = 0.1  # Angs
__DE_THRESHOLD_FILTER = 3  # kcal/mol
__GMD_STOP_AT_FIRST_MINIMUM = False

__BREAKING_NUM = 2
__FORMATION_NUM = 2
__MAX_FRAG = -1
__MAX_RADICAL = 0

# result coordinates
result_coord_filename = 'result_coord.xyz'
# result hash + SPRINT coordinate
result_hash_filename = 'result_hash.log'
# unique hash + SPRINT coordinate
unique_hash_filename = 'unique_hash.log'
# corresponding unique coordinates
unique_coord_filename = 'unique_coord.xyz'
# reaction network file
reaction_net_filename = 'net.log'
# expected Hash
expected_hash_filename = 'expected_hash.log'
# reaction matrix
reaction_matrix_filename = 'reaction_mat.log'
# path energies
path_energy_filename = 'path_energy.log'
# decay strucutre filename
decay_traj_filename = 'decay_coord.xyz'
# job filename head
job_filename_head = 'test_'
opt_result_filename = 'opt.xyz'
log_filename = 'run'
# interface_type = 'xTB_S0'
interface_type = 'OM2MRCI_S1S0'


def init_MNDO_interface(state, mult):
    interface = InterfaceMNDO()
    IMULT = mult
    # ICI1, ICI2 = (6, 6)
    ICI1, ICI2 = (7, 6)
    # 0 for ground state and 1 for first exited state
    if state > 0:
        # calculate state i and state i-1
        state_line = f' {state} {state+1}'
        quant_num = 2
    else:
        state_line = f' {state+1}'
        quant_num = 1
    interface.set_head_line('IOP=-6 JOP=-2 IGEOM=1 IFORM=1 KITSCF=1000 +\n'
                            'ICROSS=1 IPRINT=1 IOUTCI=2 +\n'
                            f'KHARGE=0 IMULT={IMULT} MULTCI=0 IUHF=-1 NPRINT=2 +\n'
                            f'NCIGRD={quant_num} KCI=5 MOVO=0 ICI1={ICI1} ICI2={ICI2} +\n'
                            'NCIREF=3 MCIREF=0 LEVEXC=2 IROOT=3 LROOT=0 ICUTG=-1 ICUTS=-1 +\n'
                            'ktrial=11 ipubo=1 \n', ICI1=ICI1, ICI2=ICI2)
    interface.set_end_line(state_line)
    interface.set_head_line_MECI('IOP=-6 JOP=0 IGEOM=1 IFORM=1 KITSCF=1000 +\n'
                                 'ICROSS=3 IPRINT=1 IOUTCI=2 +\n'
                                 f'KHARGE=0 IMULT={IMULT} MULTCI=0 IUHF=-1 NPRINT=2 +\n'
                                 f'NCIGRD={quant_num} KCI=5 MOVO=0 ICI1={ICI1} ICI2={ICI2} +\n'
                                 'NCIREF=3 MCIREF=0 LEVEXC=2 IROOT=3 LROOT=0 MAXRTL=3000\n')
    interface.set_end_line_MECI(state_line + '\n'
                                '5.000   5.000')
    interface.set_act_space(ICI1, ICI2)
    interface.set_quant_num(quant_num)
    if mult == 1:
        interface.set_working_filename('QM_S')
    elif mult == 3:
        interface.set_working_filename('QM_T')
    else:
        interface.set_working_filename('QM')
    interface.init_run()
    return interface


def init_xTB_interface_sp(state, mult, file_name=1):
    interface = InterfaceXTB()
    interface.set_quant_num(1)
    interface.set_cwd(os.getcwd())
    interface.set_multiplicity(mult)
    interface.set_job_str(f'--grad --gfn 2 --spinpol --tblite --uhf {mult-1}')
    if mult == 1:
        work_name = 'S'
    elif mult == 3:
        work_name = 'T'
    interface.set_working_filename('QM_' + work_name)
    # interface.init_run('/dev/shm', 'xtb_' + str(file_name))
    interface.init_run(os.getcwd(), 'QM_' + work_name)
    return interface


def init_xTB_interface(state, mult, file_name=1):
    interface = InterfaceXTB()
    interface.set_quant_num(1)
    interface.set_cwd(os.getcwd())
    interface.set_multiplicity(mult)
    interface.set_job_str(f'--grad --gfn 2 --uhf {mult-1}')
    if mult == 1:
        work_name = 'S'
    else:
        work_name = 'T'
    interface.set_working_filename('QM_' + work_name)
    # interface.init_run('/dev/shm', 'xtb_' + str(file_name))
    interface.init_run(os.getcwd(), 'QM_' + work_name)
    return interface


def init_interface(state, mult, qm_atom_index, ele):
    if interface_type == 'OM2MRCI_S1S0':
        mm_atom_num = len(ele) - len(qm_atom_index)
        if mult == 1:
            # S state
            S_state = state
            main_interface_index = 0
        elif mult == 3:
            S_state = 1
            main_interface_index = 1
        T_state = 0

        S_state_name_list = ['S0', 'S1']
        # S_state_name_list = ['S0']
        # T_state_name_list = ['T1']
        T_state_name_list = []
        state_name_list = S_state_name_list + T_state_name_list
        # prmtop_filename = r'/home/liuyq/gaussian/graph_driven/test_qmmm/input.prmtop'
        # S0 and S1 (state = 1 and 2)
        MNDO_interface_S = init_MNDO_interface(S_state, mult=1)
        MNDO_interface_S.set_elements(ele)
        MNDO_interface_S.set_data_read_mode(read_mode=1)
        return ([MNDO_interface_S], main_interface_index, state_name_list)
    if interface_type == 'xTB_S0':
        if mult == 1:
            # S state
            S_state = state
            main_interface_index = 0
        elif mult == 3:
            S_state = 1
            main_interface_index = 1
        S_state_name_list = ['S0']
        T_state_name_list = []
        state_name_list = S_state_name_list + T_state_name_list
        # S0 and S1 (state = 1 and 2)
        pid = os.getpid()
        xtb_interface_S = init_xTB_interface(state=0, mult=1, file_name=pid)
        xtb_interface_S.set_elements(ele)
        return ([xtb_interface_S], main_interface_index, state_name_list)
    if interface_type == 'xTB_T1S0':
        if mult == 1:
            # S state
            S_state = state
            main_interface_index = 0
        elif mult == 3:
            S_state = 1
            main_interface_index = 1

        S_state_name_list = ['S0']
        T_state_name_list = ['T1']
        # state_name_list = T_state_name_list + S_state_name_list
        state_name_list = S_state_name_list + T_state_name_list
        # prmtop_filename = r'/home/liuyq/gaussian/graph_driven/test_qmmm/input.prmtop'
        # S0 and S1 (state = 1 and 2)
        pid = os.getpid()
        xtb_interface_S = init_xTB_interface_sp(state=0, mult=1, file_name=pid)
        xtb_interface_T = init_xTB_interface_sp(state=0, mult=3, file_name=pid)
        xtb_interface_S.set_elements(ele)
        xtb_interface_T.set_elements(ele)

        return ([xtb_interface_S, xtb_interface_T], main_interface_index, state_name_list)
        # return ([xtb_interface_S], main_interface_index, state_name_list)


def init_interface_test(state, mult, qm_atom_index, ele):
    main_interface_index = 0
    S_state = state
    S_state_name_list = ['S0']
    state_name_list = S_state_name_list
    # prmtop_filename = r'/home/liuyq/gaussian/graph_driven/test_qmmm/input.prmtop'
    # S0 and S1 (state = 1 and 2)
    xtb_interface = init_xTB_interface(state=S_state, mult=mult, file_id=1)
    xtb_interface.set_elements(ele)
    xtb_interface.set_quant_num(1)
    return ([xtb_interface], main_interface_index, state_name_list)


def check_new_minimum(energy_list, geom_coord, geom_diff_list, half_time_window=0.05, de_threshold=0.01):
    geom_sort_index = np.argsort(geom_coord)[::-1]
    e_list_time = np.array(energy_list)
    geom_coord_geom = np.array(geom_coord)
    analy_info = analy_traj_on_the_fly(e_list_time[geom_sort_index], geom_coord_geom[geom_sort_index], sort_index=geom_sort_index, half_time_window=half_time_window, de_threshold=de_threshold)
    for i in range(len(analy_info)):
        index_time, index_geom, species_name = analy_info[i]
        if species_name == 'Min' and geom_diff_list[index_time]:
            return True
    return False


def run_GraphDriven(cwd, interface_list, job_filename, copy_filename_list,
                    random_seed, ele_total, p_total, bond_index, bond_type, restrain_index, state, state_name_list,
                    main_interface_index, QM_index=None, freeze_index=[]):
    logging.info('Start Graph-Based Molecular Dynamics.')
    os.chdir(cwd)
    np.random.seed(random_seed)
    float_type = np.float64
    if not os.path.exists(job_filename):
        os.mkdir(job_filename)
    for filename in copy_filename_list:
        shutil.copy(filename, os.path.join(job_filename, filename))
    os.chdir(job_filename)
    logging.info('Path:' + os.path.join(cwd, job_filename))
    logging.info('Seed:' + f'{random_seed}')
    Const = Constants()
    remove_RT = True
    atom_num_total = len(ele_total)
    if QM_index is None:
        QM_index = [i for i in range(atom_num_total)]
    p_total = np.array(p_total, float_type).flatten() / Const.bohr2angs
    p_reshape = p_total.reshape(-1, 3)
    p_total_old = np.copy(p_total)
    p_QM = p_reshape[QM_index].flatten()
    ele_QM = [ele_total[i] for i in QM_index]
    reactant_adj_mat = generate_adj_mat(ele_QM, p_QM * Const.bohr2angs)
    ########## input file Setting ##########
    Format = FormatOutput()
    mass_amu = get_mass(ele_total)
    mass_total = np.array(mass_amu, float_type).repeat(3).flatten() * Const.mass_amu2au
    ########## MNDO Interface Setting ##########
    quant_num_list = [inter.quant_num for inter in interface_list]
    current_state_index = interface_list[main_interface_index].get_state_index(state)
    current_state_index += sum(quant_num_list[:main_interface_index])
    ########## init MD Setting ##########
    dt_fs = 1                       # time step (fs)
    dt = dt_fs * Const.time_fs2au   # fs to a.u.
    total_step = 10000
    addition_relax_step = 0
    target_T = 300                  # temperature
    DOF = len(p_total) - 6
    DOF = DOF - len(freeze_index) * 3

    ########## Initilize velocity ##########
    v = init_velocity(p_total, atom_num_total, mass_total, target_T=target_T)
    ########## Propagater Setting ##########
    prop_class = PropagationClassical()
    ########## Bias Setting ##########
    target_distance = get_target_distance(ele_QM, bond_index, type_list=bond_type)
    target_distance = [item / Const.bohr2angs for item in target_distance]
    # unit: bohr
    bias = BondGuidedBias(target_value=target_distance, type_list=bond_type)
    bias.set_restrain_table(p_QM, restrain_index)
    bias_step = 20
    ########## Write setting ##########
    # calculate energy, force and NAC
    traj_file = r'traj.xyz'
    # velo_file = r'velo.xyz'
    data_file = r'data.out'
    traj_file_obj = open(traj_file, 'w')
    # velo_file_obj = open(velo_file, 'w')
    data_file_obj = open(data_file, 'w')

    run_result_list = [interface.run(p_total * Const.bohr2angs, None, save_err=True, save_nor=True) for interface in interface_list]
    if None not in run_result_list:
        tmp_Ei = []
        tmp_Fi = []
        for item in run_result_list:
            tmp_Ei.extend(item[0])
            tmp_Fi.extend(item[1])
        # tmp_NAC = [item[2] for item in run_result_list]
    else:
        logging.info('Job error, abort.')
        return False
    Ei_list = tmp_Ei
    Fi = tmp_Fi[current_state_index]
    old_Fi = np.copy(Fi)

    traj_block = Format.format_traj_data(ele_total, p_total * Const.bohr2angs, 0, 0)
    for line in traj_block:
        traj_file_obj.writelines(line)
    # velo_block = Format.format_traj_data(ele_total, v, 0, 0)
    # for line in velo_block:
    #     velo_file_obj.writelines(line)
    converge_step = 0
    energy_too_high_flag = False
    reactant_energy = Ei_list[current_state_index]
    energy_list = []
    bias_energy_list = []
    history_energy_list = []
    geom_diff_list = []
    cv_list = []
    max_relax_loop = 3
    relax_step = 0
    normal_finished = True
    # data format
    data_str = 'step state ' + ' '.join([f'E{name:<7}' for name in state_name_list]) + ' Ekin     Etot     |F|     ' + ' '.join([f'b_{item[0]+1}_{item[1]+1}    ' for item in bond_index])
    data_file_obj.writelines(data_str + '\n')
    logging.info('step cv |Fbias|')
    for current_step in range(total_step):
        bond_length_list, bond_vec_list = get_bond_length_vec(p_QM, bond_index)

        K_ene = np.dot(v*mass_total, v) / 2
        data_str = '{:<5d} {:<3} '.format(current_step, state_name_list[current_state_index]) +\
                   ' '.join([f'{item:<8.6f}' for item in Ei_list]) +\
                   ' {:<8.6f} {:<8.6f}'.format(K_ene, Ei_list[current_state_index] + K_ene) +\
                   ' {:<8.6f} '.format(np.linalg.norm(Fi)) +\
                   ' '.join(['{:<8.6f}'.format(item * Const.bohr2angs) for item in bond_length_list])
        data_file_obj.writelines(data_str + '\n')

        # add cv
        cv = bias.get_cv_value(bond_length_list)
        cv_list.append(cv)
        # history_cv.append(cv)

        bias_e, bias_f = bias.get_bias_force(bond_length_list, bond_vec_list)
        energy_list.append(Ei_list[current_state_index])
        history_energy_list.append(Ei_list[current_state_index])
        if converge_step == 0:
            bias_energy_list.append(bias_e)
        else:
            bias_energy_list.append(0)
        geom_diff_list.append(np.any(reactant_adj_mat != generate_adj_mat(ele_QM, p_QM * Const.bohr2angs)))

        if current_step % bias_step == 0:
            # check energy
            if converge_step == 0:
                energy_result = check_energy(reactant_energy, history_energy_list, bias_energy_list, E_thres=__MAX_BARRIER_IN_MD)
                bias_energy_list = []
                history_energy_list = []
                if energy_result > 0:
                    energy_too_high_flag = True
                else:
                    relax_step = 0
                    energy_too_high_flag = False
            else:
                # do not check energy in relax step
                energy_too_high_flag = False

            # standard step
            if converge_step == 0 and not energy_too_high_flag:
                bias.add_bias(cv)
                # bias.add_bias(min(history_cv))
                # history_cv = []
            # relax step
            elif energy_too_high_flag:
                relax_step += 1
                if relax_step > max_relax_loop:
                    normal_finished = False
                    if energy_result == 1:
                        logging.info('PES energy is too high, abort.')
                    elif energy_result == 2:
                        logging.info('Bias energy is too high, abort.')
                    break
            if len(energy_list) > 0 and converge_step == 0 and __GMD_STOP_AT_FIRST_MINIMUM:
                # check if new Min is found
                if check_new_minimum(energy_list, cv_list, geom_diff_list, half_time_window=__HALF_STEP_WINDOW / Const.bohr2angs, de_threshold=__DE_THRESHOLD_FILTER / Const.hartree2kcal):
                    logging.info('New minimum found.')
                    converge_step = current_step

        if converge_step == 0:
            restrain_f = bias.get_restrain_force(p_total)
            bias_f = bias_f + restrain_f
            logging.info(f'{current_step:<4} {cv:<.6f} {np.linalg.norm(bias_f):<.6f}')
        else:
            bias_f = 0
            logging.info(f'{current_step:<4} {cv:<.6f}')

        # update velocity first half
        v = prop_class.update_velocity_half_step(v, dt, Fi+bias_f, mass_total)
        # scale velocity
        v = scale_velocity(p_total, v, mass_total, atom_num_total, target_T, remove_RT, freeze_index, DOF)
        # velocity constraints
        # v = v.reshape(-1, 3)
        # p_total = p_total.reshape(-1, 3)
        # for j in range(len(water_index_list)):
        #     v[water_index_list[j]] = settle_velocity(p_total[water_index_list[j]], v[water_index_list[j]], C_mat, water_mass, dt)
        # v[freeze_index] = 0
        # v = v.flatten()
        # update coordinates
        p_total_old[:] = p_total
        # p_total = p_total.flatten()
        p_total = prop_class.update_coordinates(p_total, v, dt)
        p_QM = p_total.reshape(-1, 3)[QM_index].flatten()
        # position constraints
        # p_total = p_total.reshape(-1, 3)
        # for j in range(len(water_index_list)):
            # p_total[water_index_list[j]] = settle_position(p_total_old[water_index_list[j]], p_total[water_index_list[j]], water_mass, ra, rb, rc)
        # p_total = p_total.flatten()

        run_result_list = [interface.run(p_total * Const.bohr2angs, None, save_err=True, save_nor=True) for interface in interface_list]
        if None not in run_result_list:
            tmp_Ei = []
            tmp_Fi = []
            for item in run_result_list:
                tmp_Ei.extend(item[0])
                tmp_Fi.extend(item[1])
            # tmp_NAC = [item[2] for item in run_result_list]
        else:
            logging.info('Job error, abort.')
            return False
        Ei_list = tmp_Ei
        Fi = tmp_Fi[current_state_index]

        # update velocity second half
        v = prop_class.update_velocity_half_step(v, dt, Fi+bias_f, mass_total)

        old_Fi[:] = Fi

        traj_block = Format.format_traj_data(ele_total, p_total * Const.bohr2angs, current_step*dt_fs, current_step)
        for line in traj_block:
            traj_file_obj.writelines(line)
        # velo_block = Format.format_traj_data(ele_total, v, current_step*dt_fs, current_step)
        # for line in velo_block:
        #     velo_file_obj.writelines(line)
        if cv <= 1e-6 and converge_step == 0:
            converge_step = current_step
            logging.info('Converged, running additional relax step.')
            # logging.info('MD converged')
        if converge_step != 0:
            if current_step - converge_step >= addition_relax_step:
                logging.info('Dynamics finished')
                break

    traj_file_obj.close()
    data_file_obj.close()
    # velo_file_obj.close()
    return normal_finished
    

def run_MECI_minimization(cwd, quant_num, p, ele, coord_index, state):
    logging.info('Start to optimize Minimum from MECI.')
    os.chdir(cwd)
    min_ediff = __min_ediff
    interface_opt = init_MNDO_interface(state=state)
    interface_opt.set_elements(ele)

    ########## init state setting ##########
    interface_opt.set_quant_num(quant_num)
    interface_opt.set_data_read_mode(read_mode=1)
    opt = Optimizer(state=state)
    opt.set_interface(interface_opt)
    opt.set_min_ediff(min_ediff=min_ediff)

    ########## Minimum ##########
    result = opt.opt_min(p, state=state)
    # Result type. 0: local minimum, 1: optimization failed, 2: energy degenerate
    final_coord, result_type, energy_list = result
    finished_flag = result_type
    if result_type == 0 or result_type == 1:
        # success or fail
        second_line = ' '.join(['Min', str(finished_flag), str(coord_index)] + [str(item) for item in energy_list])
        write_xyz('Minimum.xyz', [ele], [final_coord], [second_line])
        return
    elif result_type == 2:
        # could be MECI
        ########## MECI ##########
        logging.info('Start to optimize MECI.')
        # optimize MECI
        save_filename = 'MECI_' + str(coord_index) + '.log'
        result = opt.opt_MECI_MNDO(final_coord, save_filename=save_filename)
        if result is not None:
            final_coord, energy_list = result
            result_type = 0
            MECI_second_line = ' '.join(['MECI', str(result_type), str(coord_index)] + [str(item) for item in energy_list])
        else:
            result_type = 1
            second_line = ' '.join(['MECI', str(result_type), str(coord_index)] + [str(item) for item in energy_list])
        write_xyz('MECI.xyz', [ele], [final_coord], [MECI_second_line])

    ########## Done ##########
    logging.info('Optimization done.')


def run_init_minimization(cwd_opt, file_index, p, ele, mult, state, qm_atom_index, freeze_index):
    os.chdir(cwd_opt)
    init_logging(job_filename_head + str(file_index))
    logging.info('Start to optimize Minimum initial point.')
    ########## Init Interface ##########
    interface_list, main_interface_index, state_name_list = init_interface(state, mult, qm_atom_index, ele)
    tmp_interface = interface_list[main_interface_index]
    quant_num = tmp_interface.quant_num
    ########## Init Optimizer ##########
    opt = Optimizer(state=state)
    opt.set_interface(tmp_interface)
    opt.set_min_ediff(min_ediff=__min_ediff)
    ########## Optimization ##########
    species_name = 'Min'
    info_dict = {}
    info_dict['ele'] =  ele
    info_dict['coord'] = p
    info_dict['file_index'] = file_index
    info_dict['species_name'] = species_name
    info_dict['finished_flag'] = 1
    info_dict['state'] = state
    info_dict['path_index'] = 0
    info_dict['index_time'] = 0
    info_dict['index_geom'] = 0
    info_dict['energy_list'] = [None for _ in range(quant_num)]
    start_structure_info = [info_dict]
    state_index = tmp_interface.get_state_index(state)
    run_optimization(tmp_interface, start_structure_info, react_energy=None, state_index=state_index)
    ########## Done ##########
    logging.shutdown()


def run_only_plot():
    cwd = r'D:\code\graph_driven\benzene_plot'
    os.chdir(cwd)
    # read file
    input_file = r'input.xyz'
    ele, p = read_xyz(os.path.join(cwd, input_file))
    # generate graph
    connect_mat = generate_adj_mat(ele, p)
    react_adj_mat = np.array(connect_mat, np.int32)
    # generate product
    hash_list = [gen_graph_hash(ele, react_adj_mat)]
    hash_list = []
    product_adj_mat, hash_list = graph_generator(ele, react_adj_mat, hash_list, b=__BREAKING_NUM, f=__FORMATION_NUM)
    product_num = len(product_adj_mat)
    # print('WL num:', product_num)
    # networkx_check(ele, product_adj_mat)
    # exit()
    graph_plot(ele, product_adj_mat, list(hash_list), 'test')
    print('ALL FINISHED')


def work_flow(cwd, job_filename, file_index, copy_filename_list, random_seed, ele_total, react_p, state, mult, react_adj, product_adj, qm_atom_index=[], freeze_index=[]):
    """
    Run graph-based MD and optimize.
    """
    Const = Constants()
    try:
        ########## MNDO Interface Setting ##########
        # qm_interface_list, mm_interface, main_interface_index, state_name_list = init_interface(state=state, mult=mult, qm_atom_index=qm_atom_index, ele=[ele_total[i] for i in qm_atom_index])
        os.chdir(cwd)
        if not os.path.exists(job_filename):
            os.mkdir(job_filename)
        os.chdir(job_filename)
        interface_list, main_interface_index, state_name_list = init_interface(state, mult, qm_atom_index, [ele_total[i] for i in qm_atom_index])
        init_logging(job_filename)
        bond_index, bond_type, restrain_index = get_reaction_direction(react_adj, product_adj)
        write_reaction_mat(react_adj, product_adj)
        normal_finished = run_GraphDriven(cwd, interface_list, job_filename, copy_filename_list,
                                          random_seed, ele_total, react_p, bond_index, bond_type,
                                          restrain_index, state, state_name_list,
                                          main_interface_index, qm_atom_index, freeze_index)
        # TODO:
        if normal_finished:
            # Graph-based MD is finished
            # os.chdir(os.path.join(cwd, job_filename))
            quant_num = sum([inter.quant_num for inter in interface_list])
            state_index = sum([interface_list[i].quant_num for i in range(main_interface_index)]) + interface_list[main_interface_index].get_state_index(state)
            # now do the post process
            # run_optimization(cwd, interface, job_filename, quant_num=quant_num)
            structure_info_list, react_energy = perpare_opt_strctures(cwd, job_filename, file_index, quant_num, half_step_window=__HALF_STEP_WINDOW, de_threshold=__DE_THRESHOLD_FILTER)
            interface_MECP = None
            if len(interface_list) >= 2:
                if main_interface_index == 0:
                    interface_MECP = interface_list[1]
                else:
                    interface_MECP = interface_list[0]
            run_optimization(interface_list[main_interface_index], structure_info_list, react_energy, state_index=state_index, write_opt_result_flag=True, write_path_flag=True, interface_MECP=interface_MECP)
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.info(' '.join(err_info))

    logging.info(f'Job {job_filename} finished.')
    logging.shutdown()


def work_MECI_min(cwd, job_filename, copy_filename_list, ele, MECI_coord, state, coord_index):
    """
    Run MECI minimization
    """
    try:
        ### initilize logging
        ### check state
        os.chdir(cwd)
        if not os.path.exists(job_filename):
            os.mkdir(job_filename)
        os.chdir(job_filename)
        init_logging(job_filename)
        if state > 0:
            quant_num = 2
        else:
            quant_num = 1
        run_MECI_minimization(os.path.join(cwd, job_filename), quant_num, MECI_coord, ele, coord_index, state)

    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err_info = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.info(' '.join(err_info))

    logging.info(f'Job {job_filename} finished.')
    logging.shutdown()


def read_reaction(filename_index=[]):
    """
    Read coordinates and information from single job.

    Also add hash code.
    """
    # read file
    all_opt_info_list = []
    for i in range(len(filename_index)):
        filename = job_filename_head + str(filename_index[i])
        file_path = os.path.join(filename, opt_result_filename)
        if os.path.exists(file_path):
            opt_info_list = read_opt_result(file_path)
            if len(opt_info_list) != 0:
                # optimization finished, read coordinate
                all_opt_info_list.extend(opt_info_list)
    # add hash code and SPRINT coordinate
    for info_dict in all_opt_info_list:
        coord = info_dict['coord']
        ele = info_dict['ele']
        adj_mat = generate_adj_mat(ele, coord)
        hash_str = gen_graph_hash(ele, adj_mat)
        info_dict['hash_str'] = hash_str
    return all_opt_info_list


def read_decay_structure(path):
    result = read_xyz(path, return_all_elements=False)
    if result is None:
        return ([], [], [], [], [], [])
    ele, p_list, second_line_list = result
    p_type = []
    finished_flag = []
    decay_from_index = []
    energy_list = []
    for second_line in second_line_list:
        data = second_line.strip().split()
        p_type.append(data[0])
        finished_flag.append(int(data[1]))
        decay_from_index.append(int(data[2]))
        energy_list.append(float(data[3]))
    return (ele, p_list, p_type, finished_flag, decay_from_index, energy_list)


def check_unique_hash(new_hash, unique_hash_list):
    """
    Returns
    -------

    (unique_flag, index)

    unique_flag: bool
        True for unique, False for not.

    index: int
        Index for this hash.
    """
    unique_flag = True
    repeat_hash_index = len(unique_hash_list)
    if new_hash in unique_hash_list:
        repeat_hash_index = unique_hash_list.index(new_hash)
        unique_flag = False
        '''
        SPRINT_threshold = 1e-2
        repeat_hash_index_list = [index for index, item in enumerate(hash_list) if item == new_hash]
        for repeat_hash_index in repeat_hash_index_list:
            if np.max(np.abs(new_S - S_list[repeat_hash_index])) < SPRINT_threshold:
                # repeat structure, add index ot all_index_list
                unique_flag = False
                return (False, repeat_hash_index)
        '''
    return (unique_flag, repeat_hash_index)


def prepare_analy_file():
    """
    Generate empty files.
    """
    # write coordinate
    with open(result_coord_filename, 'w') as f:
        pass
    with open(result_hash_filename, 'w') as f:
        pass
    with open(unique_hash_filename, 'w') as f:
        pass
    with open(unique_coord_filename, 'w') as f:
        pass
    with open(reaction_net_filename, 'w') as f:
        pass
    with open(expected_hash_filename, 'w') as f:
        pass


def analy_result_file(filename_index=[], state_index=None, reactant_index=0, write_mode='w', write_net=True, log_info=True):
    """
    Analysis current results.
    """
    # read file
    current_result_info_list = read_reaction(filename_index)

    # read old result
    histroy_result_info_list = read_result_coord(result_coord_filename, result_hash_filename)
    if len(histroy_result_info_list) != 0:
        reactant_result_info = histroy_result_info_list[reactant_index]
    histroy_unique_info_list = read_unique_coord(unique_coord_filename, unique_hash_filename)

    # add 'result_index' term
    start_index_result_coord = len(histroy_result_info_list)
    for i, info_dict in enumerate(current_result_info_list):
        info_dict['result_index'] = i + start_index_result_coord

    # find unique structure
    all_unique_info_list = []
    all_unique_info_list.extend(histroy_unique_info_list)

    # search unique structure in current result
    unique_hash_list = [info_dict['hash_str'] for info_dict in histroy_unique_info_list]
    for info_dict in current_result_info_list:
        if info_dict['species_name'] == 'TS':
            info_dict['unique_index'] = -1
            continue
        # elif info_dict['species_name'] == 'MECP':
        #     info_dict['unique_index'] = -2
        #     continue
        hash_str = info_dict['hash_str']
        unique_flag, unique_index = check_unique_hash(hash_str, unique_hash_list)
        info_dict['unique_index'] = unique_index
        if unique_flag:
            unique_hash_list.append(hash_str)
            unique_info_dict = copy.deepcopy(info_dict)
            unique_info_dict['unique_index'] = len(all_unique_info_list)
            unique_info_dict['repeat_index'] = [unique_info_dict['result_index']]
            all_unique_info_list.append(unique_info_dict)
        else:
            # check finished flag
            first_unique_info = all_unique_info_list[unique_index]
            if first_unique_info['finished_flag'] == 1 and info_dict['finished_flag'] == 0:
                # relapce failed result
                first_unique_info['repeat_index'].insert(0, info_dict['result_index'])
                tmp_repeat_index = first_unique_info['repeat_index']
                first_unique_info = copy.deepcopy(info_dict)
                first_unique_info['repeat_index'] = tmp_repeat_index
            else:
                first_unique_info['repeat_index'].append(info_dict['result_index'])

    for i, info_dict in enumerate(histroy_unique_info_list):
        if reactant_index in info_dict['repeat_index']:
            reactant_result_info['unique_index'] = i
            break
    if log_info:
        logging.info(f"Reactant unique index: {reactant_result_info['unique_index']}")
        logging.info(f"Result index: {reactant_index}")
        logging.info(f'New unique product number: {len(all_unique_info_list)-len(histroy_unique_info_list)}')
    
    # write coordinate
    write_result_coord(current_result_info_list, result_coord_filename, result_hash_filename, mode=write_mode)
    # write unique result. Note: always write all
    write_unique_coord(all_unique_info_list, unique_coord_filename, unique_hash_filename, mode='w')
    if write_net:
        # state = all_unique_info_list[0]['state']
        # state_index = interface.get_state_index(state)
        # write reaction network
        write_reaction_net(reaction_net_filename, state_index, current_result_info_list, reactant_result_info, mode=write_mode)


def select_next_reactant(type='normal'):
    logging.info('Try to select next reactant')
    # expected hash list
    # expected_hash_list = read_hash(expected_hash_filename)
    net_info_list = read_reaction_net(reaction_net_filename)
    # read unique result file (find all finished product)
    # unique_info_list = read_unique_coord(unique_coord_filename, unique_hash_filename)
    result_info_list = read_result_coord(result_coord_filename, result_hash_filename)
    if type == 'init':
        return (result_info_list[0], None)
    # reactant_result_index_list = list(set([path_info['result_index_list'][0] for path_info in net_info_list]))
    reactant_unique_index_list = list(set([path_info['unique_index_list'][0] for path_info in net_info_list]))
    valid_path_info_list = []
    for path_info in net_info_list:
        species_num = len(path_info['unique_index_list'])
        species_name_list = [item for item in path_info['species_name_list']]
        # replace the name of reactant with R
        for i in range(species_num):
            if path_info['unique_index_list'][i] in reactant_unique_index_list:
                # it is a reactant
                species_name_list[i] = 'R'
        path_energy_list = path_info['path_energy_list']
        # path_max_barrier = (max(path_energy_list) - path_energy_list[0]) * __au2kcalmol
        # print(path_info['result_index_list'], species_name_list, path_max_barrier)
        # find valid path
        path_max_barrier = 0
        if 'Min' in species_name_list or 'MinCI' in species_name_list:
            for i in range(1, species_num):
                path_max_barrier = max(path_max_barrier, (path_energy_list[i]-path_energy_list[i-1])*__au2kcalmol)
                if path_max_barrier > __MAX_BARRIER_IN_SELECT_NEXT_REACTANT:
                    break
                if species_name_list[i] == 'Min' or species_name_list[i] == 'MinCI':
                    valid_path_info_list.append({'max_barrier': path_max_barrier,
                                                'target_index': i,
                                                'path_info': path_info})
                    break
        """
        if 'Min' in species_name_list or 'MinCI' in species_name_list:
            path_max_barrier = None
            path_max_barrier = (max(path_energy_list) - path_energy_list[0]) * __au2kcalmol
            if path_max_barrier > __MAX_BARRIER_IN_SELECT_NEXT_REACTANT:
                continue
            for i in range(species_num-1, 0, -1):
                if species_name_list[i] in ['Min', 'MinCI']:
                    valid_path_info_list.append({'max_barrier': path_max_barrier,
                                                'target_index': i,
                                                'path_info': path_info})
                    break
        """
        """
            for i in range(1, species_num):
                # calculate energy barrier
                barrier = (path_energy_list[i] - path_energy_list[i-1]) * __au2kcalmol
                if path_max_barrier is None:
                    path_max_barrier = barrier
                else:
                    path_max_barrier = max(path_max_barrier, barrier)
                # check energy barrier
                if path_max_barrier > __MAX_BARRIER_IN_SELECT_NEXT_REACTANT:
                    break
                if species_name_list[i] in ['Min', 'MinCI']:
                    # this species is a good new reactatn
                    valid_path_info_list.append({'max_barrier': path_max_barrier,
                                                 'target_index': i,
                                                 'path_info': path_info})
                    break
        """
    if len(valid_path_info_list) == 0:
        logging.info('No valid path.')
        return None
    # find best path and new reactant
    barrier_list = [path_info['max_barrier'] for path_info in valid_path_info_list]
    min_barrier_index = np.argmin(barrier_list)
    best_path_info = valid_path_info_list[min_barrier_index]
    barrier = best_path_info['max_barrier']
    next_path_info = best_path_info['path_info']
    next_reactant_info = result_info_list[next_path_info['result_index_list'][best_path_info['target_index']]]

    logging.info(f"Best index: {next_reactant_info['result_index']}")
    logging.info(f'Path result index: ' + ' '.join([str(item) for item in next_path_info['result_index_list']]))
    logging.info(f'Path unique index: ' + ' '.join([str(item) for item in next_path_info['unique_index_list']]))
    logging.info(f'Path species name: ' + ' '.join([item for item in next_path_info['species_name_list']]))
    logging.info(f'Path energy (a.u.): ' + ' '.join([str(item) for item in next_path_info['path_energy_list']]))
    logging.info(f'Max barrier (kcal/mol): {barrier}')
    return (next_reactant_info, barrier)


def get_all_MECI():
    """
    Read MECI structure
    """
    # read coordinates
    ele, coord_list, second_line_list = read_xyz(unique_coord_filename, return_all_elements=False)
    MECI_coord_list = []
    MECI_index = []
    for i in range(len(coord_list)):
        second_data = second_line_list[i].strip().split()
        if len(second_data) >= 4:
            if second_data[2] == "MECI" and second_data[3] == '0':
                # MECI structure
                MECI_coord_list.append(coord_list[i])
                MECI_index.append(i)
    return (ele, MECI_coord_list, MECI_index)


def get_all_MECI_in_path():
    """
    Read MECI structure in main reaction paths.
    """
    # read coordinates
    ele, coord_list, second_line_list = read_xyz(unique_coord_filename, return_all_elements=False)
    reaction_file_index_list, reactant_index_list, product_index_list = read_reaction_net(reaction_net_filename)
    MECI_coord_list = []
    MECI_index = []
    for i in range(len(coord_list)):
        second_data = second_line_list[i].strip().split()
        if len(second_data) >= 4:
            if second_data[2] == "MECI" and second_data[3] == '0':
                if i in reactant_index_list:
                    # MECI structure
                    MECI_coord_list.append(coord_list[i])
                    MECI_index.append(i)
    return (ele, MECI_coord_list, MECI_index)


def get_quant_num(state):
    if state > 0:
        quant_num = 2
    else:
        quant_num = 1
    return quant_num


def run_network_expansion(cwd, cwd_state, mult, state, state_name, ele_total, p_total, quant_num, n_proc, copy_filename_list=[], qm_atom_index=[], freeze_index=[], iter=None, max_iter=None):
    os.chdir(cwd_state)
    interface_list, main_interface_index, state_name_list = init_interface(state, mult, qm_atom_index, ele_total)
    tmp_interface = interface_list[main_interface_index]
    state_index = sum([interface_list[i].quant_num for i in range(main_interface_index)]) + interface_list[main_interface_index].get_state_index(state)
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
        analy_result_file(filename_index=[0], state_index=state_index, reactant_index=None, write_mode='w', write_net=False, log_info=False)
    os.chdir(cwd)
    init_logging('run')
    logging.info(f'=============== Current State: {state_name} Iter: {iter} / {max_iter}===============')
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
    mm_atom_index = [i for i in range(len(p_reshape)) if i not in qm_atom_index]
    ele_qm = [ele_total[index] for index in qm_atom_index]
    p_qm = p_reshape[qm_atom_index].flatten()
    ele_mm = [ele_total[index] for index in mm_atom_index]
    p_mm = p_reshape[mm_atom_index].flatten()

    # read hash list
    react_adj = generate_adj_mat(ele_qm, p_qm)
    logging.info(f"Prepare reactant, index: {next_info['result_index']}")
    # generate product matrix
    unique_info_list = read_unique_coord(unique_coord_filename, unique_hash_filename)
    hash_list = [info_dict['hash_str'] for info_dict in unique_info_list]
    logging.info('Generate product matrix.')
    product_adj_list, product_hash_list = graph_generator(ele_qm, react_adj, hash_list, b=__BREAKING_NUM, f=__FORMATION_NUM, max_frag_num=__MAX_FRAG, max_radical_num=__MAX_RADICAL, nproc=n_proc)
    # product_adj_list = product_adj_list[:n_proc]
    # product_hash_list = product_hash_list[:n_proc]
    # product_adj_list, product_hash_list = graph_generator_with_input(ele, react_adj, hash_list, [[[13, 10, 1], [12, 13, 0]]])
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
            # logging.info(f'Add job:' + ' '.join([str(i+file_start_index) for i in range(len(product_adj_list))]))
            logging.info(f'Add job:' + ' '.join([str(file_start_index), '--', str(file_start_index + len(product_adj_list) - 1)]))
            logging.info(f'Waitting job finish...')
            work_pool.close()
            work_pool.join()
            logging.info(f'{state_name} All job finished.')
        # analysis result
        filename_index = [i+file_start_index for i in range(len(product_adj_list))]
        analy_result_file(filename_index=filename_index, state_index=state_index, reactant_index=next_info['result_index'], write_mode='a')
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


def run_decay_optimization(cwd_state, state, n_proc):
    """
    Minimize MECI at lower state. Select MECI on the main path.
    """
    # get MECI index, optimize state-1
    if state > 0:
        decay_state = state-1
        logging.info(f'Lower state: {decay_state}')
        work_pool = Pool(processes=n_proc)
        ele, MECI_coord_list, MECI_index_list = get_all_MECI_in_path()
        try:
            logging.info(f'S{decay_state} Start MECI decay.')
            decay_path = os.path.join(cwd_state, 'decay')
            if not os.path.exists(decay_path):
                os.mkdir(decay_path)
            os.chdir(decay_path)

            # read already optimized MECI-min
            if os.path.exists(os.path.join(decay_path, decay_traj_filename)):
                _, decay_p_list, p_type, finished_flag, decay_from_index, energy_list = read_decay_structure(os.path.join(cwd_state, 'decay', decay_traj_filename))
                index_list = [i for i in range(len(MECI_index_list)) if MECI_index_list[i] not in decay_from_index]
                # decay_p_list = [decay_p_list[index] for index in index_list]
                # p_type = [p_type[index] for index in index_list]
                # finished_flag = [finished_flag[index] for index in index_list]
                MECI_coord_list = [MECI_coord_list[index] for index in index_list]
                MECI_index_list = [MECI_index_list[index] for index in index_list]

            # read already optimized MECI-min file
            all_decay_file_list = os.listdir(decay_path)
            decay_start_index = 1
            for decay_filename in all_decay_file_list:
                if os.path.isdir(decay_filename) and decay_filename[:6] == 'decay_':
                    decay_start_index += 1

            for i in range(len(MECI_coord_list)):
                job_index = i + decay_start_index
                job_filename = 'decay_' + str(job_index)
                work_pool.apply_async(func=work_MECI_min, args=(decay_path, job_filename, copy_filename_list, ele, MECI_coord_list[i], decay_state, MECI_index_list[i]))
            logging.info(f'Add decay job ' + ' '.join(str(i+decay_start_index) for i in range(len(MECI_coord_list))))
            logging.info(f'Waitting decay job finish...')
            work_pool.close()
            work_pool.join()
            logging.info('Decay job finished.')
            # read decay structure
            decay_p_list = []
            decay_second_line_list = []
            for i in range(len(MECI_coord_list)):
                job_index = i + decay_start_index
                job_filename = 'decay_' + str(job_index)
                if os.path.exists(os.path.join(job_filename, 'Minimum.xyz')):
                    _, decay_p, second_line = read_xyz(os.path.join(job_filename, 'Minimum.xyz'), return_all_elements=False)
                elif os.path.exists(os.path.join(job_filename, 'MECI.xyz')):
                    _, decay_p, second_line = read_xyz(os.path.join(job_filename, 'MECI.xyz'), return_all_elements=False)
                else:
                    continue
                decay_p_list.append(decay_p[0])
                decay_second_line_list.append(second_line[0])
            if os.path.exists(os.path.join(decay_path, decay_traj_filename)):
                write_mode = 'a'
            else:
                write_mode = 'w'
            write_xyz(decay_traj_filename, [ele], decay_p_list, decay_second_line_list, mode=write_mode)
            
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
    else:
        logging.info('Ground state, skip decay.')
    os.chdir(cwd_state)
    logging.shutdown()


def main_state_expansion(copy_filename_list=[]):
    # start_time = time.time()
    job_type = 'MNDO'
    job_type = 'XTB'
    cwd = os.getcwd()
    generate_logging(log_filename)
    n_proc = 48
    input_file = r'input.xyz'
    ele, p_list, _ = read_xyz(input_file)
    p = p_list[0]
    atom_index = [i for i in range(len(p.reshape(-1, 3)))]
    # S1 S0 T1
    # state_list = [(1, 1), (1, 0), (3, 0)]
    # state_name_list = ['S1', 'S0', 'T0']
    if job_type == 'MNDO':
        state_list = [(1, 1)]
        state_name_list = ['S1']
    elif job_type == 'XTB':
        state_list = [(3, 0)]
        state_name_list = ['T1']
    # state_list = [(1, 0)]
    # state_name_list = ['S0']
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
                                                  iter=iter_num, max_iter=__MAX_ITERATION)
            if iter_num >= __MAX_ITERATION:
                continue_flag = False
            # end_time = time.time()
            # print('time:', end_time-start_time)


def GMD_test(copy_filename_list=[]):
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
    state_name_list = ['S1']
    state_name = state_name_list[0]
    mult, state = (1, 1)
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
        analy_result_file(filename_index=[0], interface=tmp_interface, reactant_index=None, write_mode='w', write_net=False, log_info=False)
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

    # fulvene
    bond_list = [[[11, 9, 1], [11, 10, 0], [10, 5, 1], [9, 5, 0]]]
    # styrene
    bond_list = [[[3, 4, 1], [1, 3, 0]]]
    # butadiene
    bond_list = [[[1, 8, 1]]]
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
        analy_result_file(filename_index=filename_index, interface=tmp_interface, reactant_index=next_info['result_index'], write_mode='a')
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


def test_select_next_reactant():
    init_logging('debug')
    path = r'/home/liuyq/gaussian/graph_driven/example/pyruvic/state_S1'
    os.chdir(path)
    select_next_reactant()


def test_analy():
    tmp_interface = init_interface(2)
    tmp_interface.set_quant_num(2)
    os.chdir(r'/home/liuyq/gaussian/graph_driven/test_new_code/state_2')
    filename_index = [i+21 for i in range(20)]
    analy_result_file(filename_index=filename_index, interface=tmp_interface, reactant_index=21, write_mode='w', write_net=False, log_info=False)



if __name__ == "__main__":
    main_state_expansion()
    # GMD_test()