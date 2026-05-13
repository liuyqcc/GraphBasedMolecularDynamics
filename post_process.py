import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from graph_tools import get_bond_from_mat
from bond_connect import get_radius_table


def read_data(filename, quant_num):
    """
    Read energy and state.
    """
    with open(filename, 'r') as f:
        filelines = f.readlines()
    state_index_list = []
    energy_list = []
    geom_list = []
    data = filelines[0].strip().split()
    # ES1 -> S1
    state_name_list = [item[1:] for item in data[2:2+quant_num]]
    for line_str in filelines[1:]:
        data = line_str.strip().split()
        if len(data) != 0:
            state_index_list.append(state_name_list.index(data[1]))
            e_list = [float(item) for item in data[2:2+quant_num]]
            energy_list.append(e_list)
            geom = [float(item) for item in data[quant_num+5:]]
            geom_list.append(geom)
    energy_list = np.array(energy_list).T
    geom_list = np.array(geom_list).T
    return (state_name_list, state_index_list, energy_list, geom_list)


def minimum_filter(energy_list, x_list, half_step_window=0.1):
    new_e_list = []
    window = deque()
    correspond_index = []
    N = len(x_list)
    dr = half_step_window
    left_index = 0
    right_index = 0

    for i in range(N):
        center = x_list[i]
        # move right side
        while right_index < N and abs(x_list[right_index]-center) <= dr:
            window.append(energy_list[right_index])
            right_index += 1
        # move left side
        while left_index < i and abs(x_list[left_index]-center) > dr:
            window.popleft()
            left_index += 1
        min_index = np.argmin(window) + left_index
        new_e_list.append(energy_list[min_index])
        correspond_index.append(min_index)
        # new_e_list.append(max(window))
    return (new_e_list, correspond_index)


def scan_minimum_TS_old(filtered_energy_list, relax_step=100, de_threhold=5):
    """
    Find local minimum and TS from filtered energy profile.

    Parameters
    ----------
    filtered_energy_list: list
        Energy list processed by minimum_filter.
    relax_step: int (default: 100)
        Relax step in MD simulation.
    de_threshold: float (default: 5)
        Max energy ignored in searching (kcal/mol).

    Returns
    -------
    minimum_coord_list: list
        Local minimum coordinates.
    TS_coord_list: list
        Transition state coordinates.
    """
    energy_list = filtered_energy_list
    de_max = max(de_threhold, 0) / 627.5095
    minimum_coord_list = []
    TS_coord_list = []
    min_energy = energy_list[0]
    max_energy = min_energy
    minium_coord = 0
    TS_coord = 0
    # search 0: local minimum or 1: TS (local maximum)
    search_target = 0
    for i in range(1, len(energy_list)):
        if search_target == 0:
            # search local minimum
            if energy_list[i] - min_energy < de_max:
                if min_energy > energy_list[i]:
                    # energy lower, new local minima
                    min_energy = energy_list[i]
                    minium_coord = i
                if i == len(energy_list)-1:
                    # the final structure may be a local minimum
                    if len(minimum_coord_list) == 0 or len(energy_list) - minimum_coord_list[-1] > relax_step:
                        minimum_coord_list.append(minium_coord)
            else:
                # energy raise, turn to find TS
                minimum_coord_list.append(minium_coord)
                search_target = 1
                max_energy = energy_list[i]
                TS_coord = i
        elif search_target == 1:
            # search TS
            if max_energy - energy_list[i] < de_max:
                if max_energy < energy_list[i]:
                    # energy raise, new TS
                    max_energy = energy_list[i]
                    TS_coord = i
            else:
                # energy lower, turn to find local minimum
                TS_coord_list.append(TS_coord)
                search_target = 0
                min_energy = energy_list[i]
                minium_coord = i
    return (minimum_coord_list, TS_coord_list)


def scan_minimum_TS(filtered_energy_list, de_threhold=3, end_with_minimum=True):
    """
    Find local minimum and TS from filtered energy profile.

    Parameters
    ----------
    filtered_energy_list: list
        Energy list processed by minimum_filter.
    de_threshold: float (default: 5)
        Max energy ignored in searching (kcal/mol).

    Returns
    -------
    minimum_coord_list: list
        Local minimum coordinates.
    TS_coord_list: list
        Transition state coordinates.
    """
    energy_list = filtered_energy_list
    de_max = max(de_threhold, 0) / 627.5095
    minimum_coord_list = []
    TS_coord_list = []
    min_energy = energy_list[0]
    max_energy = min_energy
    minium_coord = 0
    tmp_TS_coord = []
    TS_coord = 0
    # search 0: local minimum or 1: TS (local maximum)
    search_target = 0
    for i in range(1, len(energy_list)):
        if search_target == 0:
            # search local minimum
            if energy_list[i] - min_energy < de_max:
                if min_energy > energy_list[i]:
                    # energy lower, new local minima
                    min_energy = energy_list[i]
                    minium_coord = i
                if i == len(energy_list)-1 and end_with_minimum:
                    TS_coord_list.extend(tmp_TS_coord)
                    minimum_coord_list.append(minium_coord)
            else:
                # start to climb up
                # new TS is added when new local minimum is found
                TS_coord_list.extend(tmp_TS_coord)
                tmp_TS_coord = []
                # energy raise, turn to find TS
                minimum_coord_list.append(minium_coord)
                search_target = 1
                max_energy = energy_list[i]
                TS_coord = i
        elif search_target == 1:
            # search TS
            if max_energy - energy_list[i] < de_max:
                if max_energy < energy_list[i]:
                    # energy raise, new TS
                    max_energy = energy_list[i]
                    TS_coord = i
            else:
                # energy lower, turn to find local minimum
                tmp_TS_coord.append(TS_coord)
                search_target = 0
                min_energy = energy_list[i]
                minium_coord = i
    return (minimum_coord_list, TS_coord_list)


def coord_refine(org_energy_list, min_coord_list, TS_coord_list, half_time_window=20):
    # refine coord
    refine_minimum_coord = []
    refine_TS_coord = []
    for coord in min_coord_list:
        start = max(coord - half_time_window, 0)
        end = min(coord + half_time_window, len(org_energy_list))
        refine_minimum_coord.append(np.argmin(org_energy_list[start:end])+start)
    for coord in TS_coord_list:
        start = max(coord - half_time_window, 0)
        end = min(coord + half_time_window, len(org_energy_list))
        refine_TS_coord.append(np.argmin(org_energy_list[start:end])+start)
    return (refine_minimum_coord, refine_TS_coord)


def scan_MECI(org_e_list, CI_coord_list, TS_coord):
    e_list = np.array(org_e_list)
    segments = [0]
    segments.extend(TS_coord)
    segments.append(len(org_e_list)-1)
    segments = np.array(segments, np.int64)
    segments = np.sort(segments)
    MECI_coord_list = []
    # no CI
    if len(CI_coord_list) == 0:
        return []
    # find MECI between each TS
    index_list = np.searchsorted(segments, CI_coord_list, side='right')-1
    for i in range(len(segments)-1):
        in_segment = CI_coord_list[index_list == i]
        if len(in_segment) > 0:
            energy_list = e_list[in_segment]
            min_index = in_segment[np.argmin(energy_list)]
            MECI_coord_list.append(min_index)
    return MECI_coord_list


def remove_degenerate_index(min_coord_list, MECI_coord_list):
    """
    If the local minimum is a MECI, turn into MinCI.
    If the MECI is a TS, remove the MECI.

    Returns
    -------
    new_min_list: list
        Local minimum list.
    """
    new_coord_list = []
    MinCI_list = []
    for item in min_coord_list:
        if item in MECI_coord_list:
            MinCI_list.append(item)
            MECI_coord_list.remove(item)
        else:
            new_coord_list.append(item)
    return (new_coord_list, MECI_coord_list, MinCI_list)


def analy_traj(energy_list, CI_coord_list, CP_coord_list, x_list, sort_index, half_time_window, de_threshold=5):
    """
    Returns
    -------
    result_list: list[[index_time, index_geom, species_name]...]
        index in time order, index in geometry coordinate order, corresponding species name
    """
    # process the energy list
    e_filtered_list, correspond_index_geom = minimum_filter(energy_list, x_list=x_list, half_step_window=half_time_window)
    # find local minimum and TS
    min_coord_index_geom, TS_coord_index_geom = scan_minimum_TS(e_filtered_list, de_threhold=de_threshold)
    # refind the coordinate
    min_coord_index_geom = [correspond_index_geom[i] for i in min_coord_index_geom]
    TS_coord_index_geom = [correspond_index_geom[i] for i in TS_coord_index_geom]
    # local_min_coord, TS_coord = coord_refine(energy_list, local_min_coord, TS_coord, half_time_window=half_time_window)
    # find MECI coordinate
    MECI_coord_index_geom = scan_MECI(energy_list, CI_coord_list, TS_coord_index_geom)
    MECP_coord_index_geom = scan_MECI(energy_list, CP_coord_list, TS_coord_index_geom)
    # remove degenerate local minimum
    min_coord_index_geom, MECI_coord_index_geom, MinCI_coord_index_geom = remove_degenerate_index(min_coord_index_geom, MECI_coord_index_geom)
    # MECI_coord_index_geom = remove_degenerate_index(MECI_coord_index_geom, TS_coord_index_geom)

    # coordinate index in orginal order (time order)
    min_coord_index_time = [sort_index[i] for i in min_coord_index_geom]
    TS_coord_index_time = [sort_index[i] for i in TS_coord_index_geom]
    MECI_coord_index_time = [sort_index[i] for i in MECI_coord_index_geom]
    MECP_coord_index_time = [sort_index[i] for i in MECP_coord_index_geom]
    MinCI_coord_index_time = [sort_index[i] for i in MinCI_coord_index_geom]
    if 0 in min_coord_index_time:
        # remove the first point
        index = min_coord_index_time.index(0)
        min_coord_index_time.remove(0)
        min_coord_index_geom.pop(index)
    if 0 in MECI_coord_index_time:
        index = MECI_coord_index_time.index(0)
        MECI_coord_index_time.remove(0)
        MECI_coord_index_geom.pop(index)
    # use sorted index to identify order
    all_coord_index_time = min_coord_index_time + TS_coord_index_time + MECI_coord_index_time + MECP_coord_index_time + MinCI_coord_index_time
    all_coord_index_geom = min_coord_index_geom + TS_coord_index_geom + MECI_coord_index_geom + MECP_coord_index_geom + MinCI_coord_index_geom
    species_name_list = ['Min' for _ in range(len(min_coord_index_geom))] + \
                        ['TS' for _ in range(len(TS_coord_index_geom))] + \
                        ['MECI' for _ in range(len(MECI_coord_index_geom))] + \
                        ['MECP' for _ in range(len(MECP_coord_index_geom))] + \
                        ['MinCI' for _ in range(len(MinCI_coord_index_geom))]
    argsort_index = np.argsort(all_coord_index_geom)
    all_coord_index_geom = [all_coord_index_geom[index] for index in argsort_index]
    species_name_list = [species_name_list[index] for index in argsort_index]
    all_coord_index_time = [all_coord_index_time[index] for index in argsort_index]
    result_list = []
    for i in range(len(all_coord_index_time)):
        data = [all_coord_index_time[i], all_coord_index_geom[i], species_name_list[i]]
        result_list.append(data)
    return result_list


def analy_traj_on_the_fly(energy_list, x_list, sort_index, half_time_window, de_threshold=5):
    """
    Find new local minimum.


    Returns
    -------
    result_list: list[[index_time, index_geom, species_name]...]
        index in time order, index in geometry coordinate order, corresponding species name
    """
    # process the energy list
    e_filtered_list, correspond_index_geom = minimum_filter(energy_list, x_list=x_list, half_step_window=half_time_window)
    # find local minimum and TS
    min_coord_index_geom, TS_coord_index_geom = scan_minimum_TS(e_filtered_list, de_threhold=de_threshold, end_with_minimum=False)
    result_list = []
    for i in range(len(min_coord_index_geom)):
        data = [sort_index[min_coord_index_geom[i]], min_coord_index_geom[i], 'Min']
        result_list.append(data)
    return result_list


def analy_MECI_traj(energy_list):
    """
    Returns
    -------
    local_min_coord: int
        Index of the minimum.
    """
    return np.argmin(energy_list)


def analy_path_energy(diff_mat, p_list, energy_list):
    bond_index, bond_type = get_bond_from_mat(diff_mat)
    diff = np.triu(diff_mat, k=1)
    reaction_bond_index = np.where(diff != 0)
    bond_index = sorted([sorted([i, j]) for i, j in zip(reaction_bond_index[0], reaction_bond_index[1])])
    bond_length_list = []
    P = np.array(p_list).reshape(len(p_list), -1, 3)
    for i, j in bond_index:
        bond_length = np.linalg.norm(P[:, i, :] - P[:, j, :], axis=1)
        bond_length_list.append(bond_length)
    # bond_type = [diff[item] for item in bond_index]


def test_path_analy_based_time():
    k = 627.5095
    path = r'D:\tmp\data.out'
    quant_num = 2
    # state_list, energy_list, geom_list = read_data(path, quant_num)
    state_name_list, state_index_list, energy_list, geom_list = read_data(path, quant_num=quant_num)
    e_list = energy_list[1]
    reactant_energy = e_list[0]
    # e_list = (e_list - reactant_energy) * 627.5095
    e_list = (e_list - reactant_energy)
    x = [i for i in range(len(e_list))]
    e_processed_list, correspond_index = minimum_filter(e_list, x_list=x, half_step_window=10)
    # plt.plot(x, e_list, c='black')
    min_coord_list, TS_coord_list = scan_minimum_TS(e_processed_list)
    min_coord_list = [correspond_index[i] for i in min_coord_list]
    TS_coord_list = [correspond_index[i] for i in TS_coord_list]
    print(min_coord_list, TS_coord_list)
    # min_coord_list, TS_coord_list = coord_refine(e_list, min_coord_list, TS_coord_list, half_time_window=10)

    # plt.xlim(-10, 300)
    # plt.ylim(-20, 30)
    plt.margins(0)
    plt.tight_layout(pad=0)

    plt.plot(x, e_list*k, c='black', alpha=0.5)
    plt.plot(x, np.array(e_processed_list)*k, c='red')

    # for i in range(len(min_coord_list)):
    #     coord = min_coord_list[i]
    #     plt.scatter(coord, e_list[coord], c='blue')

    # for i in range(len(TS_coord_list)):
    #     coord = TS_coord_list[i]
    #     plt.scatter(coord, e_list[coord], c='green')

    # find MECI
    e_diff = energy_list[1] - energy_list[0]
    CI_coord_list = np.where(e_diff <= 5/627.5095)[0]
    MECI_coord_list = scan_MECI(e_list, CI_coord_list, TS_coord_list)
    print(MECI_coord_list)
    CP_coord_list = []


    min_coord, TS_coord, MECI_coord, all_index = analy_traj(e_list, CI_coord_list, CP_coord_list=CP_coord_list, x_list=x, sort_index=x, half_time_window=10, de_threshold=5)
    print(min_coord, TS_coord, MECI_coord)
    # exit()

    for coord in CI_coord_list:
        plt.scatter(coord, e_list[coord]*k, c='green')
    # plt.plot(x, e_diff, c='green')
    for i in range(len(min_coord)):
        coord = min_coord[i]
        plt.scatter(coord, e_list[coord]*k, c='black')
    # plt.plot(x, e_diff, c='green')
    for i in range(len(MECI_coord_list)):
        coord = MECI_coord_list[i]
        plt.scatter(coord, e_list[coord]*k, c='red')
    for i in range(len(TS_coord_list)):
        coord = TS_coord_list[i]
        plt.scatter(coord, e_list[coord]*k, c='blue')

    plt.show()
    # find MECI


def test_path_analy_based_geom():
    k = 627.5095
    path = r'D:\tmp\data.out'
    path = r'D:\Gaussian\graph_driven\test_fulvene\state_2_new\data.out'
    path = r'D:\Gaussian\graph_driven\test_fulvene\test4\state_2\test_42\data.out'
    path = r'D:\Gaussian\graph_driven\test_fulvene\test7\data.out'
    path = r'D:\Gaussian\graph_driven\test_bias\test_LastBias_20step_10h_0.1w\test_1\data.out'
    path = r'D:\Gaussian\graph_driven\test_bias\test_LastBias_20step_40h_0.2w\test_1\data.out'
    path = r'D:\Gaussian\graph_driven\test_bias\test_LastBias_20step_5h_0.2w\test_1\data.out'
    path = r'D:\Gaussian\graph_driven\test_bias\test_LastBias_20step_10h_0.2w\test_1\data.out'
    path = r'D:\Gaussian\GraphDriving\isoprene\state_2\test_44\data.out'
    path = r'D:\Gaussian\GMD\test\state_S1\test_17\data.out'
    path = r'D:\tmp\data (7).out'
    path = r'D:\Gaussian\GraphDriving\TestGMD\example_1\state_S0\test_1\data.out'
    path = r'D:\tmp\data (14).out'
    path = r'D:\Gaussian\GraphDriving\TestGMD\example_3\state_S1\test_1\data.out'
    path = r'D:\Gaussian\GraphDriving\TestGMD\example_2\state_S0\test_1\data.out'
    path = r'D:\tmp\data (15).out'
    quant_num = 1
    # state_list, energy_list, geom_list = read_data(path, quant_num)
    state_name_list, state_index_list, energy_list, geom_list = read_data(path, quant_num=quant_num)
    # geom_1_org = -geom_list[0] + geom_list[1]
    radius_table, ele_table = get_radius_table()
    ref_CH = radius_table['C'] + radius_table['H']
    ref_CC = radius_table['C'] + radius_table['C']
    # geom_1_org = geom_list[1] - (geom_list[0]-1.8*ref_CH)
    # geom_1_org = -(geom_list[0]-1.8*ref_CC) + geom_list[1] + geom_list[2] - (geom_list[3]-1.8*ref_CH)
    geom_1_org = geom_list[0]

    e_list_time = energy_list[0]
    reactant_energy = e_list_time[0]
    # e_list = (e_list - reactant_energy) * 627.5095
    e_list_time = e_list_time - reactant_energy
    x = [i for i in range(len(e_list_time))]
    # plt.plot(x, e_list, c='black')
    sort_index = np.argsort(geom_1_org)[::-1]
    geom_1_geom = geom_1_org[sort_index]
    e_list_geom = e_list_time[sort_index]
    e_processed_list, correspond_index_geom = minimum_filter(e_list_geom, x_list=geom_1_geom, half_step_window=0.1)
    min_coord_list_geom, TS_coord_list_geom = scan_minimum_TS(e_processed_list)
    # min_coord_list, TS_coord_list = coord_refine(e_list, min_coord_list, TS_coord_list, half_time_window=20)
    min_coord_list_geom = [correspond_index_geom[i] for i in min_coord_list_geom]
    TS_coord_list_geom = [correspond_index_geom[i] for i in TS_coord_list_geom]
    print(min_coord_list_geom, TS_coord_list_geom)

    # plt.xlim(-10, 600)
    # plt.ylim(-40, 50)
    # plt.ylim(-20, 30)
    # plt.ylim(0, 50)
    plt.margins(0)
    plt.tight_layout(pad=0)
    plt.gca().invert_xaxis()

    # plt.plot(x, e_list, c='black', alpha=0.5)

    # plt.plot(geom_1, e_list*k, c='black', alpha=0.5)
    e_list_time_2 = np.array(energy_list[0])
    e_list_time_2 = e_list_time_2 - reactant_energy
    e_list_geom_2 = e_list_time_2[sort_index]
    plt.plot(geom_1_geom, e_list_geom*k, c='black', alpha=0.5)
    plt.plot(geom_1_geom, e_list_geom_2*k, c='blue', alpha=0.5)
    plt.plot(geom_1_geom, np.array(e_processed_list)*k, c='red', alpha=0.5)

    # find MECI
    # e_diff = (energy_list[1] - energy_list[0]) * 627.5095
    # e_diff = e_diff[sort_index]
    e_diff = []
    # CI_coord_list = np.where(e_diff <= 5)[0]
    CI_coord_list = []
    MECI_coord_list = scan_MECI(e_list_geom, CI_coord_list, TS_coord_list_geom)
    print('MECI coord index:', MECI_coord_list)


    result_list = analy_traj(e_list_geom, CI_coord_list, [], geom_1_geom, sort_index=sort_index, half_time_window=0.1, de_threshold=1)
    for item in result_list:
        print(item)
    # min_coord = [sort_index[i] for i in min_coord]
    # TS_coord = [sort_index[i] for i in TS_coord]
    # MECI_coord = [sort_index[i] for i in MECI_coord]
    # print('E_ref:', [(energy_list[1][index]-energy_list[1][0])*627.5095 for index in TS_coord])
    # exit()
    for i in range(len(result_list)):
        index_time, index_geom, species_name = result_list[i]
        print(result_list[i])
        coord_geom = geom_1_geom[index_geom]
        if species_name == 'Min' or species_name == 'MinCI':
            color = 'orange'
        elif species_name == 'MECI':
            color = 'red'
        elif species_name == 'TS':
            color = 'blue'
        plt.scatter(coord_geom, e_list_geom[index_geom]*k, c=color)
    write_data = True
    write_data = False
    if write_data:
        with open('example_2_traj_analy.txt', 'w') as f:
            for i in range(len(geom_1_geom)):
                # f.writelines(f'{geom_1_geom[i]} {e_list_geom[i]*k} {e_processed_list[i]*k}\n')
                f.writelines(f'{geom_list[0][sort_index][i]} {geom_list[1][sort_index][i]} {geom_1_geom[i]} {e_list_geom_2[i]*k} {e_list_geom[i]*k} {e_processed_list[i]*k}\n')
            f.writelines('\n')
            for i in range(len(result_list)):
                index_time, index_geom, species_name = result_list[i]
                coord_geom = geom_1_geom[index_geom]
                f.writelines(f'{species_name} {coord_geom} {e_list_geom[i]*k}\n')

    plt.show()
    # find MECI



if __name__ == "__main__":
    # test_path_analy_based_time()
    test_path_analy_based_geom()
