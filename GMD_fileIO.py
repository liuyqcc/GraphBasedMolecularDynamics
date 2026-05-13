import os
import copy
import numpy as np
import logging
from collections import defaultdict
from potential.IOfunction import read_xyz, write_xyz

print_level = 1
# result coordinates
result_filename = 'result_coord.xyz'
# result hash + SPRINT coordinate
result_info_filename = 'hash_info.log'
# unique hash + SPRINT coordinate
result_unique_filename = 'hash_unique.log'
# corresponding unique coordinates
unique_coord_filename = 'unique_coord.xyz'
# reaction network file
reaction_net_filename = 'net.log'
# expected Hash
expected_hash_filename = 'hash.log'
# reaction matrix
reaction_matrix_filename = 'reaction_mat.log'
# path energies
path_energy_filename = 'path_energy.log'
# decay strucutre filename
decay_traj_filename = 'decay_coord.xyz'
# job filename head
job_filename_head = 'test_'
# optimization output filename
opt_result_filename = 'opt.xyz'

def generate_logging(log_filename):
    init_logging(log_filename=log_filename, mode='w')
    logging.shutdown()


def init_logging(log_filename, mode='a'):
    logging.getLogger().handlers.clear()
    logging.basicConfig(
    filename=f'{log_filename}.log',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode=mode
    )
    # logging.info(f'Start logging.')


#### Reaction Matrix ####
def write_reaction_mat(react_adj, product_adj):
    with open(reaction_matrix_filename, 'w') as f:
        f.writelines(f'matrix shape: {react_adj.shape[0]}\n')
        f.writelines('reactant ' + ' '.join([str(item) for item in react_adj.flatten()]) + '\n')
        f.writelines('product ' + ' '.join([str(item) for item in product_adj.flatten()]) + '\n')


def read_reaction_mat(path):
    with open(os.path.join(path, reaction_matrix_filename), 'r') as f:
        filelines = f.readlines()
    n = int(filelines[0].strip().split()[2])
    react_mat = filelines[1].strip().split()[1:]
    react_mat = np.array([int(item) for item in react_mat], np.int32).reshape(n, n)
    product_mat = filelines[2].strip().split()[1:]
    product_mat = np.array([int(item) for item in product_mat], np.int32).reshape(n, n)
    return (react_mat, product_mat)


#### Hash String ####
def write_hash(filename, hash_info_list, mode='w'):
    if not os.path.exists(filename):
        mode = 'w'
    with open(filename, mode) as f:
        for info_dict in hash_info_list:
            result_index = info_dict['result_index']
            hash_str = info_dict['hash_str']
            f.writelines(f'{result_index} {hash_str}\n')


def read_hash(filename):
    hash_info_list = []
    with open(filename, 'r') as f:
        filelines = f.readlines()
    hash_info_list = []
    for line in filelines:
        data = line.strip().split()
        if len(data) != 0:
            result_index = int(data[0])
            hash_str = data[1]
            info_dict = {'result_index': result_index,
                         'hash_str': hash_str}
            hash_info_list.append(info_dict)
    return hash_info_list


#### Unique Hash String ####
def write_unique_hash(filename, unique_hash_info_list, mode='w'):
    with open(filename, mode) as f:
        for info_dict in unique_hash_info_list:
            unique_index = info_dict['unique_index']
            finished_flag = info_dict['finished_flag']
            repeat_index = info_dict['repeat_index']
            hash_str = info_dict['hash_str']
            f.writelines(f'{unique_index} {finished_flag} {" ".join(str(item) for item in repeat_index)} : {hash_str}\n')


def read_unique_hash(filename):
    """
    Read unique hash file.

    Returns
    -------
    unique_hash_info_list: list
        [{'unique_index': int,
          'success_flag': int(0 or 1),
          'repeat_index': list[int, ...]},
          'hash_str': str
          }, ...]
    """
    success_flag = []
    unique_hash_info_list = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            filelines = f.readlines()
            for i in range(len(filelines)):
                info_dict = {}
                index_str, hash_str = filelines[i].strip().split(':')
                index_data = [int(item) for item in index_str.strip().split()]
                unique_index = int(index_data[0])
                success_flag = int(index_data[1])
                repeat_index = [int(item) for item in index_data[2:]]
                info_dict['unique_index'] = unique_index
                info_dict['success_flag'] = success_flag
                info_dict['repeat_index'] = repeat_index
                info_dict['hash_str'] = hash_str.strip().split()[0]
                unique_hash_info_list.append(info_dict)
    return unique_hash_info_list


#### Path Information ####
def write_path_energy(filename, path_species_name_list, all_coord_index, path_energy_list, reactant_energy):
    with open(filename, 'w') as f:
        f.writelines(f'reactant_energy: {reactant_energy}\n')
        f.writelines('species: ' + ' '.join([item for item in path_species_name_list]) + '\n')
        f.writelines('path_index: ' + ' '.join([str(item) for item in all_coord_index]) + '\n')
        f.writelines('estimate_energy: ' + ' '.join([str(item) for item in path_energy_list]) + '\n')


def read_path_energy(filename):
    with open(filename, 'r') as f:
        filelines = f.readlines()
    reactant_energy_line = filelines[0].strip().split()
    species_line = filelines[1].strip().split()
    path_line = filelines[2].strip().split()
    energy_line = filelines[3].strip().split()
    reactant_energy = float(reactant_energy_line[1])
    species = [item for item in species_line[1:]]
    coord_index = [int(item) for item in path_line[1:]]
    energy_list = [float(item) for item in energy_line[1:]]
    return (reactant_energy, species, coord_index, energy_list)


#### Path Information ####
def write_reaction_net(filename, state_index, result_info_list, reactant_info, mode='w'):
    if reactant_info is None:
        net_str_list = []
    else:
        react_unique_index = reactant_info['unique_index']
        react_species_name = reactant_info['species_name']
        react_index = reactant_info['result_index']
        if len(reactant_info['energy_list']) > state_index:
            energy = reactant_info['energy_list'][state_index]
        else:
            energy = reactant_info['energy_list'][0]
        # reactant_file_index = reactant_info['file_index']
        # net_str_list = [f'Reactant {react_unique_index} | {react_species_name} | {react_index} | {reactant_file_index} | {energy}']
        net_str_list = [f'Reactant {react_unique_index} | {react_species_name} | {react_index} | {energy}']
        # result structure in each path
        file_index_list = [info_dict['file_index'] for info_dict in result_info_list]
        path_index_list = defaultdict(list)
        for i, index in enumerate(file_index_list):
            path_index_list[index].append(i)
        # write reactant_index 
        for react_file_index, product_index_list in path_index_list.items():
            # unique index
            tmp_str_list = []
            tmp_str_list.append(' '.join(str(result_info_list[index]['unique_index']) for index in product_index_list))
            tmp_str_list.append(' '.join(str(result_info_list[index]['species_name']) for index in product_index_list))
            tmp_str_list.append(' '.join(str(result_info_list[index]['result_index']) for index in product_index_list))
            tmp_energy_list = []
            for index in product_index_list:
                if len(result_info_list[index]['energy_list']) > state_index:
                    tmp_energy_list.append(result_info_list[index]['energy_list'][state_index])
                else:
                    tmp_energy_list.append(result_info_list[index]['energy_list'][0])
            # tmp_str_list.append(' '.join(str(result_info_list[index]['energy_list'][state_index]) for index in product_index_list))
            tmp_str_list.append(' '.join([str(item) for item in tmp_energy_list]))
            tmp_str = f'{react_file_index} ' + ' | '.join(tmp_str_list)
            net_str_list.append(tmp_str)
    if not os.path.exists(filename):
        with open(filename, mode='w') as f:
            for line in net_str_list:
                f.writelines(line + '\n')
    else:
        with open(filename, mode=mode) as f:
            for line in net_str_list:
                f.writelines(line + '\n')


def read_reaction_net(filename):
    """
    Read reaction path from file.
    """
    with open(filename, 'r') as f:
        filelines = f.readlines()
    net_info_list = []
    for line in filelines:
        data = line.strip().split()
        if len(data) != 0:
            path_info_dict = {}
            if 'Reactant' in line:
                current_reactant_info_dict = {'unique_index': int(data[1]),
                                               'species_name': str(data[3]),
                                               'result_index': int(data[5]),
                                               'energy': float(data[7])}
                react_file_index = None
                unique_index_list = []
                species_name_list = []
                result_index_list = []
                product_energy_list = []
            else:
                split_data = line.strip().split('|')
                first_part = split_data[0].split()
                react_file_index = int(first_part[0])
                unique_index_list = [int(item) for item in first_part[1:]]
                species_name_list = [str(item) for item in split_data[1].split()]
                result_index_list = [int(item) for item in split_data[2].split()]
                product_energy_list = [float(item) for item in split_data[3].split()]
            path_info_dict['file_index'] = react_file_index
            path_info_dict['unique_index_list'] = [current_reactant_info_dict['unique_index']] + unique_index_list
            path_info_dict['species_name_list'] = [current_reactant_info_dict['species_name']] + species_name_list
            path_info_dict['result_index_list'] = [current_reactant_info_dict['result_index']] + result_index_list
            path_info_dict['path_energy_list'] = [current_reactant_info_dict['energy']] + product_energy_list
            net_info_list.append(path_info_dict)
    return net_info_list


#### Result Information ####
def read_result_coord(coord_filename, hash_filename):
    """
    Read infomation from the result_coord.xyz file.
    """
    parsers, parsers_key_list = __get_parsers(data_type=1)
    result_info_list = read_result_with_parsers(coord_filename, parsers)
    # read hash
    hash_info_list = read_hash(hash_filename)
    for i, info_dict in enumerate(result_info_list):
        corresponding_hash_info = hash_info_list[i]
        info_dict['hash_str'] = corresponding_hash_info['hash_str']
    return result_info_list


def write_result_coord(result_info_list, result_coord_filename, result_hash_filename, mode='w'):
    parsers, parsers_key_list = __get_parsers(data_type=1)
    coord_list = []
    ele_list = []
    second_line_list = []
    for info_dict in result_info_list:
        coord_list.append(info_dict['coord'])
        ele_list.append(info_dict['ele'])
        second_line = [info_dict[key] for key in parsers_key_list if key != 'energy_list']
        second_line.extend([item for item in info_dict['energy_list']])
        second_line_list.append(' '.join([str(item) for item in second_line]))
    write_xyz(result_coord_filename, ele_list, coord_list, second_line_list, mode=mode)
    write_hash(result_hash_filename, result_info_list, mode=mode)


#### Unique Information ####
def read_unique_coord(coord_filename, hash_filename):
    """
    Read result coordinates and hash.
    """
    parsers, parsers_key_list = __get_parsers(data_type=2)
    result_info_list = read_result_with_parsers(coord_filename, parsers)
    # also read hash from unique hash file
    unique_hash_info_list = read_unique_hash(hash_filename)
    for i, info_dict in enumerate(result_info_list):
        corresponding_unique_hash_info = unique_hash_info_list[i]
        info_dict['repeat_index'] = corresponding_unique_hash_info['repeat_index']
        info_dict['hash_str'] = corresponding_unique_hash_info['hash_str']
    return result_info_list


def write_unique_coord(unique_info_list, unique_coord_filename, unique_hash_filename, mode='w'):
    """
    Write unique coordinates and unique hash.
    """
    parsers, parsers_key_list = __get_parsers(data_type=2)
    coord_list = []
    ele_list = []
    second_line_list = []
    hash_list = []
    for info_dict in unique_info_list:
        coord_list.append(info_dict['coord'])
        ele_list.append(info_dict['ele'])
        second_line = [info_dict[key] for key in parsers_key_list if key not in ['energy_list', 'repeat_index']]
        second_line.extend([item for item in info_dict['energy_list']])
        second_line_list.append(' '.join([str(item) for item in second_line]))
        hash_list.append(info_dict['hash_str'])
    write_xyz(unique_coord_filename, ele_list, coord_list, second_line_list, mode=mode)
    write_unique_hash(unique_hash_filename, unique_info_list, mode=mode)


def write_opt_result(filename, opt_info_list, mode='w'):
    """
    Write optimized structure in .xyz format.
    """
    parsers, parsers_key_list = __get_parsers(data_type=0)
    coord_list = []
    second_line_list = []
    ele_list = []
    for info_dict in opt_info_list:
        coord_list.append(info_dict['coord'])
        ele_list.append(info_dict['ele'])
        second_line = [info_dict[key] for key in parsers_key_list if key != 'energy_list']
        second_line.extend([item for item in info_dict['energy_list']])
        second_line_list.append(' '.join([str(item) for item in second_line]))
    write_xyz(filename, ele_list, coord_list, second_line_list, mode=mode)


def read_opt_result(filename):
    """
    Read optimization result from the single GMD job file.
    """
    parsers, parsers_key_list = __get_parsers(data_type=0)
    result_info_list = read_result_with_parsers(filename, parsers)
    return result_info_list


def read_result_with_parsers(filename, parsers):
    """
    Read optimization result using parser format.
    """
    result_info_list = []
    # read structures
    if os.path.exists(filename):
        result = read_xyz(filename)
        if result is not None:
            ele, p_list, second_line_list = result
            for i in range(len(p_list)):
                data = second_line_list[i].strip().split()
                info_dict = {
                    'ele': ele,
                    'coord': p_list[i]
                }
                for key, func in parsers:
                    info_dict[key] = func(data)
                result_info_list.append(info_dict)
    else:
        logging.info('No such file:' + filename)
    return result_info_list


def __get_parsers(data_type):
    parsers = []
    parsers_key_list = []
    offset = 0
    if data_type >= 2:
        # add unique index
        parsers_key_list.append('unique_index')
        parsers.append((parsers_key_list[-1], lambda d, i=offset: int(d[i])))
        parsers_key_list.append('repeat_index')
        parsers.append((parsers_key_list[-1], lambda d, i=offset: int(d[i])))
        offset += 1
    if data_type >= 1:
        # add global result index
        parsers_key_list.append('result_index')
        parsers.append((parsers_key_list[-1], lambda d, i=offset: int(d[i])))
        offset += 1
    basic_parsers = [
        ('file_index',    lambda d: int(d[offset+0])),
        ('species_name',  lambda d: str(d[offset+1])),
        ('finished_flag', lambda d: int(d[offset+2])),
        ('state',         lambda d: int(d[offset+3])),
        ('path_index',    lambda d: int(d[offset+4])),
        ('index_time',    lambda d: int(d[offset+5])),
        ('index_geom',    lambda d: int(d[offset+6])),
        ('energy_list',   lambda d: [float(item) for item in d[offset+7:]])
    ]
    parsers_key_list.extend([item[0] for item in basic_parsers])
    parsers.extend(basic_parsers)
    return (parsers, parsers_key_list)

