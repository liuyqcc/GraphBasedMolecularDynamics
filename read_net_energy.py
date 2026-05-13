import os
import sys
import numpy as np
from analy.read_traj import read_traj, write_traj
import copy

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
# reaction energy file for post-process
reaction_net_energy_filename = 'energy_net.log'



def read_hash(filename):
    with open(filename, 'r') as f:
        filelines = f.readlines()
    hash_list = []
    for line in filelines:
        data = line.strip().split()
        if len(data) != 0:
            hash_list.append(data[0])
    return hash_list


def read_reaction_net(filename):
    """
    Read reaction path from filename

    Returns
    -------

    reaction_file_index_list: list
        Job filename index list.

    reactant_index_list: list
        Reactant index list.
    
    product_index_list: list[[int, int, ...], ...]
        Product index list
    """
    with open(filename, 'r') as f:
        filelines = f.readlines()
    reaction_file_index_list = []
    reactant_index_list = []
    product_index_list = []
    for line in filelines:
        data = line.strip().split()
        if len(data) != 0:
            if 'Reactant' in line:
                reactant_index_list.append(int(data[1]))
                product_index_list.append([])
                reaction_file_index_list.append([])
            else:
                reaction_file_index_list[-1].append(int(data[0]))
                product_index_list[-1].append([int(item) for item in data[1:]])
    return (reaction_file_index_list, reactant_index_list, product_index_list)


def read_unique_hash_info(filename):
    """
    Returns
    -------
    unique_index_list: list[[int, ...]]
        Repeated coordinate index.

    unique_hash_list: list[str...]
        Unique hash

    unique_S_list: list[numpy.array]
        Unique SPRINT coordinate
    """
    unique_hash = []
    unique_S = []
    unique_index_list = []
    success_flag = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            filelines = f.readlines()
            for i in range(len(filelines)):
                if ':' in filelines[i]:
                    index_str, hash_S_str = filelines[i].strip().split(':')
                    index_data = [int(item) for item in index_str.strip().split()]
                    hash_S_data = [item for item in hash_S_str.strip().split()]
                    unique_hash.append(hash_S_data[0])
                    unique_S.append(np.array([float(item) for item in hash_S_data[1:]]))
                    success_flag.append(index_data[1])
                    unique_index_list.append(index_data[2:])
    return (unique_index_list, success_flag, unique_hash, unique_S)


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


def write_reaction_net_energy(split_energy_list, split_reaction_path, split_species_list):
    # write data
    with open(reaction_net_energy_filename, 'w') as f:
        for i in range(len(split_energy_list)):
            # index
            f.writelines(' '.join(str(item) for item in split_reaction_path[i]) + '\n')
            # species
            f.writelines(' '.join(str(item) for item in split_species_list[i]) + '\n')
            # energies
            f.writelines(' '.join(str(item) for item in split_energy_list[i]) + '\n')


def read_reaction_net_energy(path):
    with open(path, 'r') as f:
        filelines = f.readlines()
    energy_list, reaction_path, species_list = [], [], []
    for i in range(len(filelines)):
        data = filelines[i].strip().split()
        if len(data) != 0:
            if i % 3 == 0:
                reaction_path.append([int(item) for item in data])
            elif i % 3 == 1:
                species_list.append(data)
            elif i % 3 == 2:
                energy_list.append([float(item) for item in data])
    return (energy_list, reaction_path, species_list)


def get_reation_energy():
    k = 627.5095
    # expected hash list
    # logging.info('Read hash')
    expected_hash_list = read_hash(expected_hash_filename)
    # logging.info('Read reaction net')
    # read reaction net data (find all reactant)
    reaction_file_index_list, reactant_index_list, product_index_list = read_reaction_net(reaction_net_filename)
    # read unique Hash file (find all finished product)
    # logging.info('Read unique hash')
    unique_index_list, unique_success_flag, unique_hash_list, unique_S_list = read_unique_hash_info(result_unique_filename)
    # read corresponding coordinates
    # logging.info('Read unique coordinates')
    ele, coord_list, second_line_list = read_traj(unique_coord_filename)
    species_unique_coord = [item.strip().split()[2].lower() for item in second_line_list]
    # read path energy and species
    all_species_list = []
    all_path_energy_list = []
    for i, file_index_list in enumerate(reaction_file_index_list):
        for file_index in file_index_list:
            reactant_index = reactant_index_list[i]
            reactant_energy, species_list, coord_index, path_energy = read_path_energy(os.path.join(job_filename_head + str(file_index), 'path_energy.log'))
            all_species_list.append([species_unique_coord[reactant_index]] + species_list)
            all_path_energy_list.append([reactant_energy] + path_energy)
    reactant_hash_list = [unique_hash_list[i] for i in reactant_index_list]
    all_hash = []
    all_hash.extend(expected_hash_list)
    for i in range(len(unique_hash_list)):
        if unique_hash_list[i] not in all_hash:
            all_hash.append(unique_hash_list[i])

    # best first search
    all_reactant = []
    all_reaction_path = []
    for i in range(len(reactant_hash_list)):
        reactant = reactant_index_list[i]
        all_reactant.append(reactant)
        all_reaction_path.extend([[reactant] + product_list for product_list in product_index_list[i]])

    # replace all reactant with 'r' in all_species_list
    for i in range(len(all_species_list)):
        species_list = all_species_list[i]
        for j in range(len(species_list)):
            if species_list[j] == 'ts':
                all_reaction_path[i].insert(j, -1)
    # split reaction path
    split_reaction_path = []
    split_energy_list = []
    split_species_list = []
    for i in range(len(all_species_list)):
        reaction_path_index = all_reaction_path[i]
        species_list = all_species_list[i]
        start_index = 0
        # identify reactant in path
        # reactant_in_path_index = [j for j in range(len(reaction_path_index)) if reaction_path_index[j] in reactant_index_list]
        # identify MECI and MIN in path
        reactant_in_path_index = [j for j in range(len(reaction_path_index)) if species_list[j].lower() in ['min', 'meci']]
        N = len(reactant_in_path_index)
        for j in range(N):
            start_index = reactant_in_path_index[j]
            if j == N-1:
                end_index = len(species_list)
            else:
                end_index = reactant_in_path_index[j+1] + 1
            
            if reaction_path_index[start_index] == reaction_path_index[end_index-1]:
                continue

            # split reactions
            slice_species = species_list[start_index:end_index]
            split_species_list.append(slice_species)
            split_reaction_path.append(all_reaction_path[i][start_index:end_index])
            tmp_energy_list = all_path_energy_list[i][start_index:end_index]
            split_energy_list.append(tmp_energy_list)

    # write data
    with open(reaction_net_energy_filename, 'w') as f:
        for i in range(len(split_energy_list)):
            # index
            f.writelines(' '.join(str(item) for item in split_reaction_path[i]) + '\n')
            # species
            f.writelines(' '.join(str(item) for item in split_species_list[i]) + '\n')
            # energies
            f.writelines(' '.join(str(item) for item in split_energy_list[i]) + '\n')


if __name__ == "__main__":
    get_reation_energy()