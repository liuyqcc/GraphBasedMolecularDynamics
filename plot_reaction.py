import os
import numpy as np
import matplotlib.pyplot as plt
from analy.read_traj import read_traj
from bond_connect import generate_adj_mat
from read_net_energy import read_reaction_net_energy, reaction_net_energy_filename


def read_reaction_net(filename):
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
            else:
                reaction_file_index_list.append(int(data[0]))
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


def mol_from_adj(ele, adj_mat):
    from rdkit import Chem
    mol = Chem.RWMol()
    index_list = []
    for i, item in enumerate(ele):
        atom = Chem.Atom(item)
        # label = item + str(i+1)
        label = item
        atom.SetProp('atomLabel', label)
        atom_index = mol.AddAtom(atom)
        index_list.append(atom_index)
    adj_mat_up = np.triu(adj_mat, k=1)
    bond_index = np.where(adj_mat_up != 0)
    for i, j in zip(bond_index[0], bond_index[1]):
        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    return mol.GetMol()

