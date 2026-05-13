"""
Check bond
"""

import os
import numpy as np


def get_radius_table():
    """
    Covalent radius table
    """
    radius = {
        'H': 0.4, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
        'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
        'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
        'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39,
        'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
        'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16,
        'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
        'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
        'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4,
        'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
        'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
        'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
        'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36,
        'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48
    }
    ele_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi']
    return (radius, ele_table)


def generate_distance_matrix(coord):
    """
    Generate disntance matrix
    """
    coord = coord.reshape(-1, 3)
    num = coord.shape[0]
    dis_mat = np.zeros((num, num), 'f')
    tmp_coord = np.zeros(coord.shape, 'f')
    for i in range(num):
        tmp_coord = coord - coord[i]
        dis_mat[i] = np.linalg.norm(tmp_coord, axis=1)
        dis_mat[i, i] = 0
    return dis_mat


def generate_bond_length_matrix(element):
    """
    Generate max bonding length.
    """
    radius, ele_table = get_radius_table()
    num = len(element)
    length_matrix = np.zeros((num, num), 'f')
    radius_list = np.array([radius[ele.capitalize()] for ele in element], 'f')
    for i in range(num):
        length_matrix[i] = (radius_list + radius_list[i]) * 1.3
    return length_matrix


def get_atomic_number(element):
    """
    Elements to atomic number
    """
    radius, ele_table = get_radius_table()
    ele_num_list = []
    for item in element:
        ele_num_list.append(ele_table.index(item.capitalize())+1)
    return ele_num_list


def generate_adj_mat(element, coord):
    """
    Generate bonding matrix from corodinates (unit: Angs)

    Returns
    -------
    bond_mat: numpy.array
        Bonding matrix, 1 for bond and 0 for nonbond
    """
    dis_mat = generate_distance_matrix(coord)
    len_mat = generate_bond_length_matrix(element)
    adj_mat = np.where(dis_mat <= len_mat, 1, 0)
    # not bonding ifself
    np.fill_diagonal(adj_mat, 0)
    return adj_mat


def get_bond_table(adj_mat):
    bond_mat = np.triu(adj_mat, k=1)
    bond_index = np.where(bond_mat == 1)
    bond_table = [[bond_index[0][i], bond_index[1][i]] for i in range(len(bond_index[0]))]
    return bond_table


def get_target_distance(ele, index_list, type_list):
    """
    Distance is given in Angs.
    """
    radius, ele_table = get_radius_table()
    r_list = []
    for i in range(len(index_list)):
        i1, i2 = index_list[i]
        ele1 = ele[i1]
        ele2 = ele[i2]
        r1 = radius[ele1]
        r2 = radius[ele2]
        if type_list[i] == 1:
            r = r1 + r2
        elif type_list[i] == -1:
            r = (r1 + r2) * 1.8
        else:
            print('Unexpected type:', type_list[i])
            return
        r_list.append(r)
    return r_list


def get_SPRINT_coord(ele, coord):
    """
    Generate SPRINT coordinates.

    Returns
    S: numpy.array (1-D)
        sqrt(num) * eig_val * eig_vec
    -------
    """
    p = coord.reshape(-1, 3)
    num = len(p)
    distance_mat = np.zeros((num, num), dtype=np.float64)
    reference_mat = np.zeros_like(distance_mat)
    radius, ele_table = get_radius_table()
    ref_radius = np.array([radius[item.capitalize()] for item in ele], 'f')
    alpha_mat = np.zeros_like(distance_mat)
    for i in range(num):
        distance_mat[i, :] = np.linalg.norm(p[i] - p, axis=1)
        reference_mat[i, :] = ref_radius
        reference_mat[i, :] += ref_radius[i]
    # alpha_mat = (1-(distance_mat/reference_mat)**6) / (1-(distance_mat/reference_mat)**12)
    # a simpler version
    alpha_mat = 1 / (1+(distance_mat/reference_mat)**6)
    np.fill_diagonal(alpha_mat, 1)
    eig_val, eig_vec = np.linalg.eigh(alpha_mat)
    lambd = eig_val[-1]
    v = list((eig_vec.T)[-1])
    vn = list(-(eig_vec.T)[-1])
    # filp the direction if necessary
    if sorted(v, reverse=True) < sorted(vn, reverse=True):
        v = vn
    # sort the eigvector
    atomic_num = get_atomic_number(ele)
    order_list = [(atomic_num[i], v[i]) for i in range(num)]
    order_list = sorted(order_list)
    new_v = np.array([item[1] for item in order_list])
    S = np.sqrt(num) * lambd * new_v
    return S


if __name__ == "__main__":
    data = ['C                  0.18413002    1.25352350    0.32249616',
            'H                  0.64932437    0.31486442    0.54024221',
            'H                  0.64933921    1.53427062    1.24426698',
            'H                  0.54080286    1.75792169   -0.55115535',
            'H                 -1.07586998    1.25353903    0.32249616',
            'H                 -10.07586998    1.25353903    0.32249616',
            'H                 -10.17586998    1.25353903    0.32249616']
    elements = []
    coord = []
    for line in data:
        datatxt = line.strip().split()
        elements.append(datatxt[0])
        coord.append([float(item) for item in datatxt[1:]])
    # elem = BC.element_trans(elements)
    coord = np.array(coord)
    bond = generate_adj_mat(elements, coord)
    S = get_SPRINT_coord(elements, coord)
    print(S)
    exit()
    print(bond)
    a = np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]], bool)
    a = a | a.T
    bond_mat = a
    # 一步可及矩阵
    a = a | np.identity(a.shape[0], bool)
    last_mat = np.array(a)
    for i in range(a.shape[0]):
        a = np.dot(a, a)
        if (last_mat == a).all():
            break
        else:
            last_mat[:] = a
    a[0, 3] = False
    a = a | a.T
    a = np.dot(a, a)
    a = np.dot(a, a)
