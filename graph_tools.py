import numpy as np
import time
import itertools
import copy
import logging
from collections import deque
from multiprocessing.pool import Pool


def graph_label_update(atom_num, adj_mat, init_info):
    labels = np.zeros(atom_num, np.int32)
    last_labels = np.zeros_like(labels)
    max_connection = np.max(np.sum(adj_mat, axis=1))
    # info matrix: init_info, current_index, key1, key2..., addition_key
    info_mat = np.zeros((atom_num, max_connection+3), dtype=np.int32)
    info_mat[:, 0] = init_info
    counter = np.zeros_like(labels, np.int32)
    while True:
        # sort key
        sort_index = np.lexsort(info_mat.T[::-1])
        # update labels
        sort_key = info_mat[sort_index]
        unique_element, unique_labels, unique_count = np.unique(sort_key, axis=0, return_inverse=True, return_counts=True)
        labels[sort_index] = unique_labels

        # now check is new labels is the same
        specialize_index = None
        if np.all(labels == last_labels):
            if len(unique_element) == atom_num:
                # all atoms are unique, exit
                return labels
            else:
                # specialize the max degenerate atom
                max_degenerate_index = np.where(unique_count > 1)[0][-1]
                max_degenerate_element = unique_element[max_degenerate_index]
                specialize_index = np.where(np.all(info_mat == max_degenerate_element, axis=1))[0][0]
        else:
            last_labels[:] = labels
        
        info_mat[:, 1] = labels
        info_mat[:, 2:] = 0
        if specialize_index is not None:
            info_mat[specialize_index, -1] = 1
        # join the neighbor index
        # the key is inserted from 2 (init_info, current_index, key1, key2...)
        counter[:] = 2
        # loop from the max label
        iter_index = np.argsort(labels)[::-1]
        for index in iter_index:
            # find the neighbor
            neighbor_index = np.where(adj_mat[index] != 0)[0]
            info_mat[neighbor_index, counter[neighbor_index]] = labels[index]
            counter[neighbor_index] += 1


def gen_graph_label(adj_mat, atomic_number_list):
    atom_num = len(atomic_number_list)
    # initilize hash, use atomic number and Coordinate Number
    init_info = atomic_number_list * 100 + np.sum(adj_mat, axis=1)
    new_info = graph_label_update(atom_num, adj_mat, init_info)
    # new_info = graph_label_update_old(atom_num, adj_mat, init_info)
    return new_info


def DFS_expand(adj_mat, label, ele):
    atom_num = len(adj_mat)
    label_set = sorted(set(label))
    label_priority = [label_set.index(item)+1 for item in label]
    visited_flag = [False for _ in range(len(label))]
    name = ['' for _ in range(len(label))]
    pri_mat = adj_mat * label_priority
    start_node = label_priority.index(max(label_priority))
    stack = [start_node]
    step = 1

    path = []
    name[start_node] = ele[start_node] + str(label[start_node]) + '_' + str(step)
    path.append(name[start_node])
    visited_flag[start_node] = True
    break_str = '|'
    branch_list = []
    min_path = []

    while len(stack) != 0:
        current_node = stack[-1]
        # next node
        if np.sum(pri_mat[current_node]) != 0:
            next_node = np.argmax(pri_mat[current_node])
            max_node_index = np.where(pri_mat[current_node] == pri_mat[current_node, next_node])[0]
            max_node_num = len(max_node_index)
            if max_node_num > 1:
                for i in range(1, max_node_num):
                    tmp_mat = pri_mat.copy()
                    tmp_mat[current_node, max_node_index[i]] = pri_mat[current_node, next_node] +1
                    tmp_mat[max_node_index[i], current_node] = pri_mat[current_node, next_node] +1
                    branch_list.append((copy.deepcopy(stack), tmp_mat, copy.deepcopy(visited_flag), copy.deepcopy(name), copy.deepcopy(path), step))

            pri_mat[current_node, next_node] = 0
            pri_mat[next_node, current_node] = 0
            if visited_flag[next_node]:
                path.append(name[next_node])
                path.append(break_str)
            else:
                stack.append(next_node)
                visited_flag[next_node] = True
                step += 1
                name[next_node] = ele[next_node] + str(label[next_node]) + '_' + str(step)
                if path[-1] == break_str:
                    path.append(name[current_node])
                    path.append(name[next_node])
                else:
                    path.append(name[next_node])
        else:
            # all connected bonds are visited
            stack.pop()
            if path[-1] != break_str:
                path.append(break_str)
        # compare path
        if len(min_path) != 0:
            if len(path) <= len(min_path):
                if path > min_path[:len(path)] and len(branch_list) != 0:
                    stack, pri_mat, visited_flag, name, path, step = branch_list.pop()
                    continue
            else:
                if len(branch_list) != 0:
                    stack, pri_mat, visited_flag, name, path, step = branch_list.pop()
                    continue
        # process fragments
        if len(stack) == 0:
            if sum(visited_flag) != atom_num:
                start_node = np.argmax([label_priority[i] if not visited_flag[i] else 0 for i in range(atom_num)])
                stack.append(start_node)
                step += 1
                name[start_node] = ele[start_node] + str(label[start_node]) + '_' + str(step)
                path.append(name[start_node])
                visited_flag[start_node] = True
            else:
                # finished, but need to deal with other branch
                if len(min_path) == 0:
                    min_path = path
                else:
                    min_path = min(min_path, path)
                if len(branch_list) != 0:
                    stack, pri_mat, visited_flag, name, path, step = branch_list.pop()
        # print(step, path)
    return ''.join(min_path)


def atomic_number_table(ele):
    element_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                     'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                     'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                     'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                     'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                     'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                     'Tl', 'Pb', 'Bi']
    return np.array([element_table.index(item)+1 for item in ele], dtype=np.int32)


def gen_graph_hash(ele, adj_mat):
    """
    adj_mat -> 1-D code
    """
    atomic_number_list = atomic_number_table(ele)
    graph_label = gen_graph_label(adj_mat, atomic_number_list)
    result = DFS_expand(adj_mat, graph_label, ele)
    return result


def get_coordinate_number(ele):
    coord_num = {'H': [1, 1], 'He': [0, 0], 'Li':[0, 1], 'Be':[0, 2], 'B':[1, 3], 'C':[2, 4], 'N':[1, 3], 'O':[1, 2], 'F':[0, 1], 'Ne':[0, 0],
                 'Na':[0, 1], 'Mg':[0, 2], 'Al':[0, 3], 'Si':[1, 4], 'P':[1, 5], 'S':[0, 6], 'Cl':[0, 1]}
    CN_min = [coord_num[item][0] for item in ele]
    CN_max = [coord_num[item][1] for item in ele]
    return (CN_min, CN_max)


def graph_generator_with_input(ele, adj_mat, hash_list, input_reaction_list):
    product_hash = []
    product = []
    for product_bond_list in input_reaction_list:
        new_adj = adj_mat.copy()
        for bond in product_bond_list:
            i, j, bond_type = bond
            i = i-1
            j = j-1
            new_adj[i, j] = new_adj[j, i] = bond_type
        hash_str = gen_graph_hash(ele, new_adj)
        if hash_str not in hash_list and hash_str not in product_hash:
            product_hash.append(hash_str)
            product.append(new_adj)
    return (product, product_hash)



def graph_generator_subprocess(ele, adj_mat, hash_list, comb_list, max_frag_num=2, max_radical_num=2):
    CN_min, CN_max = get_coordinate_number(ele)
    CN_flag = np.zeros(len(ele), dtype=bool)
    if max_radical_num >= 0:
        act_ele_list = get_active_ele_num(ele)
    CO_array = (('C', 'O'), ('O', 'C'))
    H3O_array = (('O', 'H', 'H', 'H'), ('H', 'O', 'H', 'H'), ('H', 'H', 'O', 'H'), ('H', 'H', 'H', 'O'))
    product = []
    product_hash = []

    for combination in comb_list:
        break_set, form_set = combination
        # start_time = time.time()
        if len(break_set) == 0 and len(form_set) == 0:
            continue
        new_adj = adj_mat.copy()
        # break bonds
        for i, j in break_set:
            new_adj[i, j] = new_adj[j, i] = 0
        # form bonds
        for i, j in form_set:
            new_adj[i, j] = new_adj[j, i] = 1
        # we may not care about disscoiation reaction
        frag_num, frag_index_list = get_fragment_number(new_adj)
        if max_frag_num >= 0:
            if frag_num > max_frag_num:
                continue
        # check constrains
        # Use max/min Coordinate Number
        CN_flag[:] = True
        CN = new_adj.sum(axis=0)
        CN_flag = (CN_flag & (CN >= CN_min) & (CN <= CN_max))
        # some expections
        for frag in frag_index_list:
            if len(frag) == 2:
                # CO
                if np.all(ele[frag] == CO_array[0]) or np.all(ele[frag] == CO_array[1]):
                    CN_flag[frag[0]] = True
                    CN_flag[frag[1]] = True
            # elif len(frag) == 4:
            #     # H3O
            #     # if np.all(np.sort(ele[frag]) == H3O_array):
            #     if (np.all(ele[frag] == H3O_array[0]) or np.all(ele[frag] == H3O_array[1]) or
            #         np.all(ele[frag] == H3O_array[2]) or np.all(ele[frag] == H3O_array[3])):
            #         O_index = np.where(ele[frag] == 'O')[0]
            #         CN_flag[frag[O_index]] = True
        if not np.all(CN_flag):
            continue
        # check radical number
        if max_radical_num >= 0:
            if find_ele_pair(new_adj, act_ele_list) > max_radical_num:
                continue
        hash_str = gen_graph_hash(ele, new_adj)
        if hash_str not in hash_list and hash_str not in product_hash:
            product_hash.append(hash_str)
            product.append(new_adj)
    return (product, product_hash)


def graph_generator(ele, adj_mat, hash_list, b=2, f=2, max_frag_num=2, max_radical_num=2, nproc=1):
    product_hash = []
    ele = np.array(ele)
    atom_num = len(adj_mat)
    bonds = [(i, j) for i in range(atom_num) for j in range(i+1, atom_num) if adj_mat[i, j] == 1]
    nonbonds = [(i, j) for i in range(atom_num) for j in range(i+1, atom_num) if adj_mat[i, j] == 0]
    CN_min, CN_max = get_coordinate_number(ele)
    product = []
    # start_time = time.time()
    # time_A = 0
    # counter = 0
    if max_radical_num >= 0:
        act_ele_list = get_active_ele_num(ele)
    comb_list = []
    for break_num in range(b+1):
        break_combos = list(itertools.combinations(bonds, break_num))
        for form_num in range(f+1):
            form_combos = list(itertools.combinations(nonbonds, form_num))
            for break_set, form_set in itertools.product(break_combos, form_combos):
                comb_list.append((break_set, form_set))
    comb_block_list = []
    comb_num = len(comb_list)
    block_size = comb_num // nproc
    remainder = comb_num % nproc
    start = 0
    for i in range(nproc):
        end = start + block_size + (1 if i < remainder else 0)
        comb_block_list.append(comb_list[start:end])
        start = end

    product = []
    product_hash = []
    pool = Pool(processes=nproc)
    result_list = []
    for sub_comb_list in comb_block_list:
        result = pool.apply_async(func=graph_generator_subprocess, args=(ele, adj_mat, hash_list, sub_comb_list, max_frag_num, max_radical_num))
        result_list.append(result)
    for result in result_list:
        tmp_product_adj_mat_list, tmp_product_hash_list = result.get()
        for tmp_adj, tmp_hash in zip(tmp_product_adj_mat_list, tmp_product_hash_list):
            if tmp_hash not in hash_list and tmp_hash not in product_hash:
                product_hash.append(tmp_hash)
                product.append(tmp_adj)
    pool.close()
    pool.join()
    return (product, product_hash)


def get_reaction_direction(adj_react, adj_product):
    """
    Transfer reaction matrix to bond index and bond type.

    Returns
    ----------
    bond_index: list
        bond index pair
    bond_type: list
        1 for formation and -1 for breaking
    restrain_index: list
        restrained bond index pair
    """
    logging.info('Generate reaction direction...')
    diff = adj_product - adj_react
    diff = np.triu(diff, k=1)
    reaction_index = np.where(diff != 0)
    bond_index = [[i, j] for i, j in zip(reaction_index[0], reaction_index[1])]
    bond_type = diff[reaction_index]
    restrain_mat = np.where(np.triu((adj_react==1) & (adj_product==1), k=1))
    restarin_index = [[i, j] for i, j in zip(restrain_mat[0], restrain_mat[1])]
    logging.info('Reaction direction done.')
    return (bond_index, bond_type, restarin_index)


def get_fragment_number(adj_mat):
    atom_num = len(adj_mat)
    # bool matrix
    a = np.where(adj_mat != 0, True, False)
    a = a | a.T | np.identity(atom_num, dtype=bool)
    # calculate reachability matrix
    last_a = a.copy()
    for _ in range(atom_num):
        a = a.dot(a)
        if (last_a == a).all():
            break
        else:
            last_a[:] = a
    visited_flag = np.array([False for _ in range(len(a))])
    mol_frag_list = []
    ### identify fragments via reachability matrix
    for i in range(atom_num):
        if not visited_flag[i]:
            frag_index = np.where(a[i])[0]
            visited_flag[frag_index] = True
            mol_frag_list.append(frag_index)
    return (len(mol_frag_list), mol_frag_list)


def get_active_ele_num(ele):
    ele_num_table = {'H':1, 'He':0, 'Li':1, 'Be':2, 'B':3, 'C':4, 'N':3, 'O':2, 'F':1, 'Ne':0,
                        'Na':1, 'Mg':2, 'Al':3, 'Si':4, 'P':3, 'S':2, 'Cl':1}
    electron_num = []
    for item in ele:
        electron_num.append(ele_num_table.get(item, 0))
    return electron_num


def pair_electron(adj_mat, rest_ele):
    pi_bond_mat = np.zeros_like(adj_mat, np.int32)
    # initial pair
    for i in range(len(rest_ele)):
        if rest_ele[i] != 0:
            # connected atom
            bonded_index = np.where(adj_mat[i] != 0)[0]
            for j in bonded_index:
                if rest_ele[j] > 0:
                    # find a pair
                    min_pi_bond_num = min(rest_ele[i], rest_ele[j])
                    rest_ele[i] = rest_ele[i] - min_pi_bond_num
                    rest_ele[j] = rest_ele[j] - min_pi_bond_num
                    pi_bond_mat[i, j] += min_pi_bond_num
                    pi_bond_mat[j, i] += min_pi_bond_num
                    if rest_ele[i] == 0:
                        break

    radical_index_need_to_process = []
    for i in np.where(rest_ele != 0)[0]:
        # only form ONE pi bond at each loop, so treat multi radical into multi one-radical
        radical_index_need_to_process.extend([i for _ in range(rest_ele[i])])
    no_pair_radical = []
    max_loop_num = sum(rest_ele)
    adj_list = [np.where(adj_mat[i] != 0)[0].tolist() for i in range(len(adj_mat))]
    for i in range(max_loop_num):
        if len(radical_index_need_to_process) == 0:
            break
        pair_flag = False
        radical_index = radical_index_need_to_process.pop(0)
        radical_possibal_position = deque([radical_index])
        # BFS search radical possibal position and pi-bond flip path
        pre_index = {}
        visited_pos = set([radical_index])
        pi_bond_list = [np.where(pi_bond_mat[i] != 0)[0].tolist() for i in range(len(pi_bond_mat))]
        while radical_possibal_position:
            # one step forward from farest position
            index = radical_possibal_position.popleft()
            # sigma bonded atom index
            sigma_bond_atom_list = adj_list[index]
            # pi bonded atom index
            for near_atom in sigma_bond_atom_list:
                if near_atom in radical_index_need_to_process:
                    # found a new pi bond
                    # form a new pi bond
                    pi_bond_mat[index, near_atom] += 1
                    pi_bond_mat[near_atom, index] += 1
                    # adjust pi-bond path
                    C = index
                    while C in pre_index.keys():
                        # traceback
                        # A*-B=C -> A=B-C*
                        A, B = pre_index[C]
                        pi_bond_mat[B, C] -= 1
                        pi_bond_mat[C, B] -= 1
                        pi_bond_mat[A, B] += 1
                        pi_bond_mat[B, A] += 1
                        C = A
                    rest_ele[radical_index] -= 1
                    rest_ele[near_atom] -= 1
                    radical_index_need_to_process.remove(near_atom)
                    pair_flag = True
                    break
                if pair_flag:
                    break

                # C*-C=C <- this is the third atom
                # third_atom_list = np.where(pi_bond_mat[near_atom] != 0)[0]
                third_atom_list = pi_bond_list[near_atom]
                for third_atom_index in third_atom_list:
                    # avoid repeat
                    if third_atom_index not in visited_pos:
                        visited_pos.add(third_atom_index)
                        pre_index[third_atom_index] = [index, near_atom]
                        radical_possibal_position.append(third_atom_index)
            if pair_flag:
                break
        # not paired, record it
        if not pair_flag:
            no_pair_radical.append(radical_index)
    return len(no_pair_radical)


def find_ele_pair(adj_mat, act_ele_num_list):
    """
    Search possible electron pair and radical.
    """
    sum_bond = np.sum(adj_mat, axis=1)
    rest_ele = np.array(act_ele_num_list, np.int32) - sum_bond
    radical_num = pair_electron(adj_mat, rest_ele)
    return radical_num
    # unusual_atom = np.where(rest_ele < 0)[0]
    # if len(unusual_atom) == 0:
    #     # try to pair
    #     radical_num = pair_electron(adj_mat, rest_ele)
    #     return radical_num
    # else:
    #     # try to cut
    #     for i in range(len(unusual_atom)):
    #         pass


def get_bond_from_mat(diff_mat):
    """
    Get bond index and bond type from connectivity matrix.
    """
    react_mat = diff_mat
    diff = np.triu(react_mat, k=1)
    reaction_bond_index = np.where(diff != 0)
    bond_index = sorted([[i, j] for i, j in zip(reaction_bond_index[0], reaction_bond_index[1])])
    bond_type = [react_mat[item[0], item[1]] for item in bond_index]
    return (bond_index, bond_type)

