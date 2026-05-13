import sys
import os
import numpy as np


def read_traj(filename):
    """
    Read traj from file.

    Returns
    -------
    ele: list
        element list
    coord_list: list[numpy.array]
        coordinates
    second_line: list
        second line list
    """
    with open(filename, 'r') as f:
        data_line = f.readlines()
    second_line = []
    coord_list = []
    ele_list = []
    temp_coord = []
    i = 0
    while i < len(data_line):
        data = data_line[i].strip().split()
        if len(data) == 0:
            i += 1
            continue
        elif len(data) == 1:
            ele = []
            atom_num = int(data[0])
            i += 1
            second_line.append(data_line[i].strip())
            i += 1
            for j in range(atom_num):
                data = data_line[i].strip().split()
                temp_coord.append([float(item) for item in data[1:4]])
                ele.append(data[0])
                i += 1
            coord_list.append(np.array(temp_coord).flatten())
            ele_list.append(ele)
            temp_coord = []
    if len(coord_list) != 0:
        return (ele_list[0], coord_list, second_line)
    else:
        return None


def write_traj(filename, ele, p_list, second_line_list=[], mode='w'):
    head_line = str(len(ele)) + '\n'
    write_str = []
    for i, coord in enumerate(p_list):
        coord = coord.reshape(-1, 3)
        if len(second_line_list) != 0:
            second_line = second_line_list[i] + '\n'
        else:
            second_line = str(i) + '\n'
        write_str.extend([head_line, second_line])
        for j in range(len(coord)):
            write_str.append(' {:>2}      '
                             '{:>11.7f}   '
                             '{:>11.7f}   '
                             '{:>11.7f}\n'.format(ele[j],
                                                  coord[j][0],
                                                  coord[j][1],
                                                  coord[j][2]))
        write_str.append('\n')
    data = ''.join(write_str)
    with open(filename, mode=mode) as f:
        f.write(data)


def read_all_last_coord():
    all_filename = os.listdir()
    file_list = []
    for filename in all_filename:
        if os.path.isdir(filename) and filename[:4] == 'run_':
            file_list.append(filename)
    index = []
    for name in file_list:
        tag = name[4:]
        index.append(int(tag))
    index.sort()
    last_p_list = []
    for item in index:
        filename = 'run_' + str(item)
        traj_path = os.path.join(filename, 'traj.xyz')
        ele, p_list, _ = read_traj(traj_path)
        last_p_list.append(p_list[-1])
    write_traj('last_coord.xyz', ele, last_p_list)


if __name__ == "__main__":
    read_all_last_coord()
