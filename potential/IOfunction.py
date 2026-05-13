import numpy as np

def read_xyz(input_xyzfile, return_all_elements=False):
    """
    Read xyz file

    Parameters
    ----------
    input_xyzfile: str
        Input file path
    return_all_elements: bool
        Return all element list (True) or just the first one (False).

    Returns
    -------
    ele: list
        element list
    coord_list: list[numpy.array]
        coordinates
    second_line: list
        second line list
    """
    with open(input_xyzfile, 'r') as f:
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
            for _ in range(atom_num):
                data = data_line[i].strip().split()
                temp_coord.append([float(item) for item in data[1:4]])
                ele.append(data[0])
                i += 1
            coord_list.append(np.array(temp_coord).flatten())
            ele_list.append(ele)
            temp_coord = []
    if len(coord_list) != 0:
        if return_all_elements:
            return (ele_list, coord_list, second_line)
        else:
            return (ele_list[0], coord_list, second_line)
    else:
        return None


def write_xyz(filename, ele_list, p_list, second_line_list=[], mode='w', add_empty_line=False):
    write_str = []
    for i, coord in enumerate(p_list):
        if len(ele_list) == 1:
            ele = ele_list[0]
        else:
            ele = ele_list[i]
        head_line = str(len(ele)) + '\n'
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
        if add_empty_line:
            write_str.append('\n')
    data = ''.join(write_str)
    with open(filename, mode=mode) as f:
        f.write(data)


def read_gjf(input_gjffile):
    with open(input_gjffile, 'r') as f:
        fileline = f.readlines()
    empty_line_counter = 0
    start_line = 0
    end_line = 0
    for i in range(len(fileline)):
        data = fileline[i].strip().split()
        if len(data) == 0:
            empty_line_counter += 1
            if empty_line_counter == 2:
                start_line = i + 2
            elif empty_line_counter == 3:
                end_line = i
    ele = []
    coord = []
    for line in fileline[start_line:end_line]:
        line_data = line.strip().split()
        ele.append(line_data[0])
        coord.append([float(item) for item in line_data[1:4]])
    return (ele, np.array(coord).flatten())


def read_ChemShell_inputfile(input_chemshfile):
    """
    Read coordinates from ChemShell coordinate file (Unit: Bohr)
    """
    with open(input_chemshfile, 'r') as f:
        filelines = f.readlines()
    coordinate_line = 0
    for i in range(len(filelines)):
        if 'block = coordinates' in filelines[i]:
            coordinate_line = i
            break
    coordinate_line_num = int(filelines[coordinate_line].strip().split()[-1])
    file_head = filelines[:coordinate_line+1]
    file_end = filelines[coordinate_line+coordinate_line_num+1:]
    ele_list = []
    coord_list = []
    for i in range(coordinate_line+1, coordinate_line+coordinate_line_num+1):
        data = filelines[i].strip().split()
        ele_list.append(data[0])
        coord_list.extend([float(item) for item in data[1:]])
    return (ele_list, np.array(coord_list).flatten(), file_head, file_end)


def align(coord1, coord2):
    """
    coord1: coordinate used as reference
    coord2: coordinate will be rotated
    Align two molecules
    """
    coord1 = coord1.reshape(-1, 3)
    coord2 = coord2.reshape(-1, 3)
    target_coord1 = coord1
    target_coord2 = coord2
    center1 = np.sum(target_coord1, axis=0) / target_coord1.shape[0]
    center2 = np.sum(target_coord2, axis=0) / target_coord2.shape[0]
    coord1 = coord1 - center1
    coord2 = coord2 - center2
    target_coord1 = coord1
    target_coord2 = coord2
    P = np.asmatrix(target_coord1, dtype=np.float64)
    I = np.identity(3, np.float64)
    for i in range(10):
        Q = np.asmatrix(target_coord2, dtype=np.float64)
        # Kabsch
        h = np.matmul(P.T, Q)
        u, s, vt = np.linalg.svd(h)
        v = vt.T
        d = np.linalg.det(np.matmul(v, u.T))
        e = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, d]], np.float64)
        r = v @ e @ u.T
        coord3 = np.array(np.matmul(target_coord2, r), np.float64)
        if (np.abs(I-r) < 1e-12).all():
            break
        else:
            target_coord2 = coord3
    return (coord1.flatten(), coord3.flatten(), r)
