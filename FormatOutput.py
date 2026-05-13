import numpy as np

class FormatOutput():
    def __init__(self):
        pass

    def format_traj_data(self, ele, coord, current_time, current_step):
        """
        Record coordinates and time.
        """
        atom_num = len(ele)
        traj_block = [' {:d}\n'.format(atom_num),
                      ' step: {:>6d}'' time: {:>5.1f} \n'.format(current_step, current_time)]
        p = coord.reshape(-1, 3)
        for i in range(len(p)):
            traj_block.append(' {:>3} {:>13.6f} {:>13.6f} {:>13.6f}\n'.format(ele[i],
                                                                              p[i][0],
                                                                              p[i][1],
                                                                              p[i][2]))
        return traj_block

    def format_dege_data(self, ele, coord, current_time, current_step, energy_list):
        """
        Record coordinates, time, energies.
        """
        atom_num = len(ele)
        energy_block = ' '.join(['{:>6.4f}'.format(item) for item in energy_list])
        traj_block = [' {:d}\n'.format(atom_num),
                      ' step: {:>6d}'' time: {:>5.1f} '.format(current_step, current_time) + energy_block+ '\n']
        p = coord.reshape(-1, 3)
        for i in range(len(p)):
            traj_block.append(' {:>3} {:>13.6f} {:>13.6f} {:>13.6f}\n'.format(ele[i],
                                                                              p[i][0],
                                                                              p[i][1],
                                                                              p[i][2]))
        return traj_block


    def format_energy_data(self, e_list, current_time, current_step):
        energy_bloock = [' step: {:>6d} time: {:>6.1f}'.format(current_step, current_time),
                         ''.join([' E{:d}: {:>14.8f}'.format(i+1, item) for i, item in enumerate(e_list)]),
                         ' dEmax:{:>13.6f}\n'.format(max(e_list)-min(e_list))]
        return energy_bloock

    def format_force_data(self, ele, force_list, current_time, current_step):
        """
        Record force and time.
        """
        atom_num = len(ele)
        force_block = [' {:d}\n'.format(atom_num),
                      ' step: {:>6d}'' time: {:>5.4f} \n'.format(current_step, current_time)]
        new_f_list = []
        for force in force_list:
            f = force.reshape(-1, 3)
            new_f_list.append(f)
        f_block = np.concatenate(new_f_list, axis=1)
        # force number
        num = len(force_list)
        format_str = ' '.join(['{:>13.6f}']*num*3)
        for i in range(len(f_block)):
            force_block.append(' {:>3}'.format(ele[i]) + format_str.format(*f_block[i]) + '\n')
        return force_block


if __name__ == "__main__":
    ele = ['C', 'H']
    f1 = np.array([[1.0, 2.0, 0.2], [0.1, 0.2, 0.3]])
    f2 = f1 + 0.1
    f_list = [f1, f2]
    my_format = FormatOutput()
    result = my_format.format_force_data(ele, f_list, 1, 1)
    for item in result:
        print(item, end='')
