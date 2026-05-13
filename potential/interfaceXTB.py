import os
import logging
import re
import shutil
import time
import numpy as np


class InterfaceXTB():
    def __init__(self):
        self.name = 'xTB'
        self.programe_name = 'xtb'
        self.error_input_filename = 'error.inp'     # file for exception
        self.error_output_filename = 'error.out'
        self.normal_input_filename = 'init.inp'
        self.normal_output_filename = 'init.out'
        self.current_step = 0
        self.bohr2angs = 0.5291772083
        self.ev2au = 0.0367502
        self.au2kcalmol = 627.5094740631
        self.kcalmol2au = 1/self.au2kcalmol
        self.kcalmolangs2au = self.bohr2angs*self.kcalmol2au
        self.path = None
        self.job_str = ''
        self.filename_energy = 'energy'
        self.filename_grad = 'gradient'
        self.filename_input = 'xtb_input.xyz'
        self.filename_output = 'xtb_output.log'
        self.cwd = ''
        self.working_filename = 'QM'
        self.multiplicity = 1

    def set_quant_num(self, quant_num):
        self.quant_num = quant_num

    def set_elements(self, ele):
        """Atomic number"""
        self.elements = ele

    def set_job_str(self, job_str):
        self.job_str = job_str

    def set_cwd(self, cwd):
        self.cwd = cwd

    def get_geometry_str(self, coord):
        geom_block = []
        temp_coord = coord.reshape(-1, 3)
        temp_block = []
        for i in range(len(temp_coord)):
            temp_block.append('{:>2} {:>12.8f} {:>12.8f}  {:>12.8f}\n'.format(self.elements[i],
                                                                              temp_coord[i][0],
                                                                              temp_coord[i][1],
                                                                              temp_coord[i][2]))
        geom_block = ''.join(temp_block)
        return geom_block

    def search_energy(self, filelines):
        """Search energy(Hartree) from energy file"""
        energy = float(filelines[1].strip().split()[1])
        return energy

    def search_force(self, filelines):
        """Search force(-gradient Hartree/Bohr) and energy(Hartree) from gradient file"""
        energy_line = filelines[1]
        energy = float(energy_line.strip().split()[6])
        gradient_data = []
        for line in filelines[2:]:
            data = line.strip().split()
            if len(data) == 3:
                gradient_data.extend([float(item) for item in data])
        force = -np.array(gradient_data)
        return (energy, force)

    def check_normal_termination(self, filelines):
        if 'normal termination' in filelines[-1]:
            return True
        else:
            return False

    def check_opt_converge(self, text):
        fail_text = re.search(self.pattern_opt_converged, text)
        fail_text_2 = re.search(self.pattern_opt_unconverged, text)
        if fail_text is None or (fail_text_2 is not None):
            # not converged
            return False
        else:
            return True

    def init_run(self, working_path, wokring_filename):
        path = os.path.join(working_path, wokring_filename)
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path

    def terminate_run(self):
        shutil.rmtree(self.path)

    def write_data(self, input_str):
        input_path = os.path.join(self.path, self.filename_input)
        with open(input_path, 'w') as f:
            f.write(input_str)

    def remove_fort(self):
        pass

    def set_working_filename(self, filename):
        self.working_filename = filename

    def set_multiplicity(self, multiplicity=1):
        """
        Spin multiplicity, default=-1 (singlet)
        """
        self.multiplicity = multiplicity

    def get_state_index(self, state):
        return 0

    def run(self, coord, MM_info=None, save_err=False, save_nor=False):
        """
        Coordinates are given in Angs unit
        """
        cwd = os.getcwd()
        try:
            # generate input
            geom_block = self.get_geometry_str(coord)
            input_block = []
            self.current_step += 1

            input_block.append(str(len(self.elements)) + '\n')
            input_block.append('\n')
            input_block.append(geom_block)
            input_str = ''.join(input_block)

            os.chdir(self.path)
            # write input data
            self.write_data(input_str)
            if os.path.exists(self.filename_output):
                os.remove(self.filename_output)
            # run xtb
            os.system(f'{self.programe_name} {self.filename_input} {self.job_str} > {self.filename_output} 2>&1')
            filelines = []
            with open(os.path.join(self.path, self.filename_output), 'r') as f:
                filelines = f.readlines()
            # check termination
            if self.check_normal_termination(filelines=filelines):
                # search energy and force
                with open(os.path.join(self.path, self.filename_grad), 'r') as f:
                    grad_lines = f.readlines()
                e, f = self.search_force(filelines=grad_lines)
                if save_nor:
                    shutil.copy(os.path.join(self.path, self.filename_input), os.path.join(self.cwd, self.filename_input))
                    shutil.copy(os.path.join(self.path, self.filename_output), os.path.join(self.cwd, self.filename_output))
                    shutil.copy(os.path.join(self.path, self.filename_grad), os.path.join(self.cwd, self.filename_grad))
                result = ([e], [f], [])
            elif save_err:
                shutil.copy(os.path.join(self.path, self.filename_input), os.path.join(self.cwd, self.filename_input))
                shutil.copy(os.path.join(self.path, self.filename_output), os.path.join(self.cwd, self.filename_output))
                logging.info('Save error file:')
                logging.info(f'{os.path.join(self.path, self.filename_input)} {os.path.join(self.cwd, self.filename_input)}')
                logging.info(f'{os.path.join(self.path, self.filename_output)} {os.path.join(self.cwd, self.filename_output)}')
                result = None
            return result
        finally:
            os.chdir(cwd)

    def search_optimized_strucutre(self, text):
        pattern_block = re.compile(r'\s+FINAL CARTESIAN GRADIENT NORM.+?INTERATOMIC DISTANCES', re.S)
        result_block = re.search(pattern_block, text)
        result_text = result_block.group(0).strip().split('\n')
        result_text = result_text[8:-4]
        coord = []
        for line in result_text:
            data = line.strip().split()
            if len(data) == 8:
                coord.append([float(data[2]), float(data[4]), float(data[6])])
            else:
                return np.array(coord).flatten()
        return np.array(coord).flatten()

    def force(self, coord, save_err=True, save_nor=False):
        e_list, f_list = self.run(coord, save_err=save_err, save_nor=save_nor)
        return (e_list, f_list)


if __name__ == "__main__":
    # path = r'D:\git local\mtd\opt.output'
    # with open(path, 'r') as f:
        # text = f.read()
    test = InterfaceXTB()
    cwd = os.getcwd()
    test.set_cwd(cwd)
    tmp_path = r'/dev/shm'
    test.init_run(tmp_path, 'xtb_test')
    test.set_job_str('--grad --gfn 2')
    test.set_elements(['C', 'H', 'H', 'H', 'H'])
    start_time = time.perf_counter()
    coord1 = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0]])
    coord2 = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0]])
    result = test.run(coord1, save_nor=True)
    e_list, f_list, _ = result
    print(e_list)
    print(f_list)
    test.terminate_run()


