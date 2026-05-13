import os
import logging
import re
import mmap
import time
import subprocess
import tempfile
import numpy as np
from potential.atomic_mass import get_atomic_number


class FIFO():
    def __init__(self, input_filename, output_filename):
        self.input_filename = input_filename
        self.output_filename = output_filename
        os.mkfifo(self.input_filename)
        os.mkfifo(self.output_filename)

    def write_data(self, data):
        """Write data into input_filename"""
        with open(self.input_filename, 'w') as f:
            f.writelines(data)

    def read_data(self):
        """Read data from output_fielname"""
        with open(self.output_filename, 'r') as f:
            text = f.read()
        return text

    def __del__(self):
        os.remove(self.input_filename)
        os.remove(self.output_filename)


class MMAPObj():
    def __init__(self):
        # Bytes
        self.file_size_unit = 512
        self.file_size = self.file_size_unit
        self.encode = 'utf-8'
        self.filename = ''

    def init_mmap(self, filename):
        self.filename = filename
        self.file_obj = open(filename, 'w+b')
        self.file_obj.truncate(self.file_size)
        self.mm = mmap.mmap(self.file_obj.fileno(), length=self.file_size, access=mmap.ACCESS_WRITE)

    def write_data(self, str_data):
        # check data size
        data_encode = str_data.encode(self.encode)
        if len(data_encode) != self.file_size:
            self.file_size = len(data_encode)
            self.file_obj.truncate(self.file_size)
            self.mm.resize(self.file_size)
        # write data
        self.mm.seek(0)
        self.file_obj.seek(0)
        self.mm.write(data_encode)
        self.mm.flush()
        # self.mm.seek(0)

    def save_data_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.mm[:].decode(self.encode))

    def del_mmap(self):
        self.mm.close()
        self.file_obj.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)


class InterfaceMNDO():
    def __init__(self):
        self.name = 'MNDO'
        self.quant_num = 2
        self.head_line = []             # job setting
        self.head_line_opt = []         # for optimization
        self.head_line_MECI = []         # for MECI search
        self.title_line = []            # title line
        self.title_line_opt = []
        self.title_line_MECI = []
        self.end_line = []
        self.end_line_MECI = []
        self.elements = []
        self.MO_index = []
        self.programe_name = 'mndo2020'
        self.__init_pattern()
        self.error_input_filename = 'error.inp'     # file for exception
        self.error_output_filename = 'error.out'
        self.normal_input_filename = 'init.inp'
        self.normal_output_filename = 'init.out'
        self.bohr2angs = 0.5291772083
        self.ev2au = 0.0367502
        self.au2kcalmol = 627.5094740631
        self.kcalmol2au = 1/self.au2kcalmol
        self.kcalmolangs2au = self.bohr2angs*self.kcalmol2au
        self.read_mode = 1
        self.working_filename = 'mndo_job'

    def __init_pattern(self):
        self.gradient_block_pattern = re.compile(r'\s*?COORDINATES \(ANGSTROM\)\s*GRADIENTS \(KCAL/\(MOL\*ANGSTROM\)\)'
                                                 r'.*?\n\n.*?\n\n(.+?)\n\n', re.S)
        self.energy_pattern_SCF_energy = re.compile(r'\s*SCF TOTAL ENERGY\s*(-?\d*\.?\d*)\s*EV')
        self.energy_pattern_CI_energy = re.compile(r'Total energy.*?\s*?(-?\d*\.?\d*)\s*eV')
        self.energy_pattern_REF_energy_block = re.compile(r'SUMMARY OF MULTIPLE CI(?:.*\n){5}((?:.*\n)*?)\s*----')
        self.energy_pattern_REF_energy = re.compile(r'\s*\d+?\s*\d+?\s*(-?\d*\.?\d*)\s*.*\n')
        self.pattern_opt_converged = re.compile(r'OPTIMIZATION FINISHED AFTER')
        self.pattern_opt_unconverged = re.compile(r'UNSUCCESSFUL GEOMETRY OPTIMIZATION')
        self.gradient_nsav_block_pattern = re.compile(r'\s*?CARTESIAN GRADIENT FOR STATE.*?\n(.+?)\n\n', re.S)

    def set_quant_num(self, quant_num):
        self.quant_num = quant_num

    def set_data_read_mode(self, read_mode=1):
        # read_mode 1: direct read from output file
        # read_mode 2: read from sav15.dat file (slow, need set nsav15=3)
        self.read_mode = read_mode

    def set_head_line(self, head_line=[], ICI1=4, ICI2=2):
        self.head_line = ''.join(head_line)
        self.ICI1_org = ICI1
        self.ICI2_org = ICI2
        self.ICI1 = ICI1
        self.ICI2 = ICI2

    def set_head_line_opt(self, head_line=[]):
        self.head_line_opt = ''.join(head_line)

    def set_head_line_MECI(self, head_line=[]):
        self.head_line_MECI = ''.join(head_line)

    def set_end_line(self, end_line=[]):
        self.end_line = end_line

    def set_end_line_MECI(self, end_line=[]):
        self.end_line_MECI = end_line

    def set_act_space(self, ICI1, ICI2):
        self.ICI1 = ICI1
        self.ICI2 = ICI2
        self.head_line = re.sub(r'ICI1=\d+', f'ICI1={ICI1}', self.head_line)
        self.head_line = re.sub(r'ICI2=\d+', f'ICI2={ICI2}', self.head_line)

    def reset_act_space(self):
        self.set_act_space(self.ICI1_org, self.ICI2_org)

    def remove_fort(self):
        if os.path.exists('fort.11'):
            os.remove('fort.11')

    def set_working_filename(self, filename):
        self.working_filename = filename

    def set_elements(self, ele):
        """Atomic number"""
        self.elements = get_atomic_number(ele)

    def set_multiplicity(self, multiplicity=1):
        """
        Spin multiplicity, default=-1 (singlet)
        """
        self.multiplicity = multiplicity

    def get_state_index(self, state):
        if state < self.quant_num-1:
            return state
        else:
            return self.quant_num-1

    def get_geometry_str(self, coord):
        geom_block = []
        empty_str = ' ' * 6
        temp_coord = coord.reshape(-1, 3)
        temp_block = []
        for i in range(len(temp_coord)):
            temp_block.append('{:>2}  {}'
                              '{:>12.8f}   1{}'
                              '{:>12.8f}   1{}'
                              '{:>12.8f}   1\n'.format(self.elements[i],
                                                       empty_str,
                                                       temp_coord[i][0],
                                                       empty_str,
                                                       temp_coord[i][1],
                                                       empty_str,
                                                       temp_coord[i][2]))
        temp_block.append('{:>2d}  {}'
                          '{:>12.8f}   0{}'
                          '{:>12.8f}   0{}'
                          '{:>12.8f}   0\n'.format(0, empty_str, 0, empty_str, 0, empty_str, 0))
        geom_block = ''.join(temp_block)
        return geom_block

    def get_external_charge_str(self, ext_chrg):
        p, charge = ext_chrg
        p = p.reshape(-1, 3)
        tmp_block = []
        for i in range(len(p)):
            tmp_block.append('{:>12.8f}   '
                             '{:>12.8f}   '
                             '{:>12.8f}   '
                             '{:>12.8f}\n'.format(p[i][0],
                                                  p[i][1],
                                                  p[i][2],
                                                  charge[i]))
        chrg_block = ''.join(tmp_block)
        return chrg_block

    def search_energy(self, text):
        """Search energy(Hartree) from output"""
        """
        In MNDO, Ev/(kcal/mol) is set to 23.061 (strange...)
        See MNDO/2020/BLOCK0.f line 58:
            DATA EVCAL / 23.061D0/
        """
        basis_energy_match = re.findall(self.energy_pattern_CI_energy, text)                # basis energy
        basis_energy = [float(item)*self.ev2au for item in basis_energy_match]
        energy_REF_block_match = re.findall(self.energy_pattern_REF_energy_block, text)     # REF energy
        energy_abs_list = []
        for i in range(len(energy_REF_block_match)):
            block = energy_REF_block_match[i]
            energy_REF = re.findall(self.energy_pattern_REF_energy, block)
            energy_REF_list = [float(item)*self.kcalmol2au for item in energy_REF]
            # energy_abs_list.extend([basis_energy[i]+item-energy_REF_list[0] for item in energy_REF_list])
            energy_abs_list.extend(energy_REF_list)
        return energy_abs_list

    def search_force(self, text):
        """Search force(-gradient Hartree/Bohr) and energy(Hartree) from output"""
        converged_flag = True
        energy_list = []
        force_list = []
        # energy in Hartree
        energy_list = self.search_energy(text)
        # force
        gradient_matches = re.findall(self.gradient_block_pattern, text)
        for item in gradient_matches:
            gradient_data = []
            all_lines = item.split('\n')
            try:
                for line in all_lines:
                    # cause bug when gradient is too large
                    # tmp = line.strip().split()
                    # gx, gy, gz = map(float, tmp[5:8])
                    gx, gy, gz = map(float, [line[58:70], line[70:82], line[82:94]])
                    gradient_data.extend([gx, gy, gz])
                force_list.append(-np.array(gradient_data)*self.kcalmolangs2au)
            except Exception as e:
                print(e)
        # search gradient of external point charge
        # gradient_ext_chrg_matches = re.findall(self.gradient_block_pattern, text)
        if len(energy_list) != self.quant_num:
            converged_flag = False
        return (energy_list, force_list, converged_flag)

    def search_force_from_nsav(self):
        """
        Search force(-gradient Hartree/Bohr) and energy(Hartree) from fort.15
        Must use nsav15=3.
        """
        with open('fort.15', 'r') as f:
            filelines = f.readlines()
        energy_list = []
        f_list = []
        nac_list = []
        # 0: not read 1: energy, 2: grad, 3: nac
        current_state = 0
        for line in filelines:
            data = line.strip().split()
            if 'STATES, ENERGIES, CARTESIAN AND INTERNAL GRADIENT NORMS' in line:
                current_state = 1
                continue
            elif 'CARTESIAN GRADIENT FOR STATE' in line:
                current_state = 2
                f_list.append([])
                continue
            elif 'CARTESIAN INTERSTATE COUPLING GRADIENT FOR STATES' in line:
                current_state = 3
                nac_list.append([])
                continue
            elif len(data) == 0:
                current_state = 0

            if current_state == 1:
                # energy
                energy_list.append(float(data[1]) * self.kcalmol2au)
            elif current_state == 2:
                # gradient
                f_list[-1].extend([float(item) for item in data[2:]])
            elif current_state == 3:
                # nac
                nac_list[-1].extend([float(item) for item in data[2:]])
        
        f_list = [-np.array(item, np.float64)*self.kcalmolangs2au for item in f_list]
        nac_list = [np.array(item, np.float64)*self.bohr2angs for item in nac_list]
        return (energy_list, f_list, nac_list)

    def check_opt_converge(self, text):
        fail_text = re.search(self.pattern_opt_converged, text)
        fail_text_2 = re.search(self.pattern_opt_unconverged, text)
        if fail_text is None or (fail_text_2 is not None):
            # not converged
            return False
        else:
            return True

    def init_run(self):
        self.mmap_obj = MMAPObj()
        self.mmap_obj.init_mmap(tempfile.mktemp())

    def terminate_run(self):
        self.mmap_obj.del_mmap()

    def run_with_os(self, text):
        with open('mndo.inp', 'w') as f:
            f.write(text)
        start_time = time.perf_counter()
        os.system('mndo2020 < mndo.inp > mndo.out')
        print(time.perf_counter() - start_time, end=' | ')
        with open('mndo.out', 'r') as f:
            data = f.read()
        return data

    def read_CI(self, file_str):
        # --- read CI eigenvectors
        patter_block_coef = re.compile(r'GUGA-CI eigensystem:.*?(.+?)(\r?\n){4}', re.S)
        result = re.search(patter_block_coef, file_str)
        result_line = result.group(1)
        result_line = result_line.strip().split('\n')
        coeff_list = []
        for i in range(len(result_line)):
            line = result_line[i]
            if 'Root' in line or 'Energy' in line or '---' in line:
                continue
            line_data = line.strip().split()
            if len(line_data) == 0:
                continue
            index = int(line_data[0])-1
            if len(coeff_list)-1 < index:
                coeff_list.append([])
            coeff_list[index].extend([float(item) for item in line_data[1:]])
        coeff_mat = np.array(coeff_list, np.float64)
        return coeff_mat

    def read_CI_conf(self, file_str):
        # --- read configuration
        patter_block_conf = re.compile(r'List of Gelfand states represented by the DRT.*?\n(.*?)Number of upper and', re.S)
        result = re.search(patter_block_conf, file_str)
        result_line = result.group(1)
        result_line = result_line.strip().split('\n')
        conf_list = []
        for i in range(len(result_line)):
            line = result_line[i]
            if 'coeff' in line:
                data_list = line.strip().split()
                conf_list.append([int(item) for item in data_list[1:]])
        return np.array(conf_list, dtype=np.int64)

    def read_MO(self, file_str):
        """
        Read MO coefficients
        """
        # --- read MO
        patter_block_MO = re.compile(r'EIGENVALUES AND EIGENVECTORS\.(.+?)(\r?\n){5}', re.S)
        result = re.search(patter_block_MO, file_str)
        result_line = result.group(1)
        result_line = result_line.strip().split('\n')
        MO_data = []
        MO_num = 0
        for i in range(len(result_line)):
            line = result_line[i]
            data_list = line.strip().split()
            if len(data_list) == 0:
                continue
            elif '(' in line:
                for j in range(len(data_list)):
                    if ')' in data_list[j]:
                        MO_num = max(MO_num, int(data_list[0]))
                        tmp_list = [float(item) for item in data_list[j+2:]]
                        MO_data.append(tmp_list)
        # --- split MO data
        MO_mat = [[] for _ in range(MO_num)]
        for i in range(len(MO_data)):
            line_num = i % MO_num
            MO_mat[line_num].extend(MO_data[i])
        # Done --- read MO
        MO_mat = np.array(MO_mat, np.float64)
        return MO_mat

    def read_MO_mapping(self, file_str):
        patter_block_mapping = re.compile(r'SAVED\s+?STATUS\s+?CURRENT\s+?OVERLAP\n(.+?)(\r?\n){4}', re.S)
        result = re.search(patter_block_mapping, file_str)
        result_line = result.group(1)
        result_line = result_line.strip().split('\n')
        mapping_data = []
        for i in range(len(result_line)):
            line = result_line[i]
            if 'CHANGED' in line:
                data_list = line.strip().split()
                mapping_data.append(int(data_list[2]))
        mapping_data = np.array(mapping_data, np.int32)
        return mapping_data

    def get_force_and_NAC(self, force_NAC):
        """
        Convert MNDO output into forces and NAC vectors
        """
        quantum_num = self.quant_num
        # forces
        force_list = [force_NAC[i] for i in range(quantum_num)]
        if len(force_list) == quantum_num:
            return (force_list, [])
        # NAC vectors, unit should be 1/angs, and it need to be converted to 1/Bohr
        NAC_list = []
        triu_index = np.triu_indices(quantum_num, k=1)
        for index in range(len(triu_index[0])):
            NAC_list.append(-force_NAC[quantum_num+index] * self.au2kcalmol)
        return (force_list, NAC_list)

    def run(self, coord, ext_chrg=None, save_err=False, save_nor=False):
        """
        Coordinates are given in Angs unit
        """
        save_nor = True
        cwd = os.getcwd()
        work_path = os.path.join(cwd, self.working_filename)
        if not os.path.exists(work_path):
            os.mkdir(work_path)
        os.chdir(work_path)
        # generate input
        geom_block = self.get_geometry_str(coord)
        input_block = []

        input_block.append(self.head_line)
        input_block.append('\n\n')
        input_block.append(geom_block)
        if len(self.end_line) != 0:
            input_block.append(self.end_line)
        input_block.append('\n')
        # external point charge
        if ext_chrg is not None:
            charge_block = self.get_external_charge_str(ext_chrg)
            input_block.append(charge_block)

        input_block.append('99')

        input_str = ''.join(input_block)
        # write input data into MMAP
        self.mmap_obj.write_data(input_str)
        # run MNDO and get result
        process = subprocess.Popen([self.programe_name],
                                   stdin=self.mmap_obj.file_obj,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate()
        if save_nor:
            self.mmap_obj.save_data_to_file(self.normal_input_filename)
            with open(self.normal_output_filename, 'w') as f:
                f.write(stdout)
        # e_list, f_list, converged_flag = self.search_force(stdout)
        result = self.search_force(stdout)
        return_result = None
        if result is None:
            if save_err:
                self.mmap_obj.save_data_to_file(self.error_input_filename)
                with open(self.error_output_filename, 'w') as f:
                    f.write(stdout)
                logging.info('Last input {} and output {}'.format(self.error_input_filename, self.error_output_filename))
        else:
            e_list, f_list, converged_flag = result
            if ext_chrg is not None:
                new_f_list = []
                for i in range(self.quant_num):
                    tmp_f = f_list[i*2]
                    ext_chrg_f = f_list[i*2+1]
                    tmp_f = np.concatenate((tmp_f, ext_chrg_f))
                    new_f_list.append(tmp_f)
                new_f_list.extend(f_list[self.quant_num*2:])
                f_list = new_f_list
            if converged_flag:
                if self.read_mode == 1:
                    force_list, nac_list = self.get_force_and_NAC(f_list)
                    energy_list = e_list
                elif self.read_mode == 2:
                    energy_list, force_list, nac_list = self.search_force_from_nsav()
                return_result = (energy_list, force_list, nac_list)
                # return (e_list, f_list)
            else:
                # not Converged
                self.mmap_obj.save_data_to_file(self.error_input_filename)
                with open(self.error_output_filename, 'w') as f:
                    f.write(stdout)
                logging.info('Last input {} and output {}'.format(self.error_input_filename, self.error_output_filename))
        os.chdir(cwd)
        return return_result

    def run_opt(self, coord_list, save_err=False, save_nor=False):
        # generate input
        geom_block = self.get_geometry_str(coord_list)
        input_block = []
        for i in range(len(self.mult)):
            input_block.append(self.head_line)
            input_block.append(self.title_line[i])
            input_block.append('\n\n')
            input_block.append(geom_block)
            if len(self.end_line) != 0:
                input_block.append(self.end_line)
            else:
                if len(self.MO_index) != 0:
                    input_block.append(''.join(['   '+str(index) for index in self.MO_index]))
                input_block.append('\n')
                input_block.append(''.join(['   '+str(index) for index in self.CIindex[i]]))
            input_block.append('\n')
        input_block.append('\n99')
        input_str = ''.join(input_block)
        # write input data into MMAP
        self.mmap_obj.write_data(input_str)
        # run MNDO and get result
        process = subprocess.Popen([self.programe_name],
                                   stdin=self.mmap_obj.file_obj,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        # run opt
        stdout, stderr = process.communicate()
        # check opt converge
        if self.check_opt_converge(stdout):
            # get strucutre
            coord = self.search_optimizaed_strucutre(stdout)
            e_list = self.search_energy_opt(stdout)
            return (coord, e_list)
        else:
            return None

    def run_MECI(self, coord, save_err=False, save_nor=False, save_filename=None):
        # generate input
        geom_block = self.get_geometry_str(coord)
        input_block = []

        input_block.append(self.head_line_MECI)
        input_block.append('\n\n')
        input_block.append(geom_block)
        if len(self.end_line_MECI) != 0:
            input_block.append(self.end_line_MECI)
        input_block.append('\n99')
        input_str = ''.join(input_block)
        # write input data into MMAP
        self.mmap_obj.write_data(input_str)
        # run MNDO and get result
        process = subprocess.Popen([self.programe_name],
                                   stdin=self.mmap_obj.file_obj,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        # run opt
        stdout, stderr = process.communicate()
        if save_filename is not None and save_nor:
            self.mmap_obj.save_data_to_file(save_filename)
            with open(save_filename, 'w') as f:
                f.write(stdout)
        # check opt converge
        if self.check_opt_converge(stdout):
            # get strucutre
            coord = self.search_optimizaed_strucutre(stdout)
            e_list = self.search_energy_opt(stdout)
            return (coord, e_list)
        else:
            return None

    def search_optimizaed_strucutre(self, text):
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

    def search_energy_opt(self, text):
        """Search energy(a.u.) from output"""
        # basis energy
        # basis_energy_match = re.findall(self.energy_pattern_CI_energy, text)
        basis_energy_match = re.search(self.energy_pattern_SCF_energy, text)
        basis_energy = float(basis_energy_match.group(1)) * self.ev2au
        # REF energy
        energy_pattern_REF = re.compile(r'FINAL.*FORMATION\s+(-?\d*\.?\d*)\s+KCAL/MOL')
        energy_REF = re.findall(energy_pattern_REF, text)
        energy_REF_list = [float(item)*self.kcalmol2au for item in energy_REF]
        energy_abs_list = [item for item in energy_REF_list]
        return energy_abs_list

    def force(self, coord, save_err=True, save_nor=False):
        e_list, f_list = self.run(coord, save_err=save_err, save_nor=save_nor)
        return (e_list, f_list)


def get_state_index(state, quant_num):
    if state < quant_num-1:
        return state
    else:
        return quant_num-1


if __name__ == "__main__":
    path = r'D:\git local\mtd\opt.output'
    with open(path, 'r') as f:
        text = f.read()
    test = InterfaceMNDO()
    # test.search_force(text)
    test.search_optimizaed_strucutre(text)
    exit()
    test.set_molecular_properties(charge_list=[0, 0], mult_list=[0, 1], uhf_list=[0, 0], kci=[5, 5], movo=[0, 0],
                                  ici1=[6, 6], ici2=[4, 4], nciref=[3, 3], mciref=[0, 0], levexc=[2, 2],
                                  iroot=[3, 3], lroot=[1, 1])
    test.set_title_line()
    test.set_head_line()
    # print(test.head_line)
    # print(test.title_line)
    test.set_elements([6, 1, 1, 1, 1])
    # print(test.elements)
    test.init_run()
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
    e_list, f_list = test.run([coord1, coord2])
    print(e_list)
    print(f_list)
    test.terminate_run()


