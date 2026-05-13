
def __get_element_table():
    element_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                     'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                     'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                     'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                     'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                     'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                     'Tl', 'Pb', 'Bi']
    # mass (unit: amu)
    mass_table = [1.007825,  4.00260,  7.01600,  9.01218,
                  11.00931, 12.00000, 14.00307, 15.99491, 18.99840,
                  19.99244, 22.98980, 23.98504, 26.98153, 27.97693,
                  30.99376, 31.97207, 34.96885, 39.96272, 38.96371,
                  39.96259, 44.95592, 47.90000, 50.94400, 51.94050,
                  54.93810, 55.93490, 58.93320, 57.93530, 62.92980,
                  63.92910, 68.92570, 73.92190, 74.92160, 79.91650,
                  78.91830, 83.80000, 84.91170, 87.90560, 88.90540,
                  89.90430, 92.90600, 97.90550, 98.90620, 101.9037,
                  102.9048, 105.9032, 106.9041, 113.9036, 114.9041,
                  117.9018, 120.9038, 129.9067, 126.9004, 131.9042,
                  133.9051, 137.9050, 138.9061, 139.9053, 140.9070,
                  141.9075, 145.0000, 151.9195, 152.9209, 157.9241,
                  159.9250, 163.9265, 164.9303, 165.9304, 168.9344,
                  173.9390, 174.9409, 179.9468, 180.9480, 183.9510,
                  186.9560, 189.9586, 192.9633, 194.9648, 196.9666,
                  201.9706, 204.9745, 207.9766, 208.9804]
    return (element_table, mass_table)


def get_atomic_number(elements_list):
    element_table, mass_table = __get_element_table()
    atomic_number = [element_table.index(item.capitalize())+1 for item in elements_list]
    return atomic_number


def get_mass(elements_list, mass_factor_list=[]):
    """
    mass_factor_list: list
        example: [['H', 4]], the mass of H is set to (4 * H a.m.u)
    """
    # atomic number - 1 (start from 0)
    element_table, mass_table = __get_element_table()
    atomic_index = [element_table.index(item.capitalize()) for item in elements_list]
    mass_factor = {}
    for item in mass_factor_list:
        index = element_table.index(item[0].capitalize())
        factor = item[1]
        mass_factor[index] = factor
    atom_mass_list = [mass_table[item] * mass_factor.get(item, 1) for item in atomic_index]
    return atom_mass_list


def atomic_number_to_elements(atomic_number):
    element_table, mass_table = __get_element_table()
    ele_list = [element_table[item-1] for item in atomic_number]
    return ele_list


if __name__ == "__main__":
    ele_list = ['H', 'He', 'C']
    __get_element_table()

    atomic_number = get_atomic_number(ele_list)

    mass_list = get_mass(ele_list)            # orginal mass
    print('orginal mass:')
    print(mass_list, atomic_number)

    mod_mass_list = get_mass(ele_list, mass_factor_list=[['H', 4]])
    print('modified mass:')
    print(mod_mass_list, atomic_number)
