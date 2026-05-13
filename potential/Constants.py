"""
Constants list
"""

class Constants():
    def __init__(self):
        self.__init_constants()

    def __init_constants(self):
        self.hartree2kcal = 627.5095                   # Hartree to kcal/mol
        self.ev2kcal = 23.0609                         # eV to kcal/mol
        self.bohr2angs = 0.5291772083                  # Bohr to angstrom
        self.hartree2kjmol = 2625.499639479            # Hartree to kJ/mol 
        self.kjmol_angs2hartree_bohr = (1/self.hartree2kjmol) / (1/self.bohr2angs)
        kg2amu = 6.0221366516752e26
        m2bohr = 1/(5.29177249e-11)
        j2hartree = 2.2937104486906e17
        self.kb = 1.38064852e-23 * j2hartree           # Boltzmann constant (Hartree/K)
        self.Na = 6.02214076e23                        # Avogadro's number
        self.k_amu = j2hartree*1e30/m2bohr**2/kg2amu   # Mass transfer constant (Hartree*fs^2/Bohr^2/amu)
        self.threshold_zero_vector = 1e-12
        self.hbar = 1.054571800e-34                     # Reduced Planck constant (J*s)
        self.hbar = self.hbar * j2hartree * 1e15        # Reduced Planck constant (Hartree*fs)
        # Mass
        mass_au2kg = 9.1093826e-31                      # 1 a.u. mass in kg
        mass_amu2kg = 1.66053886e-27                    # 1 amu mass in kg
        self.mass_amu2au = mass_amu2kg/mass_au2kg       # Mass unit (amu to a.u.)
        # Time
        fs2s = 1e-15                                    # 1 fs to s
        au2s = 2.418884326505e-17                       # 1 a.u. to s
        self.time_fs2au = fs2s / au2s                   # Time unit (fs to a.u.)


if __name__ == "__main__":
    test = Constants()
    print(test.k_amu)
    print(test.mass_amu2au)
    print(test.time_fs2au)
    print(test.kb)
    print((1/test.hartree2kcal * test.bohr2angs / test.mass_amu2au) / ((1/test.bohr2angs)/(test.time_fs2au*1000)**2))
    print(test.kjmol_angs2hartree_bohr)
