import numpy as np
from tda import real_corr_se, my_drpa, my_dtda, symm_drpa
from mf import setup_molecule, calculate_mean_field
from fock import simple_fock, fock_dft
from iterative import g0w0
from pyscf import gw
from pyscf import tddft
import numpy as np
from tda import real_corr_se, my_drpa, my_dtda, symm_drpa
from mf import setup_molecule, calculate_mean_field
import unittest

class G0W0TestBase(unittest.TestCase):
    h2ev = 27.2114

    def compare_drpa_methods(self, species, fock, offset):
        molecule = setup_molecule(species)
        fock_mf = calculate_mean_field(molecule, 'hf')
        n_orbitals = fock_mf.mol.nao_nr()
        n_occupied = fock_mf.mol.nelectron // 2
        n_virtual = n_orbitals - n_occupied

        omega_drpa, R_drpa = my_drpa(fock_mf)
        omega_symm_drpa, R_symm_drpa = symm_drpa(fock_mf)

        for o1, o2 in zip(omega_drpa, omega_symm_drpa):
            self.assertAlmostEqual(o1, o2, places=5, msg="DRPA and Symmetric DRPA excitation energies differ")

    def run_g0w0_test(self, species, fock_mf, fock, td, comparison_td, offset):
        molecule = setup_molecule(species)  # Convert species name to a molecule object

        if fock_mf == 'hf':
            fock_mf = calculate_mean_field(molecule, 'hf')
        elif fock_mf == 'dft':
            fock_mf = calculate_mean_field(molecule, 'dft')

        # set of the common variables
        n_orbitals = fock_mf.mol.nao_nr()
        n_occupied = fock_mf.mol.nelectron//2
        n_virtual = n_orbitals - n_occupied
        homo = n_occupied - 1
        orbital_number = homo + offset 
        fock_mo = fock(fock_mf)  # Use the provided Fock matrix function

        # first do the case where you compare to pyscf
        if comparison_td == 'pyscf':
            if td == 'dtda':
                my_tddft = my_dtda
                other_tddft = tddft.dTDA(fock_mf)
            elif td == 'drpa':
                my_tddft = my_drpa
                other_tddft = tddft.dRPA(fock_mf)
            
            other_tddft.nstates = n_occupied*n_virtual
            e, xy = other_tddft.kernel()
            # Make a fake Y vector of zeros
            td_xy = list()
            if td == 'drpa':
                for x, y in other_tddft.xy:
                    td_xy.append((x, y))
            elif td == 'dtda':
                for x, y in other_tddft.xy:
                    td_xy.append((x, 0*x))
            other_tddft.xy = td_xy
            pyscf_gw = gw.GW(fock_mf, freq_int='exact', tdmf=other_tddft)
            pyscf_gw.kernel(orbs=[orbital_number])

            other_result = pyscf_gw.mo_energy*self.h2ev
            my_result = g0w0(orbital_number, fock_mo, real_corr_se, fock_mf, my_tddft)*self.h2ev
            
            if fock == simple_fock:
                print((my_tddft, other_tddft))
                self.assertAlmostEqual(my_result, other_result[orbital_number], delta=1e-7)
            elif fock == fock_dft:
                self.assertAlmostEqual(my_result, other_result[orbital_number], delta=1e-7)

        # then do the case where you compare to the symmetric implementation
        elif comparison_td == 'symm_drpa':
            my_tddft = my_drpa
            other_tddft = symm_drpa
            
            other_result = g0w0(orbital_number, fock_mo, real_corr_se, fock_mf, other_tddft)*self.h2ev
            my_result = g0w0(orbital_number, fock_mo, real_corr_se, fock_mf, my_tddft)*self.h2ev

            if fock == simple_fock:
                print((my_tddft, other_tddft))
                self.assertAlmostEqual(my_result, other_result, delta=1e-7)
            elif fock == fock_dft:
                self.assertAlmostEqual(my_result, other_result, delta=1e-7)


        

class G0W0Test(G0W0TestBase):
    pass  # Empty class, we'll add test methods to it dynamically

def add_dynamic_tests():
    species = ['water']
    offsets = [-2, -1, 0, 1, 2]

    def create_g0w0_test(species, offset):
        def test(self):
            with self.subTest(species=species, offset=offset):
                self.run_g0w0_test(species, 'hf', simple_fock, 'drpa', 'symm_drpa', offset)
        return test


    for s in species:
        for o in offsets:
            g0w0_test_method = create_g0w0_test(s, o)
            g0w0_test_method.__name__ = f'test_g0w0_{s}_offset_{o}'
            setattr(G0W0Test, g0w0_test_method.__name__, g0w0_test_method)

if __name__ == '__main__':
    add_dynamic_tests()
    unittest.main(verbosity=2)
