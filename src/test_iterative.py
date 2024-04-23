import numpy as np
from tda import real_corr_se, my_drpa, my_dtda
from mf import setup_molecule, calculate_mean_field
from fock import simple_fock, fock_dft, pyscf_fock_dft
from iterative import g0w0
from pyscf import gw
from pyscf import tddft
import pyscf
from pyscf.dft import rks
from pyscf import scf
import unittest

class G0W0TestBase(unittest.TestCase):
    def setUp(self):
        # Set up common variables
        self.h2ev = 27.2114
    
    def run_drpa(self):
        omega, R = my_drpa(self.mf_hf)
        pyscf_tddft = tddft.dRPA(self.mf_hf)
        pyscf_tddft.nstates = self.n_occupied*self.n_virtual
        e, xy = pyscf_tddft.kernel()
        # Check whether omega and e are all the same within a tolerance
        self.assertTrue(np.allclose(omega, e, atol=1e-8), "omega and e arrays do not match")
    

    def run_g0w0_test(self, species, fock_mf, fock, td, offset):
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

        if td == 'dtda':
            my_tddft = my_dtda
            pyscf_tddft = tddft.dTDA(fock_mf)
        elif td == 'drpa':
            my_tddft = my_drpa
            pyscf_tddft = tddft.dRPA(fock_mf)
        
        fock_mo = fock(fock_mf)  # Use the provided Fock matrix function
        my_result = g0w0(orbital_number, fock_mo, real_corr_se, fock_mf, my_tddft)*self.h2ev

        pyscf_tddft.nstates = n_occupied*n_virtual
        e, xy = pyscf_tddft.kernel()
        # Make a fake Y vector of zeros
        td_xy = list()
        if td == 'drpa':
            for x, y in pyscf_tddft.xy:
                td_xy.append((x, y))
        elif td == 'dtda':
            for x, y in pyscf_tddft.xy:
                td_xy.append((x, 0*x))
        pyscf_tddft.xy = td_xy
        pyscf_gw = gw.GW(fock_mf, freq_int='exact', tdmf=pyscf_tddft)
        pyscf_gw.kernel(orbs=[orbital_number])
        expected_result = pyscf_gw.mo_energy*self.h2ev
        if fock == simple_fock:
            self.assertAlmostEqual(my_result, expected_result[orbital_number], delta=1e-7)
        elif fock == fock_dft:
            self.assertAlmostEqual(my_result, expected_result[orbital_number], delta=1e-7)

class G0W0Test(G0W0TestBase):
    pass  # Empty class, we'll add test methods to it dynamically

def add_dynamic_tests():
    species = ['water', 'nh3', 'methane]
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4]

    def create_test(species, offset):
        def test(self):
            with self.subTest(species=species, offset=offset):
                self.run_g0w0_test(species, 'hf', simple_fock, 'drpa', offset)
        return test

    for s in species:
        for o in offsets:
            test_method = create_test(s, o)
            test_method.__name__ = f'test_g0w0_{s}_offset_{o}'
            setattr(G0W0Test, test_method.__name__, test_method)

if __name__ == '__main__':
    add_dynamic_tests()
    unittest.main(verbosity=2)
