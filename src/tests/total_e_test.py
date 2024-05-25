import numpy as np
from src.fns.tda import real_corr_se, my_drpa, my_dtda, symm_drpa
from src.fns.mf import setup_molecule, calculate_mean_field
from src.fns.dm import lin_gw_dm
from src.fns.total_energy_functionals import total_energy
import unittest

class G0W0TestBase(unittest.TestCase):
    def setUp(self):
        self.h2ev = 27.2114
    
    def compute_total_energy(self, mf):
        '''Calculates the total energy of a molecule using the expression given in Bruneval. Takes in a density matrix and a mean field object.'''
        td = symm_drpa(mf)
        density_matrix = lin_gw_dm(td, mf)
        total_e = total_energy(density_matrix, mf)
        return total_e

# now make some test cases
class TestTotalEnergyG0W0(G0W0TestBase):
    def test_total_energy(self):
        molecule = setup_molecule('h2')
        fock_mf = calculate_mean_field(molecule, 'dft')
        total_energy = self.compute_total_energy(fock_mf)
        self.assertAlmostEqual(total_energy, -1.152, msg="Total energy of H2 in G0W0 approximation is incorrect.")

if __name__ == '__main__':
    unittest.main()





       