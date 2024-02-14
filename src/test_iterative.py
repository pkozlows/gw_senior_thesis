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
    
    # def fock(self):
    #     """Ensure that these folk maturities are equal"""
    #     pyscf_fock_hf = self.mf_hf.get_fock(dm=self.mf_hf.make_rdm1())
    #     pyscf_fock_hf_atdft = self.mf_hf.get_fock(dm=self.mf_dft.make_rdm1())
    #     my_fock = fock_dft(self.mf_dft)
    #     print(np.linalg.norm(pyscf_fock_hf), 'pyscf hf at hf dm')
    #     print(np.linalg.norm(pyscf_fock_hf_atdft), 'pyscf hf at dft dm')
    #     print(np.linalg.norm(my_fock), 'mine')

    def run_g0w0_test(self, molecule, fock_mf, fock, td):

        if fock_mf == 'hf':
            fock_mf = calculate_mean_field(molecule, 'hf')
        if fock_mf == 'dft':
            fock_mf = calculate_mean_field(molecule, 'dft')
        # set of the common variables
        n_orbitals = fock_mf.mol.nao_nr()
        n_occupied = fock_mf.mol.nelectron//2
        n_virtual = n_orbitals - n_occupied
        orbital_number = n_occupied - 1 # do it for the HOMO
        if td == 'dtda':
            my_tddft = my_dtda
            pyscf_tddft = tddft.dTDA(fock_mf)
        if td == 'drpa':
            my_tddft = my_drpa
            pyscf_tddft = tddft.dRPA(fock_mf)
        
        fock_mo = fock(fock_mf)  # Use the provided Fock matrix function
        my_result = g0w0(orbital_number, fock_mo, real_corr_se, fock_mf, my_tddft)*self.h2ev

        pyscf_tddft.nstates = n_occupied*n_virtual
        e, xy = pyscf_tddft.kernel()
        # Make a fake Y vector of zeros
        td_xy = list()
        if td == 'drpa':
            for xy in pyscf_tddft.xy:
                x,y=xy
                td_xy.append((x, y))
        elif td == 'dtda':
            for xy in pyscf_tddft.xy:
                x,y = xy
                td_xy.append((x,0*x))
        pyscf_gw = gw.GW(fock_mf)
        # pyscf_gw = gw.GW(fock_mf, freq_int='exact', tdmf=pyscf_tddft)
        pyscf_gw.kernel(orbs=[orbital_number])
        expected_result = pyscf_gw.mo_energy*self.h2ev

        self.assertAlmostEqual(my_result, expected_result[orbital_number], delta=1e-10)
class TestG0W0WithhfAndDRPA(G0W0TestBase):
    def test_g0w0(self):
        self.run_g0w0_test('water', 'hf', simple_fock, 'drpa')

# class TestG0W0WithSimpleFockAndDTDAh20(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('water', 'hf',simple_fock, 'dtda')

# class TestG0W0WithdftAndDTDAh20(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('water', 'dft', fock_dft, 'dtda')

# class TestG0W0WithSimpleFockAndDTDAmethane(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('methane', 'hf',simple_fock, 'dtda')

# class TestG0W0WithdftAndDTDAmethane(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('methane', 'dft', fock_dft, 'dtda')

# class TestG0W0WithSimpleFockAndDTDAh2(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('h2', 'hf',simple_fock, 'dtda')

# class TestG0W0WithdftAndDTDAh2(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('h2', 'dft', fock_dft, 'dtda')
        





# class Testdrpa(G0W0TestBase):
#     def test_drpa(self):
#         self.run_drpa()

#     def run_drpa(self):
#         omega, R = my_drpa(self.mf_hf)
#         pyscf_tddft = tddft.dRPA(self.mf_hf)
#         pyscf_tddft.nstates = self.n_occupied * self.n_virtual
#         e, xy = pyscf_tddft.kernel()

#         if not np.allclose(omega, e, atol=1e-8):
#             print("\nComparison failed. Printing omega and e arrays:")
#             print("omega:", omega)
#             print("e:", e)
#             self.fail("omega and e arrays do not match")



 
if __name__ == '__main__':
     unittest.main()