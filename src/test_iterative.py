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
        self.molecule = setup_molecule()
        self.mf_hf = calculate_mean_field(self.molecule, 'hf')
        self.mf_dft = calculate_mean_field(self.molecule, 'dft')
        self.n_orbitals = self.molecule.nao_nr()
        self.n_occupied = self.molecule.nelectron//2
        self.n_virtual = self.n_orbitals - self.n_occupied
        self.orbital_number = self.n_occupied - 1 # do it for the HOMO
        self.h2ev = 27.2114

    def run_g0w0_test(self, mf, fock, td):
        if mf == 'hf':
            mf = calculate_mean_field(self.molecule, 'hf')
        if mf == 'dft':
            mf = calculate_mean_field(self.molecule, 'dft')
        if td == 'dtda':
            my_tddft = my_dtda
            pyscf_tddft = tddft.dTDA(mf)
        if td == 'drpa':
            my_tddft = my_drpa
            pyscf_tddft = tddft.dRPA(mf)
        
        fock_mo = fock(mf)  # Use the provided Fock matrix function
        my_result = g0w0(self.orbital_number, fock_mo, real_corr_se, mf, my_tddft)*self.h2ev

        pyscf_tddft.nstates = self.n_occupied*self.n_virtual
        e, xy = pyscf_tddft.kernel()
        # Make a fake Y vector of zeros
        td_xy = list()
        for xy in pyscf_tddft.xy:
            x,y = xy
            td_xy.append((x,0*x))
        pyscf_tddft.xy = td_xy
        pyscf_gw = gw.GW(mf, freq_int='exact', tdmf=pyscf_tddft)
        pyscf_gw.kernel(orbs=[self.orbital_number])
        expected_result = pyscf_gw.mo_energy*self.h2ev

        self.assertAlmostEqual(my_result, expected_result[self.orbital_number], delta=1e-8)


class TestG0W0WithSimpleFockAndDTDA(G0W0TestBase):
    def test_g0w0(self):
        self.run_g0w0_test('hf', simple_fock, 'dtda')

class TestG0W0WithdftAndDTDA(G0W0TestBase):
    def test_g0w0(self):
        self.run_g0w0_test('dft', fock_dft, 'dtda')

# class TestG0W0WithpyscfDftFockAndDTDA(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('hf', pyscf_fock_dft, 'dtda')

# class TestG0W0WithmyDftFockAndDTDA(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('hf', fock_dft, 'dtda')

# class TestG0W0WithSimpleFockAndDRPA(G0W0TestBase):
#     def test_g0w0(self):
#         self.run_g0w0_test('hf', simple_fock, 'drpa')

 
if __name__ == '__main__':
     unittest.main()