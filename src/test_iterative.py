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

    def run_g0w0_test(self, mf, fock, tddft):
        fock_mo = fock(self.mf)  # Use the provided Fock matrix function
        if mf == 'hf':
            mf = calculate_mean_field(self.molecule, 'hf')
        if mf == 'dft':
            mf = calculate_mean_field(self.molecule, 'dft')
        if tddft == 'dtda':
            my_tddft = my_dtda
            pyscf_tddft = tddft.dTDA(self.mf_hf)
        if tddft == 'drpa':
            my_tddft = my_drpa
            pyscf_tddft = tddft.dRPA(self.mf_hf)

    def test_g0w0(self):
        # Test your g0w0 function
        fock_mo = simple_fock(self.mf_hf)  # HF Fock matrix in MO basis
        my_tddft = my_dtda  # TDDFT object

        # Call your g0w0 function
        result = g0w0(self.orbital_number, fock_mo, real_corr_se, self.mf_hf, my_tddft)*self.h2ev

        # Call the corresponding implementation in PySCF
        td = tddft.dTDA(self.mf_hf)
        # td = tddft.dRPA(mf_hf)
        td.nstates = self.n_occupied*self.n_virtual
        e, xy = td.kernel()
        # Make a fake Y vector of zeros
        td_xy = list()
        for xy in td.xy:
            x,y = xy
            td_xy.append((x,0*x))
        td.xy = td_xy
        pyscf_gw = gw.GW(self.mf_hf, freq_int='exact', tdmf=td)
        pyscf_gw.kernel(orbs=[self.orbital_number])
        expected_result = pyscf_gw.mo_energy*self.h2ev


        # Compare the results
        self.assertAlmostEqual(result, expected_result[self.orbital_number], delta=1e-8)

if __name__ == '__main__':
    unittest.main()