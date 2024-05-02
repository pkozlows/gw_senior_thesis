import numpy as np
import pyscf.gw
from tda import real_corr_se, my_drpa, my_dtda, symm_drpa
from mf import setup_molecule, calculate_mean_field
from fock import simple_fock, fock_dft
from iterative import g0w0
from pyscf import gw
from pyscf import tddft
import pyscf
import numpy as np
from tda import real_corr_se, my_drpa, my_dtda, symm_drpa
from mf import setup_molecule, calculate_mean_field
import unittest

class G0W0TestBase(unittest.TestCase):
    def setUp(self):
        self.h2ev = 27.2114

    def compare_drpa_methods(self, species, fock, offset):
        molecule = setup_molecule(species)
        fock_mf = calculate_mean_field(molecule, 'hf')
        n_orbitals = fock_mf.mol.nao_nr()
        n_occupied = fock_mf.mol.nelectron // 2
        n_virtual = n_orbitals - n_occupied

        def extract_and_reshape_excitation_vectors(xy):
            # Initialize an empty list to store each reshaped vector
            reshaped_vectors = []

            # Loop over each element in the xy list
            for vec_tuple in xy:
                # Extract the first element of the tuple, which is the vector
                vec = vec_tuple[1]

                # Ensure the vector is reshaped to (nocc, nvirt)
                reshaped_vec = np.array(vec).reshape(n_occupied, n_virtual)

                # Append the reshaped vector to the list
                reshaped_vectors.append(reshaped_vec)

            # Stack all reshaped vectors along a new axis to form the final array
            # The new array will have shape (nocc, nvirt, nocc*nvirt)
            final_array = np.stack(reshaped_vectors, axis=2)

            return final_array

        omega_drpa, R_drpa = my_drpa(fock_mf)

        omega_symm_drpa, R_symm_drpa = symm_drpa(fock_mf)
        # get the excitation vectors into the appropriate shape
        symm_excitation_vectors = R_symm_drpa[:n_occupied, :n_virtual, :]

        omega_pyscf_drpa = tddft.dRPA(fock_mf)
        omega_pyscf_drpa.nstates = n_occupied * n_virtual
        omega_pyscf_drpa.kernel()
        omega_pyscf_drpa_energies = omega_pyscf_drpa.e
        pyscf_excitation_vectors_drpa = omega_pyscf_drpa.xy
        pyscf_excitation_vectors_drpa = extract_and_reshape_excitation_vectors(pyscf_excitation_vectors_drpa)



        for o1, o2 in zip(np.sort(omega_pyscf_drpa_energies), np.sort(omega_symm_drpa)):
            self.assertAlmostEqual(o1, o2, msg="PySCF and Symmetric DRPA excitation energies differ")

        for v1, v2 in zip(pyscf_excitation_vectors_drpa, symm_excitation_vectors):
            for vec1, vec2 in zip(v1, v2):
                for el1, el2 in zip(vec1, vec2):
                    self.assertAlmostEqual(el1, el2, msg="PySCF and Symmetric DRPA excitation vectors differ")

    def cd_gw_vs_exact_gw(self, species, offset):
        # perform setup
        molecule = setup_molecule(species)
        fock_mf = calculate_mean_field(molecule, 'hf')
        n_orbitals = fock_mf.mol.nao_nr()
        n_occupied = fock_mf.mol.nelectron // 2
        n_virtual = n_orbitals - n_occupied
        orbital_number = n_occupied + offset
        
        # create the tddft object
        td = tddft.dRPA(fock_mf)
        td.nstates = n_occupied * n_virtual
        td.kernel()

        # get the exact GW energy
        gw_exact = gw.GW(fock_mf, freq_int='exact', tdmf=td)
        gw_exact.kernel(orbs=[orbital_number])
        gw_exact_energy = gw_exact.mo_energy * self.h2ev

        # get the cd GW energy
        gw_cd = pyscf.gw.gw_cd.GWCD(fock_mf)
        gw_cd.kernel(orbs=[orbital_number], nw=128)
        gw_cd_energy = gw_cd.mo_energy * self.h2ev

        # compare the two
        self.assertAlmostEqual(gw_exact_energy[orbital_number], gw_cd_energy[orbital_number], delta=1e-7)
        # print(gw_cd_energy[orbital_number])

        


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
                my_tddft = symm_drpa
                other_tddft = tddft.dRPA(fock_mf)
            
            other_tddft.nstates = n_occupied*n_virtual
            e, xy = other_tddft.kernel()
            # Make a fake Y vector of zeros
            td_xy = list()
            if td == 'drpa':
                for x, y in other_tddft.xy:
                    td_xy.append((x, y))
            if td == 'dtda':
                for x, y in other_tddft.xy:
                    td_xy.append((x, 0*x))
            other_tddft.xy = td_xy
            pyscf_gw = gw.GW(fock_mf, freq_int='exact', tdmf=other_tddft)
            pyscf_gw.kernel(orbs=[orbital_number])

            other_result = pyscf_gw.mo_energy*self.h2ev
            my_result = g0w0(orbital_number, fock_mo, real_corr_se, fock_mf, my_tddft)*self.h2ev
            
            if fock == simple_fock:
                try:
                    self.assertAlmostEqual(my_result, other_result[orbital_number], delta=1e-7)
                    print(f"PASS: My result = {my_result}, PySCF result = {other_result[orbital_number]}, Delta = {abs(my_result - other_result[orbital_number])}")
                except AssertionError:
                    print(f"FAIL: My result = {my_result}, PySCF result = {other_result[orbital_number]}, Delta = {abs(my_result - other_result[orbital_number])}")
                    raise
            elif fock == fock_dft:
                self.assertAlmostEqual(my_result, other_result[orbital_number], delta=1e-6)

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
                self.assertAlmostEqual(my_result, other_result, delta=1e-6)


        

class G0W0Test(G0W0TestBase):
    pass  # Empty class, we'll add test methods to it dynamically

def add_dynamic_tests():
    species = ['water', 'hcl', 'nh3', 'lih', 'co']
    offsets = [-2, -1, 0, 1, 2, 3]

    def create_drpa_test(species, offset):
        def test(self):
            with self.subTest(species=species, offset=offset):
                self.compare_drpa_methods(species, simple_fock, offset)
        return test

    def create_g0w0_test(species, offset):
        def test(self):
            with self.subTest(species=species, offset=offset):
                self.run_g0w0_test(species, 'hf', simple_fock, 'drpa', 'pyscf', offset)
        return test
    
    def create_cd_gw_vs_exact_gw_test(species, offset):
        def test(self):
            with self.subTest(species=species, offset=offset):
                self.cd_gw_vs_exact_gw(species, offset)
        return test


    for s in species:
        for o in offsets:
            g0w0_test_method = create_g0w0_test(s, o)
            g0w0_test_method.__name__ = f'test_g0w0_{s}_offset_{o}'
            setattr(G0W0Test, g0w0_test_method.__name__, g0w0_test_method)

            pyscf_cd_gw_vs_exact_gw_test_method = create_cd_gw_vs_exact_gw_test(s, o)
            pyscf_cd_gw_vs_exact_gw_test_method.__name__ = f'test_cd_gw_vs_exact_gw_{s}_offset_{o}'
            # setattr(G0W0Test, pyscf_cd_gw_vs_exact_gw_test_method.__name__, pyscf_cd_gw_vs_exact_gw_test_method)

            energy_test_method = create_drpa_test(s, o)
            energy_test_method.__name__ = f'test_drpa_{s}_offset_{o}'
            # setattr(G0W0Test, energy_test_method.__name__, energy_test_method)

if __name__ == '__main__':
    add_dynamic_tests()
    unittest.main(verbosity=2)
