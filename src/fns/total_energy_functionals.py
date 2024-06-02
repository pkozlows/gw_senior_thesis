import numpy as np
import pyscf
from pyscf import tddft
import pyscf.cc
import pyscf.gw.rpa

# Custom imports for setting up and calculating molecular properties
from src.fns.mf import setup_molecule, calculate_mean_field
from src.fns.dm import lin_gw_dm
from src.fns.tda import my_dtda, my_drpa, symm_drpa
from src.fns.pyscf_tddft_fn import pyscf_td


class Correlation:
    def __init__(self, name, noninteracting, interacting):
        self.name = name
        self.noninteracting = noninteracting
        self.interacting = interacting

    def gm(self):
        excitation_energies, transition_densities = symm_drpa(self.noninteracting)
        orbital_energies = self.noninteracting.mo_energy
        n_occ = self.noninteracting.mol.nelectron // 2

        denominator = orbital_energies[n_occ:, None, None] - orbital_energies[None, :n_occ, None] + excitation_energies[None, None, :]
        ov_num = np.square(transition_densities[n_occ:, :n_occ, :])
        vo_num = np.swapaxes(np.square(transition_densities[:n_occ, n_occ:, :]), 0, 1)
        numerator = -2 * np.square(transition_densities[n_occ:, :n_occ, :])

        e_corr = (np.sum(numerator / denominator))
        return e_corr

    def klein_interacting(self):
        omega_tda = my_dtda(self.noninteracting)[0]
        omega_rpa = my_drpa(self.noninteracting)[0]
        return 0.5 * np.sum(omega_rpa - omega_tda)
    
    def klein_noninteracting(self):
        omega_tda = my_dtda(self.noninteracting)[0]
        omega_rpa = my_drpa(self.noninteracting)[0]
        return 0.5 * np.sum(omega_rpa - omega_tda)


def total_energy(noninteracting, interacting, correlation_name, mf=False):
    if correlation_name == 'klein_interacting':
        hcore = noninteracting.mol.intor_symmetric('int1e_kin') + interacting.mol.intor_symmetric('int1e_nuc')
    else:
        hcore = interacting.get_hcore()

    interacting_hf = interacting.to_rhf()
    vhf = interacting_hf.get_veff()
    dm = interacting.make_rdm1()
    e_elec, _ = interacting.energy_elec(dm, h1e=hcore, vhf=vhf)
    
    correlation_functional = Correlation(correlation_name, noninteracting, interacting)
    e_corr = getattr(correlation_functional, correlation_name)()
    if mf:
        return e_elec + noninteracting.energy_nuc()
    else:
        e_tot = e_elec + noninteracting.energy_nuc() + e_corr
        return e_tot, e_corr 
    


    


# Define the input variables that we will iterate over
species = ['water', 'nh3', 'lih']
methods = ['hf']
correlation_funcs = ['gm']

for molecule_name in species:
    for method in methods:
        
        noninteracting_mf = calculate_mean_field(setup_molecule(molecule_name), method)
        
        # Generate the interacting case (GW density matrix)
        my_gw_dm_mo = lin_gw_dm(my_dtda, noninteracting_mf)
        my_gw_occs, my_gw_mo = np.linalg.eigh(my_gw_dm_mo)
        my_gw_mo = my_gw_mo[:, ::-1]
        my_gw_occs = my_gw_occs[::-1]
        interacting_mf = noninteracting_mf.copy()
        interacting_mf.mo_coeff = np.einsum('ij,jk->ik', noninteracting_mf.mo_coeff, my_gw_mo)
        interacting_mf.mo_occ = my_gw_occs

        # Make a dictionary to tabulate the GW energy for different correlation functionals
        gw_energy_dict = {}
        for correlation_name in correlation_funcs:
            mf_energy = total_energy(noninteracting_mf, noninteracting_mf, correlation_name, mf=True)
            gw_total_energy, gw_correlation_energy = total_energy(noninteracting_mf, interacting_mf, correlation_name)
            gw_energy_dict[correlation_name] = gw_total_energy
            print(f'hf energy for {molecule_name} with {correlation_name}: {mf_energy}')
            print(f'GW energy for {molecule_name} with {correlation_name}: {gw_total_energy}')
            # print(f'GW correlation energy for {molecule_name} with {correlation_name}: {gw_correlation_energy}')
            # define the total and correlation energy from the pyscf rpa function
            # rpa = pyscf.gw.rpa.RPA(noninteracting_mf)
            # rpa.kernel()
            # tot_e = rpa.e_tot
            # cor_e = rpa.e_corr
            # print(f'Pyscf RPA total energy for {molecule_name} with {method}: {tot_e}')
            # print(f'Pyscf RPA correlation energy for {molecule_name} with {method}: {cor_e}')
            # get the ccsd(t) energy as a reference
            noninteracting_as_hf = noninteracting_mf.to_rhf()
            mycc = pyscf.cc.CCSD(noninteracting_as_hf).run(verbose=0)
            et = mycc.ccsd_t()
            ccsd_t_energy = mycc.e_tot + et
            print(f'CCSD(T) total energy for {molecule_name} with {method}: {ccsd_t_energy}')
            
        

