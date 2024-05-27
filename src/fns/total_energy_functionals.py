import numpy as np
import pyscf
from pyscf import tddft

# Custom imports for setting up and calculating molecular properties
from src.fns.mf import setup_molecule, calculate_mean_field
from src.fns.dm import lin_gw_dm
from src.fns.tda import my_dtda, my_drpa
from src.fns.pyscf_tddft_fn import pyscf_td


class Correlation:
    def __init__(self, name, noninteracting, interacting):
        self.name = name
        self.noninteracting = noninteracting
        self.interacting = interacting

    def gm(self):
        excitation_energies, transition_vectors = my_dtda(self.noninteracting)
        orbital_energies = self.noninteracting.mo_energy
        n_occ = self.noninteracting.mol.nelectron // 2

        denominator = orbital_energies[n_occ:, None, None] - orbital_energies[None, :n_occ, None] + excitation_energies[None, None, :]
        ov_num = np.square(transition_vectors[n_occ:, :n_occ, :])
        vo_num = np.swapaxes(np.square(transition_vectors[:n_occ, n_occ:, :]), 0, 1)

        e_corr = -0.5 * (np.sum(ov_num / denominator) + np.sum(vo_num / denominator))
        return e_corr

    def klein_interacting(self):
        omega_tda = my_dtda(self.interacting)[0]
        omega_rpa = my_drpa(self.interacting)[0]
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
        return e_elec + interacting.energy_nuc()
    else:
        e_tot = e_elec + interacting.energy_nuc() + e_corr
    
    return e_tot


# Define the input variables that we will iterate over
species = ['water', 'hcl', 'nh3', 'lih', 'co']
methods = ['hf', 'dft']
correlation_funcs = ['klein_interacting', 'klein_noninteracting', 'gm']

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
            gw_energy = total_energy(noninteracting_mf, interacting_mf, correlation_name)
            gw_energy_dict[correlation_name] = gw_energy
            print(f'{molecule_name} with {method} and {correlation_name}: ΔE = {gw_energy - mf_energy} eV.')
        
        # Print out the difference between the two Klein functionals
        if "klein_interacting" in gw_energy_dict and "klein_noninteracting" in gw_energy_dict:
            pass
            # print(f'{molecule_name} with {method}: ΔE (klein_interacting - klein_noninteracting) = {gw_energy_dict["klein_interacting"] - gw_energy_dict["klein_noninteracting"]} eV.')
        else:
            print(f'Error: Missing klein_interacting or klein_noninteracting energy values for {molecule_name} with {method}')
