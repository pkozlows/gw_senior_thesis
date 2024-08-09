import numpy as np
import pandas as pd
import pyscf
from pyscf import tddft
import pyscf.cc
import pyscf.gw.rpa

# Custom imports for setting up and calculating molecular properties
from src.fns.mf import setup_molecule, calculate_mean_field
from src.fns.dm import lin_gw_dm
from src.fns.tda import my_dtda, my_drpa, symm_drpa

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

        kinetic_integrals_ao = noninteracting.mol.intor('int1e_kin')
        # rotate the kinetic integrals to the MO basis using the noninteracting MO coefficients
        kinetic_integrals_mo = np.einsum('ij,ik,jl->kl', kinetic_integrals_ao, noninteracting.mo_coeff, noninteracting.mo_coeff)
        # rotate back to ao basis using the interacting MO coefficients
        final_kinetic_integrals = np.einsum('ij,ai,bj->ab', kinetic_integrals_mo, noninteracting.mo_coeff, noninteracting.mo_coeff)

        nuclear_integrals_ao = noninteracting.mol.intor('int1e_nuc')
        nuclear_integrals_mo = np.einsum('ij,ik,jl->kl', nuclear_integrals_ao, noninteracting.mo_coeff, noninteracting.mo_coeff)
        final_nuclear_integrals = np.einsum('ij,ai,bj->ab', nuclear_integrals_mo, interacting.mo_coeff, interacting.mo_coeff)
        # we want hcore in ao basis for later use wwithin energy_elec
        hcore = final_kinetic_integrals + final_nuclear_integrals
    else:
        hcore = noninteracting.get_hcore()

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

# Define the input variables that we will interacting over
species = ['water', 'nh3', 'lih']
methods = ['hf']
correlation_funcs = ['gm', 'klein_interacting', 'klein_noninteracting']

data = []

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

        for correlation_name in correlation_funcs:
            mf_energy = total_energy(noninteracting_mf, noninteracting_mf, correlation_name, mf=True)
            gw_total_energy, gw_correlation_energy = total_energy(noninteracting_mf, interacting_mf, correlation_name)
            print(f'HF energy for {molecule_name} with {correlation_name}: {mf_energy}')
            print(f'GW energy for {molecule_name} with {correlation_name}: {gw_total_energy}')
            
            # Get the CCSD(T) energy as a reference
            noninteracting_as_hf = noninteracting_mf.to_rhf()
            mycc = pyscf.cc.CCSD(noninteracting_as_hf).run(verbose=0)
            et = mycc.ccsd_t()
            ccsd_t_energy = mycc.e_tot + et
            print(f'CCSD(T) total energy for {molecule_name} with {method}: {ccsd_t_energy}')

            # Append results to the data list
            data.append({
                'Molecule': molecule_name,
                'HF Energy': mf_energy,
                'GW Energy': gw_total_energy,
                'CCSD(T) Energy': ccsd_t_energy,
                'Correlation Functional': correlation_name
            })

# Convert to a DataFrame and format molecule names
df = pd.DataFrame(data)
df['Molecule'] = df['Molecule'].replace({'water': 'H$_2$O', 'nh3': 'NH$_3$', 'lih': 'LiH'})

# Create separate tables for each correlation functional
for correlation_name in correlation_funcs:
    df_subset = df[df['Correlation Functional'] == correlation_name].drop(columns=['Correlation Functional'])
    latex_table = df_subset.to_latex(index=False, escape=False, float_format="%.6f")
    table_filename = f"energy_table_{correlation_name}.tex"
    
    with open(table_filename, "w") as f:
        f.write(latex_table)
    
    print(latex_table)
