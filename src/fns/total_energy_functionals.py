import numpy as np
import pyscf
from pyscf import tddft

# Custom imports for setting up and calculating molecular properties
from src.fns.mf import setup_molecule, calculate_mean_field
from src.fns.dm import lin_gw_dm
from src.fns.tda import my_dtda, my_drpa
from src.fns.pyscf_tddft_fn import pyscf_td

def calculate_correlation_energy_gm(mf, td):
    """Calculates the correlation energy using the GM formula."""
    excitation_energies, transition_vectors = td(mf)
    orbital_energies = mf.mo_energy
    n_occ = mf.mol.nelectron // 2
    n_virt = len(orbital_energies) - n_occ

    denominator = orbital_energies[n_occ:, None, None] - orbital_energies[:n_occ, None, None] + excitation_energies[None, None, :]
    ov_num = np.square(transition_vectors[n_occ:, :n_occ, :])
    vo_num = np.swapaxes(np.square(transition_vectors[:n_occ, n_occ:, :]), 0, 1)

    e_corr = -0.5 * (np.sum(ov_num / denominator) + np.sum(vo_num / denominator))
    return e_corr

def calculate_correlation_energy_klein(omega_rpa, omega_tda):
    """Calculates the correlation energy using the Klein formula."""
    return 0.5 * np.sum(omega_rpa - omega_tda)

def total_energy(mf, rpa, tda, correlation_func, is_dft=False):
    """Calculates the total molecular energy."""
    if is_dft:
        hf_mf = mf.to_rhf()
        vhf = hf_mf.get_veff()
    else:
        vhf = mf.get_veff()

    hcore = mf.get_hcore() if is_dft else mf.mol.intor_symmetric('int1e_kin') + mf.mol.intor_symmetric('int1e_nuc')
    dm = mf.make_rdm1()
    e_elec, _ = mf.energy_elec(dm, h1e=hcore, vhf=vhf)
    e_tot = e_elec + mf.energy_nuc() + correlation_func(rpa(mf)[0], tda(mf)[0])

    return e_tot

def run_energy_calculations(molecule, method, correlation_func, is_dft):
    """Run the energy calculations for a given molecule using specified method and correlation function."""
    noninteracting_mf = calculate_mean_field(setup_molecule(molecule), method)
    non_interacting_energy = total_energy(noninteracting_mf, my_drpa, my_dtda, lambda *args: 0, is_dft)  # No correlation effect
    interacting_mf = noninteracting_mf.copy()
    interacting_energy = total_energy(interacting_mf, my_drpa, my_dtda, correlation_func, is_dft)

    return non_interacting_energy, interacting_energy


# Mapping of correlation function names to their corresponding function implementations
correlation_funcs = {
    'klein_interacting': calculate_correlation_energy_klein,
    'klein_noninteracting': calculate_correlation_energy_klein,
    'gm': calculate_correlation_energy_gm
}

molecules = ['water', 'hcl', 'nh3', 'lih', 'co']
methods = ['hf', 'dft']

# Running the calculations
for molecule in molecules:
    for correlation_name in correlation_funcs:
        for method in methods:
            is_dft = method == 'dft'
            mf_energy, gw_energy = run_energy_calculations(molecule, method, correlation_funcs[correlation_name], is_dft)
            print(f'{molecule} with {method} and {correlation_name}: ΔE = {gw_energy - mf_energy} eV.')

