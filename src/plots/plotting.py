import numpy as np
import matplotlib.pyplot as plt
from iterative import g0w0
from tda import real_corr_se, symm_drpa
from mf import setup_molecule, calculate_mean_field
from fock import simple_fock

# Constants
conversion_factor = 27.2114  # Conversion factor from Hartree to eV
frequencies_ev = np.linspace(-25, 25, 100)  # Frequency range in eV for the plot
offsets = [-3, -2, -1, 0, 1, 2, 3]  # Orbital offsets

# Set up molecule and calculate mean field
mol_hcl = setup_molecule('hcl')
mf_hcl = calculate_mean_field(mol_hcl, 'hf')
my_fock_hcl = simple_fock(mf_hcl)
n_orbitals_hcl = mol_hcl.nao_nr()
n_occupied_hcl = mol_hcl.nelectron // 2

# Loop through each offset to generate and save plots
for offset in offsets:
    # Calculate the index of the orbital of interest
    orbital_index = n_occupied_hcl + offset  # Adjust index for HOMO + offset

    # Compute GW correction for the selected orbital
    iterative_solution = g0w0(orbital_index, my_fock_hcl, real_corr_se, mf_hcl, symm_drpa) * conversion_factor
    print(f"Iterative GW Solution for HCl (HOMO + {offset}): {iterative_solution}")

    # Correlation energies calculation
    frequencies = frequencies_ev / conversion_factor
    all_correlation_energies = np.array([real_corr_se(freq, symm_drpa, mf_hcl) for freq in frequencies])
    all_correlation_energies_ev = all_correlation_energies * conversion_factor  # Convert energies back to eV

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(frequencies_ev, all_correlation_energies_ev[:, orbital_index], label=f'Orbital Index HOMO + {offset}')

    # Add the line y = x + b, converting b to eV as well for HCl
    fock_element_ev_hcl = my_fock_hcl[orbital_index, orbital_index] * conversion_factor
    plt.plot(frequencies_ev, frequencies_ev - fock_element_ev_hcl, label=r'$\omega - \epsilon_p^{HF}$ (HCl)')

    plt.axvline(x=iterative_solution, color='red', linestyle='--', label=f'Solution at {iterative_solution:.2f} eV')
    plt.annotate(rf'$\omega$={iterative_solution:.2f} eV (HCl)', 
                 xy=(iterative_solution, 0), 
                 xytext=(10, 30), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle='->', color='black'))

    plt.xlabel(r'$\omega$ [eV]')
    plt.ylabel(r'$\Sigma_{pp}^{c}$ [eV]')
    plt.title(f'Correlation Energy vs. Frequency for HCl Orbital Index {orbital_index}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'correlation_energies_hcl_orbital_{orbital_index}.png', bbox_inches='tight')
    plt.close()  # Close the plot to avoid displaying it in interactive environments
