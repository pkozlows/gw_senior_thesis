import numpy as np
import matplotlib.pyplot as plt
from iterative import g0w0
from tda import real_corr_se
from mf import setup_molecule, calculate_mean_field
from fock import fock_matrix_hf

molecule = setup_molecule()
my_fock = fock_matrix_hf(molecule)
# find the number of orbitals
n_orbitals = molecule.nao_nr()
# find the number of occupied orbitals
n_occupied = molecule.nelectron//2
# find the number of virtual orbitals
n_virtual = n_orbitals - n_occupied
homo_index = n_occupied-1
lumo_index = n_occupied
conversion_factor = 27.2114

iterative_solution = g0w0(homo_index, my_fock, real_corr_se) * conversion_factor

print("Iterative Solution", iterative_solution)
# Define a range of frequencies in eV and convert them to Hartrees for calculations
frequencies_ev = np.linspace(-50, 0, 100)
frequencies = frequencies_ev / conversion_factor  # Convert to Hartrees

# Compute correlation energies for each frequency
all_correlation_energies = np.zeros((molecule.nao, len(frequencies)))
for idx, freq in enumerate(frequencies):
    correlation_energies = real_corr_se(freq)
    all_correlation_energies[:, idx] = correlation_energies

# Convert frequencies and energies from Hartrees to electron volts
frequencies_ev = frequencies * conversion_factor
all_correlation_energies_ev = all_correlation_energies * conversion_factor

# Plotting
plt.figure(figsize=(12, 8))

# Plot HOMO (and optionally LUMO) in eV
plt.plot(frequencies_ev, all_correlation_energies_ev[homo_index], label='HOMO')
# plt.plot(frequencies_ev, all_correlation_energies_ev[lumo_index], label='LUMO')



plt.title('G0W0 @HF for Water')
plt.xlabel(r'$\omega$ [eV]')
plt.ylabel(r'$\operatorname{Re}(\Sigma_{pp}^{c})$ [eV]')

# Add the line y = x + b, converting b to eV as well
fock_element_ev = my_fock[homo_index, homo_index] * conversion_factor
print('Fock element', fock_element_ev)
plt.plot(frequencies_ev, frequencies_ev - fock_element_ev, label=r'$\omega - \varepsilon_0^{HF}$')
# # find the x value add which the lines intersect
# # Difference between the HOMO curve and the omega - epsilon_0^HF line
# difference = all_correlation_energies_ev[homo_index] - (frequencies_ev - fock_element_ev)

# # Find the index where the difference changes sign
# # np.sign gives -1 for negative values and 1 for positive values, so a difference of signs will give -2 or 2
# sign_change_index = np.where(np.diff(np.sign(difference)))[0]
# # Add a gray dashed vertical line at the intersection frequency
# intersection_freq = frequencies_ev[sign_change_index]
# intersection_energy = all_correlation_energies_ev[homo_index, sign_change_index]
# print("Intersection Frequency", intersection_freq)

plt.axvline(x=iterative_solution, color='gray', linestyle='--', label='Solution')
plt.annotate(rf'$\omega$={iterative_solution:.2f} eV', 
                 xy=(iterative_solution, 2), 
                 xytext=(10, 10), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle='->', color='black'))
plt.xlim(frequencies_ev[0], frequencies_ev[-1])
plt.ylim(min(all_correlation_energies_ev[homo_index]) - 5, max(all_correlation_energies_ev[homo_index]) + 5)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.75)
plt.grid(True)
plt.savefig('correlation_energies.png', bbox_inches='tight')  # bbox_inches='tight' to ensure the legend is included in the save
