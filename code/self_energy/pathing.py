import numpy as np
import matplotlib.pyplot as plt

# Define a range of frequencies
frequencies = np.linspace(0.41, 0.42, 50)  # Here, 50 frequencies from 0 to 5 are chosen as an example

# Compute correlation energies for each frequency
all_correlation_energies = np.zeros((molecule.nao, len(frequencies)))

for idx, freq in enumerate(frequencies):
    correlation_energies = real_corr_se(molecule, freq)
    all_correlation_energies[:, idx] = correlation_energies
# the energies and frequencies are currently in hartrees; change them to eV
frequencies *= 27.2114
all_correlation_energies *= 27.2114 
# Plotting
plt.figure(figsize=(12, 8))
# let us only plot the HOMO and LUMO
# find the number of orbitals
n_orbitals = molecule.nao_nr()
# find the number of occupied orbitals
n_occupied = molecule.nelectron//2
# find the number of virtual orbitals
n_virtual = n_orbitals - n_occupied
homo_index = n_occupied-1
lumo_index = n_occupied

# Plot HOMO and LUMO with labels for the legend
plt.plot(frequencies, all_correlation_energies[homo_index], label='HOMO')
plt.plot(frequencies, all_correlation_energies[lumo_index], label='LUMO')

plt.title('Diagonal of the Real Part of the Self Energy Correlation vs. Frequency')
plt.xlabel('Frequency [eV]')
plt.ylabel('Energy [eV]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # This will display the legend
plt.subplots_adjust(right=0.75)  # Adjust subplot parameters to give more space for the legend
plt.grid(True)
plt.savefig('correlation_energies.png', bbox_inches='tight')  # bbox_inches='tight' to ensure the legend is included in the save
