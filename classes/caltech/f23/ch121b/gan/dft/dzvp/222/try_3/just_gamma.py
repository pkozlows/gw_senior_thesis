import numpy as np
import matplotlib.pyplot as plt
import os

# Load the saved data
os.chdir('/Users/patrykkozlowski/caltech/classes/caltech/f23/ch121b/gan/dft/dzvp/222/try_3')
band_kpts = np.load('band_kpts.npy')
sp_points = np.load('sp_points.npy')
e_kn_2 = np.load('e_kn_2.npy')
au2ev = 27.21139

# Find the index of the k-point closest to the Γ point
k_idx = np.argmin(np.linalg.norm(band_kpts - np.array([sp_points[3], 0, 0]), axis=1))

# Extract the energies of all bands at this k-point
energies_at_Gamma = e_kn_2[k_idx] * au2ev

# Extract the energy value for band 18 at the Γ point
energy_band_18 = energies_at_Gamma[18]
print(f"Energy of Band 18 at Γ point: {energy_band_18:.2f} eV")

# Plotting
plt.figure(figsize=(5, 6))
plt.plot(range(len(energies_at_Gamma)), energies_at_Gamma, 'o', label="Band Energies at Γ")
plt.axhline(y=0, color='k', linestyle='-', label="Fermi Level")  # Fermi energy line
plt.xlabel('Band Index')
plt.ylabel('Energy [eV]')
plt.title('Band Energies at Γ Point')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('energies_at_gamma.png')
# plt.show()
