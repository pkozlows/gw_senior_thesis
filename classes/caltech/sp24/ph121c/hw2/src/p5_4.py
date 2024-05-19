from hw1 import periodic_dense_hamiltonian_explicit
from p5_2 import entanglement_entropy, calculate_reduced_density_matrix
import numpy as np
import matplotlib.pyplot as plt

h = 1.7
L_sizes = [8, 10, 12]  # Different system sizes
entropies = {L: {i: [] for i in range(3)} for L in L_sizes}  # Dictionary to store entropies by L and index

for L in L_sizes:
    H = periodic_dense_hamiltonian_explicit(L, h)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Select three eigenstates near the middle of the spectrum
    mid_indices = [len(eigenvalues) // 2 - 1, len(eigenvalues) // 2, len(eigenvalues) // 2 + 1]
    
    for i, mid_index in enumerate(mid_indices):
        state = eigenvectors[:, mid_index]
        
        # Compute the entanglement entropy for each ell using the same procedure as in p5_2.py
        for ell in range(1, L):
            rho_A = calculate_reduced_density_matrix(state, L, ell)
            entropy = entanglement_entropy(rho_A)
            entropies[L][i].append((ell, entropy))  # Store ell and entropy for each index

# Plotting the results
line_styles = ['-', '--', '-.']  # Different line styles for the same system size
colors = ['b', 'g', 'r']  # Different colors for different system sizes

plt.figure(figsize=(12, 8))
for j, L in enumerate(L_sizes):
    for i in range(3):
        ells, L_entropies = zip(*entropies[L][i])  # Unpack ell and entropy values for each L and index
        plt.plot(ells, L_entropies, marker='o', linestyle=line_styles[i], color=colors[j], label=f'L = {L}, index = {i+1}')

plt.xlabel(f'Subsystem Size $\ell$')
plt.ylabel('Entanglement Entropy S')
plt.title('Entanglement Entropy for Eigenstates in the Middle of the Spectrum')
plt.legend()
plt.grid(True)
plt.savefig('hw2/docs/images/entanglement_entropy_middle_spectrum.png')
