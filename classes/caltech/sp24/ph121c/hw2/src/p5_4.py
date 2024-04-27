from hw1 import periodic_dense_hamiltonian_explicit
from p5_2 import entanglement_entropy, calculate_reduced_density_matrix
import numpy as np
import matplotlib.pyplot as plt

h = 1.7
L_sizes = [6, 8, 10, 12]  # Different system sizes
entropies = {L: [] for L in L_sizes}  # Dictionary to store entropies by L

for L in L_sizes:
    H = periodic_dense_hamiltonian_explicit(L, h)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Select a eigenstate near the middle of the spectrum
    mid_index = len(eigenvalues) // 2
    # make sure that it is not degenerate
    if eigenvalues[mid_index] != eigenvalues[mid_index + 1] and eigenvalues[mid_index] != eigenvalues[mid_index - 1]:
        state = eigenvectors[:, mid_index]
    else:
        # If the middle index is degenerate, shift slightly to find a non-degenerate state
        offset = 1
        while (mid_index + offset < len(eigenvalues) and eigenvalues[mid_index] == eigenvalues[mid_index + offset]) or \
              (mid_index - offset >= 0 and eigenvalues[mid_index] == eigenvalues[mid_index - offset]):
            offset += 1
        state = eigenvectors[:, mid_index + offset]

    # Compute the entanglement entropy for each ell using the same procedure as in p5_2.py
    for ell in range(1, L):
        rho_A = calculate_reduced_density_matrix(state, L, ell)
        entropy = entanglement_entropy(rho_A)
        entropies[L].append((ell, entropy))  # Store ell and entropy

# Plotting the results
plt.figure(figsize=(10, 6))
for L in L_sizes:
    ells, L_entropies = zip(*entropies[L])  # Unpack ell and entropy values for each L
    plt.plot(ells, L_entropies, marker='o', linestyle='-', label=f'L = {L}')

plt.xlabel('Subsystem Size ell')
plt.ylabel('Entanglement Entropy S')
plt.title('Entanglement Entropy for Eigenstates in the Middle of the Spectrum')
plt.legend()
plt.grid(True)
plt.savefig('entanglement_entropy_middle_spectrum.png')
