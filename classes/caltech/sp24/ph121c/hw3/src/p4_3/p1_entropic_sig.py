import matplotlib.pyplot as plt
import numpy as np
from hw3.src.p4_3.fns import periodic_dense_hamiltonian_mbl
from hw1.src.hw1 import tensor_product
from hw2.src.p5_2 import entanglement_entropy, calculate_reduced_density_matrix

# Define line styles and colors for different system sizes and observables
colors = ['blue', 'green', 'red']  # Colors for L=6, L=8, L=10

# System sizes and disorder realizations
L_values = [6, 8, 10]
num_realizations = 30  # Number of disorder realizations
W = 3  # Disorder strength

# Prepare the plot
plt.figure(figsize=(10, 6))
plt.title('Averaged Entropic Signature of Thermalization for MBL Hamiltonian')
plt.xlabel('Normalized Eigenvalue (ε_n / L)')
plt.ylabel('Averaged Normalized Entanglement Entropy (S_n / L)')

# Loop over system sizes
for i, L in enumerate(L_values):
    averaged_entropies = np.zeros((2**L,))  # Initialize array to store averaged entropies

    # Loop over disorder realizations
    for _ in range(num_realizations):
        H = periodic_dense_hamiltonian_mbl(L, W)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        entropies = []

        # Compute entropies for each eigenstate
        for eigenvector in eigenvectors.T:
            rho = calculate_reduced_density_matrix(eigenvector, L, L // 2)
            entropy = entanglement_entropy(rho)
            entropies.append(entropy)

        averaged_entropies += np.array(entropies) / num_realizations

    # Plotting the averaged entropies
    plt.plot(np.array(eigenvalues) / L, averaged_entropies / L, color=colors[i], label=f'L={L}')

# Add legend, grid, and save the plot
plt.legend()
plt.grid(True)
plt.savefig("hw3/docs/images/p4_3_entropic_signature_average.png")
