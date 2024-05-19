import matplotlib.pyplot as plt
import numpy as np
from hw3.src.p4_2.fns import translation_operator, identify_k0_sector
from hw3.src.p4_1.fns import periodic_dense_hamiltonian
from hw1.src.hw1 import tensor_product
from hw2.src.p5_2 import entanglement_entropy, calculate_reduced_density_matrix
# Define line styles and colors for different system sizes and observables
line_styles = ['-', '--', ':']
colors = ['blue', 'green', 'red']  # Colors for sigma_x, sigma_y, sigma_z
observable_labels = ['x', 'y', 'z']

# System sizes
L_values = [6, 8, 10]

# Constants
h_x = -1.05
h_z = 0.5

# Prepare the plot
plt.figure(figsize=(10, 6))
plt.title('Entropic Signature of Thermalization')
plt.xlabel(f'Normalized Eigenvalue ($\epsilon_n / L$)')
plt.ylabel(f'Normalized Entanglement Entropy ($S_n / L$)')

# Loop over system sizes
for i, L in enumerate(L_values):
    # Generate and diagonalize the Hamiltonian
    H = periodic_dense_hamiltonian(L, h_x, h_z)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Define the translation operator for the system
    translation_op = translation_operator(L)

    # Identify the k=0 sector
    k0_sector = identify_k0_sector(eigenvectors, translation_op)

    expectation_values = []
    eigenvalues_k0 = []
    for _, k0_index in enumerate(k0_sector):
        # Compute the reduced density matrix
        reduced_density_matrix = calculate_reduced_density_matrix(eigenvectors[:, k0_index], L, L // 2)
        # Compute the entanglement entropy
        entropy = entanglement_entropy(reduced_density_matrix)
        expectation_values.append(entropy)
        eigenvalue_k0 = eigenvalues[k0_index]
        eigenvalues_k0.append(eigenvalue_k0)
        
    plt.plot(np.array(eigenvalues_k0) / L, np.array(expectation_values) / L, color=colors[i])
    # add a label for the color of the curve
    plt.plot([], [], color=colors[i], label=f'L={L}')

# Add legend and show plot
plt.legend()
plt.grid()
plt.savefig(f"hw3/docs/images/p4_2_2_entropic_signature.png")
