import matplotlib.pyplot as plt
import numpy as np
from hw3.src.p4_2.fns import compute_observable_expectation_eigenvalue
from hw1.src.hw1 import tensor_product
from hw3.src.p4_3.fns import periodic_dense_hamiltonian_mbl

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
plt.title('Observable Expectation Values vs Normalized Eigenvalues for Different L')
plt.xlabel(f'Normalized Eigenvalue ($\epsilon_n / L$)')
plt.ylabel(f'Expectation Value ($\langle\sigma_1^\mu\\rangle_n$)')

# Loop over system sizes
for i, L in enumerate(L_values):
    # Generate and diagonalize the Hamiltonian
    H = periodic_dense_hamiltonian_mbl(L, h_x, h_z)
    eigenvalues, eigenvectors = np.linalg.eigh(H)


    # Define observables
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity = np.identity(2)
    full_observables = [tensor_product([sigma] + [identity] * (L - 1)) for sigma in [sigma_x, sigma_y, sigma_z]]

    # Compute and plot expectation values for each observable
    for j, observable in enumerate(full_observables):
        expectation_values = []
        for index in range(len(eigenvalues)):
            observable = compute_observable_expectation_eigenvalue(index, observable, eigenvalues, eigenvectors)
            expectation_values.append(observable)
        
        plt.plot(np.array(eigenvalues) / L, expectation_values, label=f'L={L}, $\sigma_{{1}}^{{ {observable_labels[j]} }}$', 
                 color=colors[j], linestyle=line_styles[i])

# Add legend and show plot
plt.legend()
plt.grid(True)
plt.savefig(f"hw3/docs/images/p4_3_1_filtered_expectations.png")
