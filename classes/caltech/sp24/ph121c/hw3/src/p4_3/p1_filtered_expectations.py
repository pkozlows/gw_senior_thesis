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
W = 3  # Disorder strength
num_realizations = 20  # Number of disorder realizations

# Prepare the plot
plt.figure(figsize=(10, 6))
plt.title('Averaged Observable Expectation Values vs. Energy Density')
plt.xlabel('Energy density ($\\epsilon_n / L$)')
plt.ylabel(f'Expectation Value ($\\langle\\sigma_1^\\mu\\rangle_n$) with {num_realizations} realizations')

# Define observables
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.identity(2)

# Loop over system sizes
for i, L in enumerate(L_values):
    # Initialize arrays to store averaged expectation values for each observable
    averaged_expectations = [np.zeros((2**L,)) for _ in range(3)]  # For sigma_x, sigma_y, sigma_z

    # Loop over disorder realizations
    for _ in range(num_realizations):
        H = periodic_dense_hamiltonian_mbl(L, W)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        full_observables = [tensor_product([sigma] + [identity] * (L - 1)) for sigma in [sigma_x, sigma_y, sigma_z]]

        # Compute expectation values for each observable
        for j, observable in enumerate(full_observables):
            expectation_values = []
            for index in range(len(eigenvalues)):
                result = compute_observable_expectation_eigenvalue(index, observable, eigenvectors)
                expectation_values.append(result)

            # Accumulate the results for averaging
            averaged_expectations[j] += np.array(expectation_values, dtype=np.float64) / num_realizations

    # Plotting the averaged expectation values
    for j in range(3):
        plt.plot(np.array(eigenvalues) / L, averaged_expectations[j], label=f'L={L}, $\\sigma_{{1}}^{{{observable_labels[j]}}}$', color=colors[j], linestyle=line_styles[i])

# Add legend, grid, and save the plot
plt.legend()
plt.grid(True)
plt.savefig("hw3/docs/images/p4_3_1_filtered_expectations_average.png")
