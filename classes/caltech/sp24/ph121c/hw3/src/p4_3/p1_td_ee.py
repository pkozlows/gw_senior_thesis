import numpy as np
import matplotlib.pyplot as plt
from hw3.src.p4_1.fns import make_product_state, time_dependent_state
from hw2.src.p5_2 import entanglement_entropy, calculate_reduced_density_matrix
from hw3.src.p4_3.fns import periodic_dense_hamiltonian_mbl

# Set system parameters
L_values = [6, 8, 10]
W = 3
t_values = np.linspace(0, 50, 20)
number_realizations = 10

for L in L_values:
    average_entropy_values1 = np.zeros(len(t_values))
    average_entropy_values2 = np.zeros(len(t_values))
    for realization in range(number_realizations):
        # Generate the Hamiltonian
        H = periodic_dense_hamiltonian_mbl(L, W)
        # Diagonalize the Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Initial state: tensor product of single_site across all sites
        single_site1 = np.array([1, -np.sqrt(3)]) / 2
        initial_state1 = make_product_state(single_site1, L)
        # make a second state
        single_site2 = np.array([-2, 1]) / np.sqrt(5)
        initial_state2 = make_product_state(single_site2, L)

        # Calculate the overlap coefficients
        overlap_coefficients1 = np.dot(eigenvectors.conj().T, initial_state1)
        overlap_coefficients2 = np.dot(eigenvectors.conj().T, initial_state2)

        # initialize a list of entropy values
        entropy_values1 = []
        entropy_values2 = []

        for t in t_values:
            # Compute the time-dependent state
            state1 = time_dependent_state(t, overlap_coefficients1, eigenvalues, eigenvectors)
            state2 = time_dependent_state(t, overlap_coefficients2, eigenvalues, eigenvectors)

            # Compute the reduced density matrix
            reduced_density_matrix1 = calculate_reduced_density_matrix(state1, L, L // 2)
            reduced_density_matrix2 = calculate_reduced_density_matrix(state2, L, L // 2)

            # Compute the entanglement entropy
            entropy1 = entanglement_entropy(reduced_density_matrix1)
            entropy2 = entanglement_entropy(reduced_density_matrix2)
            
            entropy_values1.append(entropy1)
            entropy_values2.append(entropy2)

        average_entropy_values1 += np.array(entropy_values1)
        average_entropy_values2 += np.array(entropy_values2)

    average_entropy_values1 /= number_realizations
    average_entropy_values2 /= number_realizations

    # Prepare to plot
    plt.figure(figsize=(10, 8))
    plt.title(f"System size L={L}")
    plt.xlabel("Time")
    plt.ylabel("Avg. Entanglement entropy")
    
    plt.plot(t_values, average_entropy_values1, label="Entanglement entropy for state 1")
    plt.plot(t_values, average_entropy_values2, label="Entanglement entropy for state 2")
    plt.legend()
    plt.grid()
    plt.savefig(f"hw3/docs/images/p4_3_1_avg_tdee_L={L}.png")
    plt.close()
