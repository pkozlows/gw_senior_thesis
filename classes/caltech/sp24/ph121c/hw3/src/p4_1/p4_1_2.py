import numpy as np
import matplotlib.pyplot as plt
from hw1.src.hw1 import tensor_product
from hw3.src.p4_1.fns import periodic_dense_hamiltonian, compute_observable_expectation, make_product_state, compute_thermal_energy, compute_thermal_observable
# Set system parameters
L_values = [8, 10]
h_x = -1.05
h_z = 0.5
beta_values = np.linspace(0, 1, 40)

# Define simple matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.identity(2)
observables_labels = [rf"$\sigma_{label}$" for label in ['x', 'y', 'z']]
observables_colors = ['blue', 'green', 'orange']

for l in L_values:
    # Generate the Hamiltonian
    H = periodic_dense_hamiltonian(l, h_x, h_z)
    # Diagonalize the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Compute thermal energies
    thermal_energies = {}
    for beta in beta_values:
        thermal_energies[beta] = compute_thermal_energy(beta, eigenvalues)

    plt.figure()
    plt.title(f"Thermal Energy vs. $\\beta$ for L={l}")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Thermal Energy")
    plt.plot(list(thermal_energies.keys()), list(thermal_energies.values()), label="Thermal Energy")
    plt.grid()

    # Generate the initial state
    single_sight = np.array([1, -np.sqrt(3)]) / 2
    initial_state = make_product_state(single_sight, l)
    
    # Calculate the overlap coefficients
    overlap_coefficients = np.dot(eigenvectors.conj().T, initial_state)

    # Compute the initial energy of the initial state
    matrix_aliment = initial_state.conj().T @ H @ initial_state
    normalization = initial_state.conj().T @ initial_state
    initial_energy = matrix_aliment / normalization     

    # Plot the horizontal dashed red line at the initial energy
    plt.axhline(y=initial_energy, color='r', linestyle='--', label=f"Initial Energy = {initial_energy:.4f}")


    # Find the intersection point with the initial energy by taking the argmin of the absolute difference
    beta_intersection = min(thermal_energies, key=lambda beta: abs(thermal_energies[beta] - initial_energy))

    # annotate the intersection point
    plt.annotate(f"Intersection at $\\beta = {beta_intersection:.2f}$", (beta_intersection, initial_energy), textcoords="offset points", xytext=(0, 10), ha='center')

    

    # Add legend and save the plot
    plt.legend()
    plt.savefig(f"hw3/docs/images/p4_1_2_thermal_energy_L{l}.png")

    # Extend observables to the full system size
    full_sigma_x = tensor_product([sigma_x] + [identity] * (l - 1))
    full_sigma_y = tensor_product([sigma_y] + [identity] * (l - 1))
    full_sigma_z = tensor_product([sigma_z] + [identity] * (l - 1))
    full_observables = [full_sigma_x, full_sigma_y, full_sigma_z]

     # Identify the value of each observable at the beta intersection if it exists
    observables_values = {}
    for observable, label in zip(full_observables, observables_labels):
        observables_values[label] = compute_thermal_observable(beta_intersection, eigenvalues, eigenvectors, observable)

    # Compute time-dependent values for each observable
    observables_values_time = {}
    t_values = np.linspace(0, 200, 40)
    for label, observable in zip(observables_labels, full_observables):
        expectations = []
        for t in t_values:
            expectation = compute_observable_expectation(t, observable, overlap_coefficients, eigenvalues, eigenvectors)
            expectations.append(expectation)
        observables_values_time[label] = expectations

    # Plotting section
    plt.figure()
    plt.title(f"Observable with Equilibrium Value Annotated for L={l}")
    plt.xlabel("Time")
    plt.ylabel("Expectation Value")

    # Plot each observable as a line plot
    for label, color in zip(observables_labels, observables_colors):
        plt.plot(t_values, observables_values_time[label], label=label, color=color)

    # Add a horizontal line for each observable
    for label, value, color in zip(observables_labels, observables_values.values(), observables_colors):
        plt.axhline(y=value, color=color, linestyle='--', label=rf"{label} at $\beta_{{eq}} = {beta_intersection:.2f}$")

    plt.legend()
    plt.grid()
    plt.savefig(f"hw3/docs/images/p4_1_2_observabl_annotation_L{l}.png")