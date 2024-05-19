import numpy as np
import matplotlib.pyplot as plt
from hw1.src.hw1 import tensor_product
from hw3.src.p4_1.fns import compute_observable_expectation, make_product_state, compute_thermal_energy, compute_thermal_observable
from hw3.src.p4_3.fns import periodic_dense_hamiltonian_mbl

# Set system parameters
L_values = [4, 6, 8]
W = 3
beta_values = np.linspace(-0.5, 0.5, 20)
number_realizations = 20

# Define simple matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.identity(2)
observables_labels = [rf"$\sigma_{label}$" for label in ['x', 'y', 'z']]
observables_colors = ['blue', 'green', 'orange']

# Define time values once
t_values = np.linspace(0, 10, 20)

# Loop over different system sizes
for L in L_values:
    # Initialize arrays to store thermal energies and overlaps
    thermal_energies = np.zeros((number_realizations, len(beta_values)))
    initial_energies = np.zeros(number_realizations)
    observables_values_time = {label: np.zeros((number_realizations, len(t_values))) for label in observables_labels}
    
    # Generate the initial state
    single_site = np.array([1, -np.sqrt(3)]) / 2
    initial_state = make_product_state(single_site, L)
    
    # Extend observables to the full system size once
    full_sigma_x = tensor_product([sigma_x] + [identity] * (L - 1))
    full_sigma_y = tensor_product([sigma_y] + [identity] * (L - 1))
    full_sigma_z = tensor_product([sigma_z] + [identity] * (L - 1))
    full_observables = [full_sigma_x, full_sigma_y, full_sigma_z]

    for realization in range(number_realizations):
        # Generate the Hamiltonian
        H = periodic_dense_hamiltonian_mbl(L, W)
        # Diagonalize the Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Compute thermal energies for each beta using vectorized operations
        thermal_energies[realization, :] = np.array([compute_thermal_energy(beta, eigenvalues) for beta in beta_values])
        
        # Compute the initial energy of the initial state
        matrix_element = initial_state.conj().T @ H @ initial_state
        normalization = initial_state.conj().T @ initial_state
        initial_energy = matrix_element / normalization
        initial_energies[realization] = initial_energy.real
        
        # Calculate the overlap coefficients
        overlap_coefficients = np.dot(eigenvectors.conj().T, initial_state)
        
        # Compute time-dependent values for each observable
        for label, observable in zip(observables_labels, full_observables):
            expectations = np.array([
                compute_observable_expectation(t, observable, overlap_coefficients, eigenvalues, eigenvectors).real
                for t in t_values
            ])
            observables_values_time[label][realization, :] = expectations

    # Average thermal energies and initial energies over realizations
    avg_thermal_energies = np.mean(thermal_energies, axis=0)
    avg_initial_energy = np.mean(initial_energies)
    
    # Average time-dependent observable values over realizations
    avg_observables_values_time = {label: np.mean(observables_values_time[label], axis=0) for label in observables_labels}
    
    # Plot the horizontal dashed red line at the initial energy
    plt.figure()
    plt.axhline(y=avg_initial_energy, color='red', linestyle='--', label=rf"Initial Energy at {avg_initial_energy:.2f}")

    # Plot the results
    plt.title(f"Thermal Energy vs $\\beta$ for L={L}")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Thermal Energy")
    plt.plot(beta_values, avg_thermal_energies)
    
    # Prepare data for interpolation
    sorted_indices = np.argsort(avg_thermal_energies)
    sorted_energies = avg_thermal_energies[sorted_indices]
    sorted_betas = beta_values[sorted_indices]

    # Interpolating to find intersection point
    beta_intersection = np.interp(avg_initial_energy, sorted_energies, sorted_betas)
    
    # Plot and annotate intersection point
    plt.plot(beta_intersection, avg_initial_energy, 'ro')
    plt.annotate(rf"$\beta = {beta_intersection:.2f}$", (beta_intersection, avg_initial_energy), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    plt.legend()
    plt.grid()
    plt.savefig(f"hw3/docs/images/p4_3_1_thermal_energy_intersection_L{L}.png")

    # Plotting section for observables
    plt.figure()
    plt.title(f"Observable with Equilibrium Value Annotated for L={L}")
    plt.xlabel("Time")
    plt.ylabel("Expectation Value")

    # Plot each observable as a line plot
    for label, color in zip(observables_labels, observables_colors):
        plt.plot(t_values, avg_observables_values_time[label], label=label, color=color)

    # Add a horizontal line for each observable
    for label, color in zip(observables_labels, observables_colors):
        thermal_observable = compute_thermal_observable(beta_intersection, eigenvalues, eigenvectors, full_observables[observables_labels.index(label)])
        plt.axhline(y=thermal_observable, color=color, linestyle='--', label=rf"{label} at $\beta_{{eq}} = {beta_intersection:.2f}$")

    plt.legend()
    plt.grid()
    plt.savefig(f"hw3/docs/images/p4_3_1_expectation_L{L}.png")
