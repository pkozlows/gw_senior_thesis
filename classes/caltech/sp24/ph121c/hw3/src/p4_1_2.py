import numpy as np
import matplotlib.pyplot as plt
from hw1.src.hw1 import tensor_product, periodic_dense_hamiltonian_explicit



def compute_thermal_energy(beta, eigenvalues):
    """
    Compute the thermal energy of a system characterized by size L at inverse temperature beta.
    
    Parameters:
    - beta: Inverse temperature.
    - L: System size.
    
    Returns:
    - thermal_energy: The thermal energy of the system.
    """
    # Compute the partition function
    Z = sum(np.exp(-beta * eigenvalues))
    
    # Initialize thermal energy
    thermal_energy = 0
    
    # Iterate over all energy levels
    for i in range(len(eigenvalues)):
        thermal_energy += eigenvalues[i] * np.exp(-beta * eigenvalues[i])
    # divided by the partition function
    thermal_energy /= Z
    return thermal_energy


def compute_thermal_observable(beta, eigenvalues, eigenvectors, observable):
    """
    Compute the thermal expectation value of an observable.
    
    Parameters:
    - beta: Inverse temperature.
    - eigenvalues: Eigenvalues from the Hamiltonian diagonalization.
    - eigenvectors: Eigenvectors from the Hamiltonian diagonalization.
    - observable: The observable matrix.
    
    Returns:
    - thermal_observable: The thermal expectation value of the observable.
    """
    # Compute the Boltzmann factors for each eigenstate
    boltzmann_factors = np.exp(-beta * eigenvalues)
    
    # Compute the observable in the eigenbasis
    observable_in_basis = eigenvectors.conj().T @ observable @ eigenvectors
    
    # Compute the weighted trace of the observable
    weighted_trace = np.sum(boltzmann_factors * np.diag(observable_in_basis))
    
    # Compute the partition function Z
    Z = np.sum(boltzmann_factors)
    
    # Calculate the thermal observable
    thermal_observable = weighted_trace / Z
    
    return thermal_observable


# Define system parameters
L_values = [4, 6, 8, 10]
h_x = -1.05
h_z = 0.5
beta_values = np.linspace(0.5, 5, 10)  # More beta values for a smoother curve

# to fined the simple matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.identity(2)
observables_labels = ['sigma_x', 'sigma_y', 'sigma_z']


for l in L_values:
    # Generate the Hamiltonian
    H = periodic_dense_hamiltonian_explicit(l, h_x, h_z)
    # Diagonalize the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # loop over the theologies of beta
    thermal_energies = {}
    for beta in beta_values:
        thermal_energies[beta] = compute_thermal_energy(beta, eigenvalues)
    plt.figure()
    plt.title(f"Thermal energy vs. beta for L={l}")
    plt.xlabel("Beta")
    plt.ylabel("Thermal energy")
    plt.plot(list(thermal_energies.keys()), list(thermal_energies.values()))
    plt.grid()
    plt.savefig(f"p4_1_2_L{l}.png")


    # Extend observables to the full system size
    full_sigma_x = tensor_product([sigma_x] + [identity] * (l - 1))
    full_sigma_y = tensor_product([sigma_y] + [identity] * (l - 1))
    full_sigma_z = tensor_product([sigma_z] + [identity] * (l - 1))
    full_observables = [full_sigma_x, full_sigma_y, full_sigma_z]
    
    plt.figure()
    plt.title(f"Thermal Observables vs. beta for L={l}")
    plt.xlabel("Beta")
    plt.ylabel("Observable Value")

    # first loop over the observables
    for index, observable in enumerate(full_observables):
        # then loop over the talus of beta
        thermal_observable_values = []
        for beta in beta_values:
            thermal_observable_values.append(compute_thermal_observable(beta, eigenvalues, eigenvectors, observable))
        plt.plot(beta_values, thermal_observable_values, label=observables_labels[index])
    plt.legend()
    plt.grid(True)
    plt.savefig(f"p4_1_2_Observables_L{l}.png")



