import numpy as np
import matplotlib.pyplot as plt
from hw1.src.hw1 import tensor_product, periodic_dense_hamiltonian_explicit

# now construct something like O(t) \equiv\langle\psi(t)|O| \psi(t)\rangle & =\sum_{m, n} c_{m}^{*} c_{n} e^{-i\left(\varepsilon_{n}-\varepsilon_{m}\right) t}\langle m|O| n\rangle  \tag{3}\\
def compute_observable_expectation_vectorized(t, observable, overlap_coefficients, eigenvalues, eigenvectors):
    # Precompute matrix elements
    observable_eigenvectors = np.dot(observable, eigenvectors)
    matrix_elements = np.dot(eigenvectors.conj().T, observable_eigenvectors)
    
    # Compute phase factors
    energy_diffs = eigenvalues[:, None] - eigenvalues[None, :]
    phase_factors = np.exp(-1j * energy_diffs * t)
    
    # Compute weights from overlap coefficients
    weight_matrix = overlap_coefficients[:, None] * overlap_coefficients.conj()[None, :]
    
    # Combine all to compute expectation
    return np.sum(weight_matrix * phase_factors * matrix_elements)


# System size and parameters
L_values = [8, 10, 12]
h_x = -1.05
h_z = 0.5
t_values = np.linspace(0, 50, 20)  # More time points for a smoother curve

# Define the observables for the first site
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.identity(2)

# Dictionary to hold plots for legends
observables_labels = ['sigma_x', 'sigma_y', 'sigma_z']

for L in L_values:
    # Extend observables to the full system size
    full_observables = {
        'sigma_x': tensor_product([sigma_x] + [identity] * (L - 1)),
        'sigma_y': tensor_product([sigma_y] + [identity] * (L - 1)),
        'sigma_z': tensor_product([sigma_z] + [identity] * (L - 1))
    }
    
    # Initial state: tensor product of single_site across all sites
    single_site = np.array([1, -np.sqrt(3)]) / 2
    initial_state = single_site.copy()  # Start with a copy of single_site

    for _ in range(1, L):
        initial_state = np.kron(initial_state, single_site)  # Properly extending to the full system size

    
    # Prepare to plot
    plt.figure(figsize=(10, 8))
    plt.title(f"System size L={L}")
    plt.xlabel("Time")
    plt.ylabel("Expectation value")

    # Generate the Hamiltonian
    H = periodic_dense_hamiltonian_explicit(L, h_x, h_z)

    # Diagonalize the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # calculate the overlap coefficients
    overlap_coefficients = np.dot(eigenvectors.conj().T, initial_state)

    for label, observable in full_observables.items():
        expectations = []
        for t in t_values:
            expectation = compute_observable_expectation_vectorized(t, observable, overlap_coefficients, eigenvalues, eigenvectors)
            expectations.append(np.real(expectation))  # Using real part; adjust if needed

        plt.plot(t_values, expectations, label=label)

    plt.legend()
    plt.grid(True)
    plt.savefig(f"Expectation_values_L_{L}.png")