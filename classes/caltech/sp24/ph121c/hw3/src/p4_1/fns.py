import numpy as np

#4.1.1

def compute_observable_expectation(t, observable, overlap_coefficients, eigenvalues, eigenvectors):
    """
    Compute the time-dependent expectation value of an observable.
    
    Parameters:
    - t (float): Time at which to compute the expectation.
    - observable (np.ndarray): The observable matrix.
    - overlap_coefficients (np.ndarray): Coefficients of the initial state in the energy eigenbasis.
    - eigenvalues (np.ndarray): Array of eigenvalues from the Hamiltonian diagonalization.
    - eigenvectors (np.ndarray): Array of eigenvectors from the Hamiltonian diagonalization.
    
    Returns:
    - expectation (complex): The expectation value of the observable at time t.
    """
    # Calculate matrix elements in the eigenbasis
    matrix_element = eigenvectors.conj().T @ observable @ eigenvectors

    # Calculate phase differences using broadcasting
    phase = np.exp(-1j * (eigenvalues[:, None] - eigenvalues[None, :]) * t)

    # Reshape overlap coefficients for broadcasting
    overlap_coefficients = overlap_coefficients[:, None]

    # Calculate expectation value
    expectation = np.sum(overlap_coefficients.conj() * overlap_coefficients.T * phase * matrix_element)

    return expectation
    

def periodic_dense_hamiltonian(L, h_x, h_z, J=1):
    # Initialize Hamiltonian to zero matrix
    H = np.zeros((2 ** L, 2 ** L))

    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    I = np.identity(2)

    # add the tensor product helper function
    def tensor_product(matrices):
        """Calculate the tensor product of a list of matrices."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result
    
    # Interaction term
    for i in range(L):  # Add periodic term at the end if periodic
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_z  # Apply sigma_z at position i
        matrices[(i + 1) % L] = sigma_z  # Apply sigma_z at position (i+1) modulo L for periodic
        H += -J * tensor_product(matrices)

    # Transverse field term for x
    for i in range(L):
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_x  # Apply sigma_x at position i
        H += -h_x * tensor_product(matrices)

    # Transverse field term for z
    for i in range(L):
        matrices = [I] * L
        matrices[i] = sigma_z
        H += -h_z * tensor_product(matrices)

    return H
        

#4.1.2

def compute_thermal_energy(beta, eigenvalues):
    """
    Compute the thermal energy of a system characterized by size L at inverse temperature beta.
    
    Parameters:
    - beta: Inverse temperature.
    - L: System size.
    
    Returns:
    - thermal_energy: The thermal energy of the system.
    """
    Z = sum(np.exp(-beta * eigenvalues))
    thermal_energy = np.sum(eigenvalues * np.exp(-beta * eigenvalues)) / Z
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
    boltzmann_factors = np.exp(-beta * eigenvalues)
    observable_in_basis = eigenvectors.conj().T @ observable @ eigenvectors
    weighted_trace = np.sum(boltzmann_factors * np.diag(observable_in_basis))
    Z = np.sum(boltzmann_factors)
    thermal_observable = weighted_trace / Z
    return thermal_observable

def make_product_state(single_site, L):
    """
    Generate the product state for a system of size L.
    
    Parameters:
    - single_site: The state of a single site.
    - L: The size of the system.
    
    Returns:
    - product_state: The product state of the system.
    """
    product_state = single_site.copy()
    for _ in range(1, L):
        product_state = np.kron(product_state, single_site)
    return product_state
    
#4.1.3
def time_dependent_state(t, overlap_coefficients, eigenvalues, eigenvectors):
    """Computes the time-dependent state |ψ(t)>."""
    return np.sum(np.exp(-1j * eigenvalues * t) * overlap_coefficients * eigenvectors, axis=1)
