import numpy as np

def periodic_dense_hamiltonian_mbl(L, W, J=1):
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

    # transverse field term for x
    for i in range(L):
        # sample from the uniform distribution defined by (-W, W) to get an h_x value
        h_x = np.random.uniform(-W, W)
        matrices = [I] * L
        matrices[i] = sigma_x
        H += -h_x * tensor_product(matrices)

    # transverse field term for z
    for i in range(L):
        # sample from the uniform distribution defined by (-W, W) to get an h_z value
        h_z = np.random.uniform(-W, W)
        matrices = [I] * L
        matrices[i] = sigma_z
        H += -h_z * tensor_product(matrices)

    return H