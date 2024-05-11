import numpy as np  


def translation_operator(L):
    """Construct the translation operator T for a system of size L."""
    T = np.zeros((2 ** L, 2 ** L))

    for i in range(2 ** L):
        state = format(i, f'0{L}b')  # Binary representation of the state
        new_state = state[-1] + state[:-1]  # Shift by one site to the right
        new_index = int(new_state, 2)  # Convert back to decimal
        T[new_index, i] = 1

    return T

def identify_k0_sector(eigenvectors, T):
    """
    Identify eigenstates in the k=0 sector using the translation operator T.
    
    Parameters:
    - eigenvectors: Array of eigenvectors of the Hamiltonian.
    - T: Translation operator.
    
    Returns:
    - k0_indices: Indices of the k=0 sector eigenstates.
    """
    k0_indices = []
    for n, eigenvector in enumerate(eigenvectors.T):
        overlap = np.dot(eigenvector.conj().T, np.dot(T, eigenvector))
        if np.isclose(overlap, 1.0, atol=1e-8):
            k0_indices.append(n)
    return k0_indices