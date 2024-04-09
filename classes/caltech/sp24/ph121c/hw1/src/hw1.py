import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
# implement a function that generates the basis of possible states for a quantum icing model which is a tensor product of L spin-1/2 systems
def generate_basis(L):
    # the the spin up is a 0 and spin down 1
    # each basis state will get a binary number that goes up from 0 to 2^{L-1}
    basis = np.zeros(2**L)
    for i in range(2**L):
        basis[i] = i
    return basis

# create a function that will turn the number from 0 to 2^{L-1} into a binary number that is a string of 0s and 1s
def binary_string(n, L):
    binary = bin(n)[2:]
    while len(binary) < L:
        binary = '0' + binary
    return binary


sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.identity(2)

def tensor_product(matrices):
    """Calculate the tensor product of a list of matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

def open_dense_hamiltonian_explicit(L, h, J=1):
    # Initialize Hamiltonian to zero matrix
    H = np.zeros((2**L, 2**L))
    
    # Interaction term
    for i in range(L - 1):  # Add periodic term at the end if periodic
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_z  # Apply sigma_z at position i
        matrices[(i + 1)] = sigma_z  # Apply sigma_z at position (i+1) modulo L for periodic
        H += -J * tensor_product(matrices)
    
    # Transverse field term
    for i in range(L):
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_x  # Apply sigma_x at position i
        H += -h * tensor_product(matrices)
    
    return H

def periodic_dense_hamiltonian_explicit(L, h, J=1):
    # Initialize Hamiltonian to zero matrix
    H = np.zeros((2**L, 2**L))
    
    # Interaction term
    for i in range(L):  # Add periodic term at the end if periodic
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_z  # Apply sigma_z at position i
        matrices[(i + 1) % L] = sigma_z  # Apply sigma_z at position (i+1) modulo L for periodic
        H += -J * tensor_product(matrices)
    
    # Transverse field term
    for i in range(L):
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_x  # Apply sigma_x at position i
        H += -h * tensor_product(matrices)
    
    return H

def sparse_hamiltonian(L, h, J=1, periodic=False):
    row = []
    col = []
    data = []
    # loop over the index i
    for i in range(2**L):
        # turn that index into a binary string
        bi = binary_string(i, L)

        # to the diagonal element first
        total_interaction = 0
        # initialize the loop range based on the boundary conditions
        loop_range = L if periodic else L - 1
        for k in range(loop_range):
            # Handle periodic boundary by wrapping the index
            next_k = (k + 1) % L if periodic else k + 1
            # Add neighbor interaction based on the spin values
            total_interaction += -J * (1 if bi[k] == bi[next_k] else -1)
        row.append(i)
        col.append(i)
        data.append(total_interaction)

        # now consider the off-diagonal elements connected by a single spin flip
        # make a function that takes the bit i and returns a list of basis states connected by a single flip
        def flip(i):
            connected_bases_states = []
            for j in range(L):
                flip_i = i ^ (1 << j)
                connected_bases_states.append(flip_i)
            return connected_bases_states
        
        # loop over the connected basis states
        for flip_i in flip(i):
            row.append(i)
            col.append(flip_i)
            data.append(-h)       

    # Create a sparse matrix from the lists
    H = scipy.sparse.coo_matrix((data, (row, col)), shape=(2**L, 2**L))
    return H


