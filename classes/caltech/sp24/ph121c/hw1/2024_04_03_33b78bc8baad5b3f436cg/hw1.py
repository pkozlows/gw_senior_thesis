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
        # loop over the index j
        for j in range(i, 2**L):
            # turn that index into a binary string
            bj = binary_string(j, L)
            # if i == j, we are on the diagonal and we want to add all the interactions in the chain
            if i == j:
                total_interaction = 0
                # initialize the loop range based on the boundary conditions
                loop_range = L if periodic else L - 1
                # loop over the spins in the chain i.e. loop_range
                for k in range(loop_range):
                    # Handle periodic boundary by wrapping the index
                    next_k = (k + 1) % L if periodic else k + 1
                    # Add neighbor interaction based on the spin values
                    total_interaction += -J * (1 if bi[k] == bi[next_k] else -1)
                # tabulate the entries
                if total_interaction != 0:
                    row.append(i)
                    col.append(j)
                    data.append(total_interaction)
            else:  # Off-diagonal elements, only for transverse field flips
                diff = [bj != bi for bj, bi in zip(bj, bi)]
                # Only one bit/spin differs
                if sum(diff) == 1:  
                    transverse_contribution = -h
                    # Transverse field contribution
                    row.append(i)
                    col.append(j)
                    data.append(transverse_contribution)

    # Create a sparse matrix from the lists
    H = scipy.sparse.coo_matrix((data, (row, col)), shape=(2**L, 2**L))
    return H


L = [8, 10, 12, 14]
h_range = np.linspace(0, 2, 7)  # Generates 8 values of h from 0 to 2

for l in L:
    plt.figure()
    plt.title(f'L={l}')
    ground_state_energies = []  # To store ground state energies for different h
    for hi in h_range:
        # for the given hi, comupte gs energy for both periodic and non-periodic
        H = sparse_hamiltonian(l, hi)
        H_periodic = sparse_hamiltonian(l, hi, periodic=True)
        # Compute the ground state energy using the lowest eigenvalue
        ground_state_energies.append(min(scipy.sparse.linalg.eigsh(H, k=1, which='SA')[0]))
        ground_state_energies.append(min(scipy.sparse.linalg.eigsh(H_periodic, k=1, which='SA')[0]))
    # plot the ground state energy as a function of h for both periodic and non-periodic
    plt.plot(h_range, ground_state_energies[:len(h_range)], label='Non-periodic')
    plt.plot(h_range, ground_state_energies[len(h_range):], label='Periodic')
    plt.xlabel('h')
    plt.ylabel('Ground state energy')
    plt.legend()
    # save a tout for the given value of L
    plt.savefig(f'sparse_ising_model_L{l}.png')

for l in L:
    plt.figure()
    plt.title(f'L={l}')
    open_ground_state_energies = []  # To store ground state energies for different h
    periodic_ground_state_energies = []  # To store ground state energies for different h
    for hi in h_range:
        # for the given hi, comupte gs energy for both periodic and non-periodic
        H = open_dense_hamiltonian_explicit(l, hi)
        print(f'H: hi={hi}\n{H}')
        H_periodic = periodic_dense_hamiltonian_explicit(l, hi)
        # Compute the ground state energy using the lowest eigenvalue
        open_ground_state_energies.append(min(np.linalg.eigvalsh(H)))
        periodic_ground_state_energies.append(min(np.linalg.eigvalsh(H_periodic)))
    # plot the ground state energy as a function of h for both periodic and non-periodic
    plt.plot(h_range, open_ground_state_energies, label='Non-periodic')
    plt.plot(h_range, periodic_ground_state_energies, label='Periodic')
    plt.xlabel('h')
    plt.ylabel('Ground state energy')
    plt.legend()
    # save a tout for the given value of L
    plt.savefig(f'dense_ising_model_L{l}.png')

def dense_hamiltonian(L, h, J=1, periodic=False):
    H = np.zeros((2**L, 2**L))

    for i in range(2**L):
        bi = binary_string(i, L)
        for j in range(2**L):
            bj = binary_string(j, L)
            if i == j:  # Diagonal elements
                # Adjust the loop range based on boundary conditions
                loop_range = L if periodic else L - 1
                for k in range(loop_range):
                    # For periodic boundary conditions, wrap the index
                    next_k = (k + 1) % L if periodic else k + 1
                    # Add neighbor interaction
                    H[i, j] -= J * (1 if bi[k] == bi[next_k] else -1)
            else:  # Off-diagonal elements, only for transverse field flips
                diff = [bj != bi for bj, bi in zip(bj, bi)]
                if sum(diff) == 1:  # Only one bit differs
                    # Transverse field contribution
                    H[i, j] -= h
    return H
