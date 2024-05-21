import numpy as np
from scipy.sparse.linalg import eigsh
# from hw1.src.hw1 import sparse_hamiltonian
import scipy.sparse
import matplotlib.pyplot as plt

def truncate_svd(matrix, k):
    """
    Truncates the singular value decomposition (SVD) of a matrix.

    Parameters:
    matrix (ndarray): The input matrix.
    k (int): The number of singular values to keep.

    Returns:
    ndarray: The truncated left singular vectors.
    ndarray: The truncated singular values.
    ndarray: The truncated right singular vectors.
    """
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :k], np.diag(s[:k]), vt[:k, :]


def compute_mps(state, k):
    # this is just the length of the system
    L = int(np.log2(state.size))
    # prepare the state for the initial SVD
    state = state.reshape((2, -1))
    # list to store the mps tensors
    mps_tensors = []
    # initialize a previous bond dimension
    previous_k = 2

    # loop over the system
    for i in range(1, L+1):
        # make the SVD
        u, s, vt = truncate_svd(state, k)
        # current bond dimension
        current_k = min(k, u.shape[1])
        # debugging
        # print(f'Current state shape: {state.shape}, u shape: {u.shape}, s shape: {s.shape}, vt shape: {vt.shape}')

        # for the first iteration
        if i == 1:
            mps_tensors.append(u.reshape((1, previous_k, -1)))
        # for the middle iterations
        elif i < L:
            # append the rank 3 tensor following the notation of the tensor network diagrams
            mps_tensors.append(u.reshape((previous_k, 2, current_k)))
        # for the last iteration
        else:
            mps_tensors.append(u.reshape((previous_k, -1, 1)))
            break
        # prepare the state for the next iteration
        state = (s @ vt).reshape((2*current_k, -1))
        # update the previous bond dimension
        previous_k = current_k

    return mps_tensors


def binary_string(n, L):
    binary = bin(n)[2:]
    while len(binary) < L:
        binary = '0' + binary
    return binary

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

def mps_correlations(mps_tensors):
    def apply_operator(tensor, operator):
        return np.einsum('jk,ikl->ijl', operator, tensor)

    # define the operators
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    # Calculate the correlation function of a state.
    contractions = []
    L = len(mps_tensors)
    bra = [t.conj().T for t in mps_tensors]
    mod_tensor1 = apply_operator(mps_tensors[0], sigma_z)
    for r in range(1, L):
        # just modify the tensor at site r
        mod_tensor_r = apply_operator(mps_tensors[r], sigma_z)
        mps_tensors_mod = mps_tensors.copy()
        mps_tensors_mod[0] = mod_tensor1
        mps_tensors_mod[r] = mod_tensor_r
        # compute the contraction
        contraction = compute_contraction(mps_tensors_mod, bra)
        contractions.append(contraction)
    return contractions
def compute_contraction(mps_tensors, bra):
    # contract the physical energy_interactions on every tensor to generate a list of 2-tensors
    contraction = np.einsum('ijk,ijl->kl', mps_tensors[0], bra[0])
    for j in range(1, len(mps_tensors)):
        if j == len(mps_tensors) - 1:
            contraction = np.einsum('kl,kmo,lmo->', contraction, mps_tensors[j], bra[j])
        else:
            contraction = np.einsum('kl,kmn,lmo->no', contraction, mps_tensors[j], bra[j])
            
    return contraction
# # Define parameters
# L = 8
# k_values = np.arange(1, 5, 1)
# h_values = [0.3, 1.7]

# for k in k_values:
#     plt.figure(figsize=(10, 6))
#     plt.title(f'Correlation function as a function of site separation for k={k}')
#     plt.xlabel('Site separation r')
#     plt.ylabel('Correlation function')
#     for h in h_values:
#         # Obtain the ground state from the Hamiltonian
#         H = sparse_hamiltonian(L, h, periodic=False)
#         eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
#         gs = eigenvectors[:, 0]
#         mps_tensors = compute_mps(gs, k)
#         correlations = mps_correlations(mps_tensors)
#         r_values = range(1, L)
#         plt.plot(r_values, correlations, 'o-', label=f'h={h}')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'5-5_correlation_k{k}.png')
