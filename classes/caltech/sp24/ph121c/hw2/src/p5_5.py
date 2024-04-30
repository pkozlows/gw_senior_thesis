import numpy as np
from scipy.sparse.linalg import eigsh  # Sparse matrix eigensolver
from hw1 import sparse_hamiltonian  # Assuming this function generates your Hamiltonian
from numpy.linalg import norm

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
            mps_tensors.append(u.reshape((previous_k, -1)))
        # for the middle iterations
        elif i < L:
            # append the rank 3 tensor following the notation of the tensor network diagrams
            mps_tensors.append(u.reshape((previous_k, 2, current_k)))
        # for the last iteration
        else:
            mps_tensors.append(u.reshape((previous_k, -1)))
            break
        # prepare the state for the next iteration
        state = (s @ vt).reshape((2*current_k, -1))
        # update the previous bond dimension
        previous_k = current_k

    return mps_tensors





def reconstruct_state(mps_tensors):
    state = mps_tensors[0]  # Start with the first tensor, assumed to be a matrix.

    for j in range(1, len(mps_tensors)):
        next_tensor = mps_tensors[j]

        # this is for the first construction
        if len(state.shape) == 2:
            # print(f"before state shape: {state.shape}, tensor shape: {next_tensor.shape}")
            state = np.einsum('ij,jkl->ikl', state, next_tensor)
        # this is for the middle cases
        elif len(state.shape) == 3 and len(next_tensor.shape) == 3:
            state = state.reshape(-1, state.shape[2])
            # print(f"before state shape: {state.shape}, tensor shape: {next_tensor.shape}")
            state = np.einsum('ij,jkl->ikl', state, next_tensor)
        # now for the final case
        elif len(next_tensor.shape) == 2:
            state = state.reshape(-1, state.shape[2])
            next_tensor = next_tensor.reshape(state.shape[1], -1)
            # print(f"before state shape: {state.shape}, tensor shape: {next_tensor.shape}")
            state = np.einsum('ij,jk->ik', state, next_tensor)
            

    return state.reshape(-1)  # Flatten the final state to a 1D array







def calculate_overlap(original, reconstructed):
    return np.abs(np.vdot(original, reconstructed) / (norm(original) * norm(reconstructed)))

# Parameters
L = 16 # System size
h_values = [5/4, 1]  # Near and at the critical point
k_values = np.arange(1, 10, 1)  # Bond dimensions for MPS, from 1 to 20 inclusive


results = {}

for h in h_values:
    H = sparse_hamiltonian(L, h, periodic=False)
    eigenvalues, eigenvectors = eigsh(H.astype(np.float64), k=1, which='SA')
    gs = eigenvectors[:, 0]
    gs /= np.linalg.norm(gs)
    results[h] = {}

    for k in k_values:
        mps_tensors = compute_mps(gs, k)
        # print(mps_tensors)
        reconstructed_gs = reconstruct_state(mps_tensors)
        # print out the gs and reconstructed_gs
        # print(f"gs: {gs}")
        # print(f"reconstructed_gs: {reconstructed_gs}")
        overlap = calculate_overlap(gs, reconstructed_gs)
        num_params_original = gs.size
        num_params_mps = sum(t.size for t in mps_tensors)
        storage_reduction = num_params_original / num_params_mps
        results[h][k] = {
            'ground_state': gs,
            'mps': mps_tensors,
            'overlap': overlap,
            'storage_reduction': storage_reduction
        }

# Printing or processing results as needed
for h in results:
    for k in results[h]:
        pass
        # print(f"h={h}, k={k}, Overlap: {results[h][k]['overlap']:.3f}, Storage Reduction: {results[h][k]['storage_reduction']:.3f}")
