import numpy as np
from scipy.sparse.linalg import eigsh
# from hw1.src.hw1 import sparse_hamiltonian
from p5_5 import compute_mps
import scipy.sparse





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

def apply_operator(tensor, operator):
    return np.einsum('jk,ikl->ijl', operator, tensor)


# define the operators
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

# Define parameters
L = 6
k_values = np.arange(1, 20, 1)
h = 1.0

# Obtain the ground state from the Hamiltonian
H = sparse_hamiltonian(L, h, periodic=False)
eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
gs = eigenvectors[:, 0]

# Iterate over various bond dimensions k
for k in k_values:
    mps_tensors = compute_mps(gs, k)
    bra = [t.conj().T for t in mps_tensors]  # Prepare the bra state by conjugate transposing every tensor in the lest
    # make a function that takes in a list of mps tensors and its corresponding bra and computes the normalization
    def compute_contraction(mps_tensors, bra):
        # contract the physical energy_interactions on every tensor to generate a list of 2-tensors
        contraction = np.einsum('ijk,lji->kl', mps_tensors[0], bra[0])
        for j in range(1, len(mps_tensors)):
            if j == len(mps_tensors) - 1:
                contraction = np.einsum('kl,kmo,lmo->', contraction, mps_tensors[j], bra[j])
            else:
                contraction = np.einsum('kl,kmn,oml->no', contraction, mps_tensors[j], bra[j])
                
        return contraction
    # Compute energy components
    energy_interactions = 0
    energy_field = 0
    
    # Apply interaction
    for j in range(len(mps_tensors) - 1):
        mod_tensor1 = apply_operator(mps_tensors[j], sigma_z)
        mod_tensor2 = apply_operator(mps_tensors[j + 1], sigma_z)
        # make a new MPS lest that replaces the j-th and j+1-th tensors with the modified ones
        mps_tensors_mod = mps_tensors.copy()
        mps_tensors_mod[j] = mod_tensor1
        mps_tensors_mod[j + 1] = mod_tensor2
        # compute the contraction of the modified tensors
        energy_interactions -= compute_contraction(mps_tensors_mod, bra)

            
    # apply transverse field
    for j in range(len(mps_tensors)):
        # now we only have one tensor to consider for the field term
        mod_tensor = apply_operator(mps_tensors[j], sigma_x)
        # make a new MPS lest that replaces the j-th and j+1-th tensors with the modified ones
        mps_tensors_mod = mps_tensors.copy()
        mps_tensors_mod[j] = mod_tensor
        # compute the contraction of the modified tensors
        energy_field -= h*compute_contraction(mps_tensors_mod, bra)
        
    normalization = compute_contraction(mps_tensors, bra)
    total_energy = (energy_interactions + energy_field) / normalization
    print(f"Energy for k={k}: {total_energy}, Exact: {eigenvalues[0]}")