import numpy as np
from scipy.sparse.linalg import eigsh
from hw1 import sparse_hamiltonian
from p5_5 import compute_mps


def apply_operator(tensor, operator):
    return np.einsum('jk,ikl->ijl', operator, tensor)


# define the operators
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])

# Define parameters
L = 6
k_values = np.arange(3, 20, 1)
h = 1.0

# Obtain the ground state from the Hamiltonian
H = sparse_hamiltonian(L, h, periodic=False)
eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
gs = eigenvectors[:, 0]

# Iterate over various bond dimensions k
for k in k_values:
    mps_tensors = compute_mps(gs, k)
    bra = [t.conj().T for t in mps_tensors]  # Prepare the bra state by conjugate transposing every tensor in the lest

    # Compute energy components
    energy_interactions = 0
    energy_field = 0

    # Apply interaction
    for j in range(len(mps_tensors) - 1):
        # we have two possible tensors to consider for the interaction term
        mod_tensor1 = apply_operator(mps_tensors[j], sigma_z)
        mod_tensor2 = apply_operator(mps_tensors[j + 1], sigma_z)
        energy_interactions -= np.einsum('ijk,kji->', mod_tensor1, bra[j]) * np.einsum('ijk,kji->', mod_tensor2, bra[j + 1])
            
    # apply transverse field
    for j in range(len(mps_tensors)):
        # now we only have one tensor to consider for the field term
        mod_tensor = apply_operator(mps_tensors[j], sigma_x)
        energy_field -= h * np.einsum('ijk,kji->', mod_tensor, bra[j])
        
    # compute the normalization
    normalization = 0
    for j in range(len(mps_tensors)):
        # we only have to consider one tensor again for the normalization
        normalization += np.einsum('ijk,kji->', mps_tensors[j], bra[j])
        
    total_energy = (energy_interactions + energy_field) / normalization
    print(f"Energy for k={k}: {total_energy}, Exact: {eigenvalues[0]}")