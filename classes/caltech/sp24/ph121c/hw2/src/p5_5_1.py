import numpy as np
from scipy.sparse.linalg import eigsh
from hw1 import sparse_hamiltonian
from p5_5 import compute_mps

def apply_interaction(tensor):
    """Apply the sigma_z interaction to the tensor."""
    sigma_z = np.array([[1, 0], [0, -1]])
    # check if we are dealing with the first tensor
    if len(tensor.shape) == 2 and tensor.shape[0] == 2:
        return np.einsum('ij,ik->jk', sigma_z, tensor)
    # check if we are dealing with a tensor in the middle
    elif len(tensor.shape) == 3:
        return np.einsum('ij,hjk->hik', sigma_z, tensor)
    # check if we are dealing with the last tensor
    elif len(tensor.shape) == 2 and tensor.shape[1] == 2:
        return np.einsum('ik,jk->ik', sigma_z, tensor)

def apply_transverse_field(tensor):
    """Apply the sigma_x field to the tensor."""
    sigma_x = np.array([[0, 1], [1, 0]])
    # check if we are dealing with the first tensor
    if len(tensor.shape) == 2 and tensor.shape[0] == 2:
        return np.einsum('ij,ik->jk', sigma_x, tensor)
    # check if we are dealing with a tensor in the middle
    elif len(tensor.shape) == 3:
        return np.einsum('ij,hjk->hik', sigma_x, tensor)
    # check if we are dealing with the last tensor
    elif len(tensor.shape) == 2 and tensor.shape[1] == 2:
        return np.einsum('ik,jk->ij', sigma_x, tensor)

def apply_identity(tensor):
    """Apply the identity operator to the tensor."""
    identity = np.eye(2)
    # check if we are dealing with the first tensor
    if len(tensor.shape) == 2 and tensor.shape[0] == 2:
        return np.einsum('ij,ik->jk', identity, tensor)
    # check if we are dealing with a tensor in the middle
    elif len(tensor.shape) == 3:
        return np.einsum('ij,hjk->hik', identity, tensor)
    # check if we are dealing with the last tensor
    elif len(tensor.shape) == 2 and tensor.shape[1] == 2:
        return np.einsum('ik,jk->ij', identity, tensor)
# Define parameters
L = 6
k_values = np.arange(1, 10, 1)
h = 1.0

# Obtain the ground state from the Hamiltonian
H = sparse_hamiltonian(L, h, periodic=False)
eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
gs = eigenvectors[:, 0]
gs /= np.linalg.norm(gs)  # Normalize the Finished and PSground state

# Iterate over various bond dimensions k
for k in k_values:
    mps_tensors = compute_mps(gs, k)
    bra = [t.conj().T for t in reversed(mps_tensors)]  # Prepare the bra state by conjugate transposing every tensor in the lest in reverse order

    # Compute energy components
    energy_interactions = 0
    energy_field = 0

    # Apply interaction
    for j in range(len(mps_tensors) - 1):
        # make a condition if the j tensor is the first tensor
        if j == 0:
            mod_tensor1 = apply_interaction(mps_tensors[j])
            mod_tensor2 = apply_interaction(mps_tensors[j+1])
            energy_interactions -= np.einsum('ij,ij->', mod_tensor1, bra[j]) * np.einsum('ijk,ijk->', mod_tensor2, bra[j+1])
        # make a condition if the j tensor is in the middle
        elif j < len(mps_tensors) - 2:
            mod_tensor1 = apply_interaction(mps_tensors[j])
            mod_tensor2 = apply_interaction(mps_tensors[j+1])
            energy_interactions -= np.einsum('ijk,ijk->', mod_tensor1, bra[j]) * np.einsum('ijk,ijk->', mod_tensor2, bra[j+1])
        # make a condition if the j tensor is the last tensor
        else:
            mod_tensor1 = apply_interaction(mps_tensors[j])
            mod_tensor2 = apply_interaction(mps_tensors[j+1])
            energy_interactions -= np.einsum('ijk,ijk->', mod_tensor1, bra[j]) * np.einsum('ij,ij->', mod_tensor2, bra[j+1])
    # apply transverse field
    for j in range(len(mps_tensors)):
        # make a condition if the j tensor is the first tensor
        if j == 0:
            mod_tensor = apply_transverse_field(mps_tensors[j])
            energy_field -= h * np.einsum('ij,ij->', mod_tensor, bra[j])
        # make a condition if the j tensor is in the middle
        elif j < len(mps_tensors) - 1:
            mod_tensor = apply_transverse_field(mps_tensors[j])
            energy_field -= h * np.einsum('ijk,ijk->', mod_tensor, bra[j])
        # make a condition if the j tensor is the last tensor
        else:
            mod_tensor = apply_transverse_field(mps_tensors[j])
            energy_field -= h * np.einsum('ij,ij->', mod_tensor, bra[j])
        
    # compute the normalization
    normalization = 0
    for j in range(len(mps_tensors)):
        # make a condition if the j tensor is the first tensor
        if j == 0:
            mod_tensor = apply_identity(mps_tensors[j])
            normalization += np.einsum('ij,ij->', mod_tensor, bra[j])
        # make a condition if the j tensor is in the middle
        elif j < len(mps_tensors) - 1:
            mod_tensor = apply_identity(mps_tensors[j])
            normalization += np.einsum('ijk,ijk->', mod_tensor, bra[j])
        # make a condition if the j tensor is the last tensor
        else:
            mod_tensor = apply_identity(mps_tensors[j])
            normalization += np.einsum('ij,ij->', mod_tensor, bra[j])
        
    total_energy = (energy_interactions + energy_field) / normalization
    print(f"Energy for k={k}: {total_energy}, Exact: {eigenvalues[0]}")
