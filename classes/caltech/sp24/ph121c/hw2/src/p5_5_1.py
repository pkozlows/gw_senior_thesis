import numpy as np
from scipy.sparse.linalg import eigsh
from hw1 import sparse_hamiltonian
from p5_5 import compute_mps


def apply_operator(tensor, operator, position):
    """
    Apply an operator to the tensor based on its shape.

    Parameters:
        tensor (ndarray): The input tensor.
        operator (ndarray): The operator to be applied.
        position (str): The position of the tensor in the operation. 
            Possible values are 'first', 'middle', or 'last'.

    Returns:
        ndarray: The result of applying the operator to the tensor.

    Raises:
        ValueError: If the position is not one of 'first', 'middle', or 'last'.
    """
    if position == 'first': # First tensor (two indices)
        return np.einsum('ij,jk->jk', operator, tensor)
    elif position == 'middle': # Middle tensor (three indices)
        return np.einsum('jl,ijk->ilk', operator, tensor)
    elif position == 'last': # Last tensor (two indices)
        return np.einsum('ij,aj->ai', operator, tensor)


# define the operators
sigma_z = np.array([[1, 0], [0, -1]])
sigma_x = np.array([[0, 1], [1, 0]])
identity = np.eye(2)

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

    # Compute energy components
    energy_interactions = 0
    energy_field = 0

    # Apply interaction
    for j in range(len(mps_tensors) - 1):
        # make a condition if the j tensor is the first tensor
        if j == 0:
            mod_tensor1 = apply_operator(mps_tensors[j], sigma_z, 'first')
            mod_tensor2 = apply_operator(mps_tensors[j+1], sigma_z, 'middle')
            energy_interactions -= np.einsum('ij,ji->', mod_tensor1, bra[j]) * np.einsum('ijk,kji->', mod_tensor2, bra[j+1])
        # make a condition if the j tensor is in the middle
        elif j < len(mps_tensors) - 2:
            mod_tensor1 = apply_operator(mps_tensors[j], sigma_z, 'middle')
            mod_tensor2 = apply_operator(mps_tensors[j+1], sigma_z, 'middle')
            energy_interactions -= np.einsum('ijk,kji->', mod_tensor1, bra[j]) * np.einsum('ijk,kji->', mod_tensor2, bra[j+1])
        # make a condition if the j tensor is the last tensor
        else:
            mod_tensor1 = apply_operator(mps_tensors[j], sigma_z, 'middle')
            mod_tensor2 = apply_operator(mps_tensors[j+1], sigma_z, 'last')
            energy_interactions -= np.einsum('ijk,kji->', mod_tensor1, bra[j]) * np.einsum('ij,ji->', mod_tensor2, bra[j+1])
            
    # apply transverse field
    for j in range(len(mps_tensors)):
        # make a condition if the j tensor is the first tensor
        if j == 0:
            mod_tensor = apply_operator(mps_tensors[j], sigma_x, 'first')
            energy_field -= h * np.einsum('ij,ji->', mod_tensor, bra[j])
        # make a condition if the j tensor is in the middle
        elif j < len(mps_tensors) - 1:
            mod_tensor = apply_operator(mps_tensors[j], sigma_x, 'middle')
            energy_field -= h * np.einsum('ijk,kji->', mod_tensor, bra[j])
        # make a condition if the j tensor is the last tensor
        else:
            mod_tensor = apply_operator(mps_tensors[j], sigma_x, 'last')
            energy_field -= h * np.einsum('ij,ji->', mod_tensor, bra[j])
        
    # compute the normalization
    normalization = 0
    for j in range(len(mps_tensors)):
        # make a condition if the j tensor is the first tensor
        if j == 0:
            mod_tensor = apply_operator(mps_tensors[j], identity, 'first')
            normalization += np.einsum('ij,ji->', mod_tensor, bra[j])
        # make a condition if the j tensor is in the middle
        elif j < len(mps_tensors) - 1:
            mod_tensor = apply_operator(mps_tensors[j], identity, 'middle')
            normalization += np.einsum('ijk,kji->', mod_tensor, bra[j])
        # make a condition if the j tensor is the last tensor
        else:
            mod_tensor = apply_operator(mps_tensors[j], identity, 'last')
            normalization += np.einsum('ij,ji->', mod_tensor, bra[j])
        
    total_energy = (energy_interactions + energy_field) / normalization
    print(f"Energy for k={k}: {total_energy}, Exact: {eigenvalues[0]}")