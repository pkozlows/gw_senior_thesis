import numpy as np
from hw2.src.p5_5 import truncate_svd



def tensor_product(matrices):
        """Calculate the tensor product of a list of matrices."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result

def open_dense_hamiltonian(L, h_x=-1.05, h_z=0.5, J=1):
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
    for i in range(L - 1):  # Add periodic term at the end if periodic
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_z  # Apply sigma_z at position i
        matrices[i + 1] = sigma_z  # Apply sigma_z at position (i+1) modulo L for periodic
        H += -J * tensor_product(matrices)

    # Transverse field term for x
    for i in range(L):
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_x  # Apply sigma_x at position i
        H += -h_x * tensor_product(matrices)

    # Longitudinal field term for z
    for i in range(L):
        matrices = [I] * L  # Start with identity matrices
        matrices[i] = sigma_z  # Apply sigma_z at position i
        H += -h_z * tensor_product(matrices)

    return H

from scipy.linalg import expm
    
def create_trotter_gates(t, h_x=-1.05, h_z=0.5, J=1):
    """Create Trotter gates for the quantum Ising model."""
    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Single site Hamiltonian term
    omega = np.array([[-h_z, -h_x], [-h_x, h_z]])
    assert np.allclose(omega, omega.conj().T), "Omega is not Hermitian"
    
    # Interaction term
    interaction = -J * np.kron(sigma_z, sigma_z)
    assert np.allclose(interaction, interaction.conj().T), "Interaction is not Hermitian"
    
    # Create the Trotter gates
    gate_field = expm(1j * t * omega)
    # check whether this
    # assert np.allclose(gate_field.conj().T @ gate_field, np.eye(2))
    gate_odd = expm(1j * t * interaction).reshape(2, 2, 2, 2)
    # print(np.exp(1j * t * interaction))
    # assert np.allclose(gate_odd.conj().T @ gate_odd, np.eye(4))
    gate_even = gate_odd
    return gate_field, gate_odd, gate_even

def trotter_gate_field(mps, gate, site):
    """Apply a single Trotter gate to the MPS tensor at the given site."""
    mps_new = mps.copy()
    if site == 0:
        mps_new[site] = np.einsum('ik,ij->jk', mps[site], gate)
    elif site == len(mps) - 1:
        mps_new[site] = np.einsum('ij,ai->aj', gate, mps[site])
    else:
        mps_new[site] = np.einsum('ij,ajk->aik', gate, mps[site])
    return mps_new

def trotter_gate_interaction(mps, gate, site1, site2):
    """Apply a two-site Trotter gate to the MPS tensors at the given sites."""
    # make a copy of the mps tensors for modification
    mps_new = mps.copy()
    if site1 == 0:
        # Contract the first site with the gate
        w = np.einsum('ab,acdf,bfg->cdg', mps[site1], gate, mps[site2])
        w = w.reshape(gate.shape[1], gate.shape[2]*mps[site2].shape[2])
        # compute the SVD
        U, S, V = np.linalg.svd(w, full_matrices=False)
        # Update the MPS tensors
        mps_new[site1] = U.reshape(2, -1)
        mps_new[site2] = (np.diag(S) @ V).reshape(-1, 2, mps[site2].shape[2])
    elif site2 == len(mps) - 1:
        # Contract the last site with the gate
        w = np.einsum('abc,bdfg,cg->adf', mps[site1], gate, mps[site2])
        w = w.reshape(mps[site1].shape[0]*gate.shape[1], gate.shape[3])
        # compute the SVD
        U, S, V = np.linalg.svd(w, full_matrices=False)
        # Update the MPS tensors
        mps_new[site1] = U.reshape(mps[site1].shape[0], 2, -1)
        mps_new[site2] = (np.diag(S) @ V).reshape(-1, 2)
    else:
        w = np.einsum('abc,befd,cdg->aefg', mps[site1], gate, mps[site2])
        w = w.reshape(mps[site1].shape[0]*gate.shape[1], gate.shape[2]*mps[site2].shape[2])
        # compute the SVD
        U, S, V = np.linalg.svd(w, full_matrices=False)
        # Update the MPS tensors
        mps_new[site1] = U.reshape(mps[site1].shape[0], 2, -1)
        mps_new[site2] = (np.diag(S) @ V).reshape(-1, 2, mps[site2].shape[2])
    return mps_new

def apply_trotter_gates(mps, gate_field, gate_odd, gate_even):
    """Apply Trotter gates to the entire MPS."""
    L = len(mps)
    # Apply field gates
    # verify that the gate is unitary
    # assert np.allclose(gate_field.conj().T @ gate_field, np.eye(2))
    # Apply field gates
    for i in range(L):
        mps = trotter_gate_field(mps, gate_field, i)
    # Apply odd interaction gates
    for i in range(0, L-1, 2):
        mps = trotter_gate_interaction(mps, gate_odd, i, i+1)
    
    # Apply even interaction gates
    for i in range(1, L-1, 2):
        mps = trotter_gate_interaction(mps, gate_even, i, i+1)
    
    return mps

def check_left_canonical(tensor):
        left_canonical = False 
        if len(tensor.shape) == 2:
            tensor = tensor.reshape(1, *tensor.shape)
        if np.allclose(np.einsum('ijk,kjm->im' , tensor.conj().T, tensor), np.eye(tensor.shape[2])):
            left_canonical = True
        return left_canonical

def check_right_canonical(tensor):
    right_canonical = False
    if len(tensor.shape) == 2:
        tensor = tensor.reshape(*tensor.shape, 1)
    if np.allclose(np.einsum('ijk,kjm->im' , tensor, tensor.conj().T), np.eye(tensor.shape[0])):
        right_canonical = True
    return right_canonical
            
def enforce_bond_dimension(mps, chi):
    """Enforce left and right canonical forms on the MPS without truncating."""
    L = len(mps)
    
    # Print the maximum value before this operation
    max_value = np.max([np.max(np.abs(tensor)) for tensor in mps])
    print(f"Maximum value of the MPS tensors before any sweeps is {max_value}")
    
    # Left-to-right sweep
    for i in range(L-1):
        if i == 0:
            contraction = np.einsum('ij,jab->iab', mps[i], mps[i+1])
            w = contraction.reshape(mps[i].shape[0], mps[i+1].shape[1]*mps[i+1].shape[2])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            mps[i] = U.reshape(mps[i].shape[0], -1)
            assert check_left_canonical(mps[i])
            mps[i+1] = (np.diag(S) @ V).reshape(-1, mps[i+1].shape[1], mps[i+1].shape[2])
        elif i == L-2:
            contraction = np.einsum('ijk,kl->ijl', mps[i], mps[i+1])
            w = contraction.reshape(mps[i].shape[0]*mps[i].shape[1], mps[i+1].shape[1])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            s = S/np.sqrt(np.sum(np.diag(S) ** 2))
            print(f'After normalization, we have {s}')
            mps[i] = U.reshape(mps[i].shape[0], mps[i].shape[1], -1)
            assert check_left_canonical(mps[i])
            mps[i+1] = (np.diag(s) @ V).reshape(-1, mps[i+1].shape[1])
            assert not check_left_canonical(mps[i+1])
        else:
            contraction = np.einsum('ijk,klm->ijlm', mps[i], mps[i+1])
            w = contraction.reshape(mps[i].shape[0]*mps[i].shape[1], mps[i+1].shape[1]*mps[i+1].shape[2])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            mps[i] = U.reshape(mps[i].shape[0], mps[i].shape[1], -1)
            assert check_left_canonical(mps[i])
            mps[i+1] = (np.diag(S) @ V).reshape(-1, mps[i+1].shape[1], mps[i+1].shape[2])

    # Print the maximum value of the MPS tensors during enforcing bond dimension
    max_value = np.max([np.max(np.abs(tensor)) for tensor in mps])
    print(f"Maximum value of the MPS tensors after left to right sweep is {max_value}")
    
    # Right-to-left sweep
    for i in range(L-1, 0, -1):
        if i == L-1:
            contraction = np.einsum('ijk,kl->ijl', mps[i-1], mps[i])
            w = contraction.reshape(mps[i-1].shape[0]*mps[i-1].shape[1], mps[i].shape[1])
            u, s_diag, vt = truncate_svd(w, chi)
            mps[i-1] = (u @ s_diag).reshape(mps[i-1].shape[0], mps[i-1].shape[1], -1)
            mps[i] = vt.reshape(-1, mps[i].shape[1])
            assert check_right_canonical(mps[i])
        elif i == 1:
            contraction = np.einsum('ai,ijk->ajk', mps[i-1], mps[i])
            w = contraction.reshape(mps[i-1].shape[0], mps[i].shape[1]*mps[i].shape[2])
            u, s_diag, vt = truncate_svd(w, chi)
            mps[i-1] = (u @ s_diag).reshape(-1, mps[i].shape[0])
            mps[i] = vt.reshape(mps[i].shape[0], mps[i].shape[1], -1)
            assert check_right_canonical(mps[i])
        else:
            contraction = np.einsum('ijk,klm->ijlm', mps[i-1], mps[i])
            w = contraction.reshape(mps[i-1].shape[0]*mps[i-1].shape[1], mps[i].shape[1]*mps[i].shape[2])
            u, s_diag, vt = truncate_svd(w, chi)
            mps[i-1] = (u @ s_diag).reshape(mps[i-1].shape[0], mps[i-1].shape[1], -1)
            mps[i] = vt.reshape(-1, mps[i].shape[1], mps[i].shape[2])
            assert check_right_canonical(mps[i])
    
    # Print the maximum value after enforcing bond dimension
    max_value = np.max([np.max(np.abs(tensor)) for tensor in mps])
    print(f"Maximum value of the MPS tensors after enforcing bond dimension is {max_value}")
    
    return mps

def compute_contraction(mod_ket, bra):
    # contract the physical energy_interactions on every tensor to generate a list of 2-tensors
    # travel from site to site and compute the terpene contractions along the way; we will construct the virtual indices along a give list and contract the physical indices between the cat and bra
    L = len(mod_ket)
    # initialize the contraction to the last tensor
    contraction = np.einsum('ij,ik->jk', bra[0], mod_ket[0])

    last_contraction = contraction
    # loop through the rest of the tensors 
    for i in range(1, L-1):
        # contract the physical indices
        contraction = np.einsum('jk,jpq,kpt->qt', last_contraction, bra[i], mod_ket[i])
        last_contraction = contraction
        
    # we need to treat the last site differentially because it has no right tensor to contract with
    contraction = np.einsum('qt,qp,tp->', contraction, bra[L-1], mod_ket[L-1])
    return contraction

def apply_local_hamiltonian(mps_enforced, h_x=-1.05, h_z=0.5, J=1):
    '''Apply the local Hamiltonian to the MPS.'''
    bra = [t.conj() for t in mps_enforced]
    def apply_operator(tensor, operator, location):
        '''Apply an operator to a tensor at a given location.'''
        if location == 'left':
            return np.einsum('jk,jl->lk', tensor, operator)
        elif location == 'right':
            return np.einsum('jk,kl->jl', tensor, operator)
        elif location == 'middle':
            return np.einsum('ijk,jl->ilk', tensor, operator)
        
    
    # Define operators
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    # Compute energy components
    energy_interactions = 0
    energy_field = 0
    
    # Apply interaction
    for j in range(len(mps_enforced) - 1):
        if j == 0:
            first_location = 'left'
            second_location = 'middle'
        elif j == len(mps_enforced) - 2:
            first_location = 'middle'
            second_location = 'right'
        else:
            first_location = 'middle'
            second_location = 'middle'
        mod_tensor1 = apply_operator(mps_enforced[j], sigma_z, first_location)
        mod_tensor2 = apply_operator(mps_enforced[j + 1], sigma_z, second_location)
        # make a new MPS lest that replaces the j-th and j+1-th tensors with the modified ones
        mps_tensors_mod = mps_enforced.copy()
        mps_tensors_mod[j] = mod_tensor1
        mps_tensors_mod[j + 1] = mod_tensor2
        # compute the contraction of the modified tensors
        energy_interactions -= J*compute_contraction(mps_tensors_mod, bra)

            
    # apply transverse field in x direction
    for j in range(len(mps_enforced)):
        if j == 0:
            location = 'left'
        elif j == len(mps_enforced) - 1:
            location = 'right'
        else:
            location = 'middle'
        # now we only have one tensor to consider for the field term
        mod_tensor = apply_operator(mps_enforced[j], sigma_x, location)
        # make a new MPS lest that replaces the j-th and j+1-th tensors with the modified ones
        mps_tensors_mod = mps_enforced.copy()
        mps_tensors_mod[j] = mod_tensor
        # compute the contraction of the modified tensors
        energy_field -= h_x*compute_contraction(mps_tensors_mod, bra)

    # apply longitudinal field in z direction
    for j in range(len(mps_enforced)):
        if j == 0:
            location = 'left'
        elif j == len(mps_enforced) - 1:
            location = 'right'
        else:
            location = 'middle'
        # now we only have one tensor to consider for the field term
        mod_tensor = apply_operator(mps_enforced[j], sigma_z, location)
        # make a new MPS lest that replaces the j-th and j+1-th tensors with the modified ones
        mps_tensors_mod = mps_enforced.copy()
        mps_tensors_mod[j] = mod_tensor
        # compute the contraction of the modified tensors
        energy_field -= h_z*compute_contraction(mps_tensors_mod, bra)

    return energy_interactions + energy_field

    

        
    
    

        
    
    