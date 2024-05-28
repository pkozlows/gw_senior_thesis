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


    
def create_trotter_gates(t, h_x=-1.05, h_z=0.5, J=1):
    """Create Trotter gates for the quantum Ising model."""
    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Single site Hamiltonian term
    omega = np.array([[-h_z, -h_x], [-h_x, h_z]])
    
    # Interaction term
    interaction = -J * np.kron(sigma_z, sigma_z)
    
    # Create the Trotter gates
    gate_field = np.exp(-1j * t * omega)
    gate_odd = np.exp(-1j * t * interaction).reshape(2, 2, 2, 2)
    gate_even = np.exp(-1j * t * interaction).reshape(2, 2, 2, 2)
    
    return gate_field, gate_odd, gate_even

def trotter_gate_field(mps, gate, site):
    """Apply a single Trotter gate to the MPS tensor at the given site."""
    mps_new = mps.copy()
    if site == 0:
        mps_new[site] = np.einsum('ik,ij->jk', mps[site], gate)
    elif site == len(mps) - 1:
        mps_new[site] = np.einsum('ij,aj->ai', gate, mps[site])
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
    for i in range(L):
        mps = trotter_gate_field(mps, gate_field, i)
    print(f'After field gates for L={L}')
    for i in range(L):
        print(mps[i].shape)
    # Apply odd interaction gates
    for i in range(0, L-1, 2):
        mps = trotter_gate_interaction(mps, gate_odd, i, i+1)
    print(f'After odd interaction gates for L={L}')
    for i in range(L):
        print(mps[i].shape)
    
    # Apply even interaction gates
    for i in range(1, L-1, 2):
        mps = trotter_gate_interaction(mps, gate_even, i, i+1)
    print(f'After even interaction gates for L={L}')
    for i in range(L):
        print(mps[i].shape)
    
    return mps

def check_left_canonical(tensor):
        if len(tensor.shape) == 2:
            tensor = tensor.reshape(1, *tensor.shape)
        if np.allclose(np.einsum('ijk,kjm->im' , tensor.conj().T, tensor), np.eye(tensor.shape[2])):
            print(f"Tensor is left-canonical.")
        else:
            print(f"Tensor is NOT left-canonical.")

def check_right_canonical(mps_tensors):
    L = len(mps_tensors)
    # iterate through the list again
    for idx, tensor in enumerate(reversed(mps_tensors)):
        if idx == 0:
            assert len(tensor.shape) == 2
            tensor = tensor.reshape(*tensor.shape, 1)
        elif idx == len(mps_tensors) - 1:
            assert len(tensor.shape) == 2
            tensor = tensor.reshape(1, *tensor.shape)
        # Right canonical check
        if np.allclose(np.einsum('ijk,kjm->im' , tensor, tensor.conj().T), np.eye(tensor.shape[0])):
            print(f"Tensor at site {idx-L} is right-canonical.")
        else:
            print(f"Tensor at site {idx-L} is NOT right-canonical.")
            
def enforce_bond_dimension(mps, chi):
    """Enforce left canonical form on the MPS without truncating."""
    L = len(mps)
    mps_new = mps.copy()
    
    for i in range(L-1):
        # Contract the i and i+1 tensors to prepare for SVD
        if i == 0:
            contraction = np.einsum('ij,jab->iab', mps_new[i], mps_new[i+1])
            w = contraction.reshape(mps_new[i].shape[0], mps_new[i+1].shape[1]*mps_new[i+1].shape[2])
            # Compute the SVD
            U, S, V = np.linalg.svd(w, full_matrices=False)
            # Update the MPS tensors
            mps_new[i] = U.reshape(mps_new[i].shape[0], mps_new[i].shape[1])
            print(i)
            check_left_canonical(mps_new[i])
            mps_new[i+1] = (np.diag(S)@V).reshape(mps_new[i+1].shape[0], mps_new[i+1].shape[1], mps_new[i+1].shape[2])
        elif i == L-2:
            contraction = np.einsum('ijk,kl->ijl', mps_new[i], mps_new[i+1])
            0
            w = contraction.reshape(mps_new[i].shape[0]*mps_new[i].shape[1], mps_new[i+1].shape[1])
            # Compute the SVD
            U, S, V = np.linalg.svd(w, full_matrices=False)
            # Update the MPS tensors
            mps_new[i] = U.reshape(mps_new[i].shape[0], mps_new[i].shape[1], mps_new[i].shape[2])
            print(i)
            check_left_canonical(mps_new[i])
            mps_new[i+1] = (np.diag(S)@V).reshape(mps_new[i+1].shape[0], mps_new[i+1].shape[1])
            check_left_canonical(mps_new[i+1])
        else:
            contraction = np.einsum('ijk,klm->ijlm', mps_new[i], mps_new[i+1])
            w = contraction.reshape(mps_new[i].shape[0] * mps_new[i].shape[1], mps_new[i+1].shape[1] * mps_new[i+1].shape[2])
            # Compute the SVD
            U, S, V = np.linalg.svd(w, full_matrices=False)
            # Update the MPS tensors
            mps_new[i] = U.reshape(mps_new[i].shape[0], mps_new[i].shape[1], -1)
            print(i)
            check_left_canonical(mps_new[i])
            mps_new[i+1] = (np.diag(S)@V).reshape(-1, mps_new[i+1].shape[1], mps_new[i+1].shape[2]) 

    mps = mps_new.copy()
    # check_left_canonical(mps)
    # print the shapes of the mps tensors
    for i in range(L):
        print(mps[i].shape)
    mps_new = mps.copy()
    # now throw it in reverse while enforcing the bond dimension
    for i in range(L-1, 0, -1):
        if i == L-1:
            contraction = np.einsum('ijk,kl->ijl', mps_new[i-1], mps_new[i])
            w = contraction.reshape(mps_new[i-1].shape[0]*mps_new[i-1].shape[1], mps_new[i].shape[1])
            # Compute the SVD
            u, s_diag, vt = truncate_svd(w, chi)
            # Update the MPS tensors
            mps_new[i-1] = (u @ s_diag).reshape(mps_new[i-1].shape[0], mps_new[i-1].shape[1], mps_new[i-1].shape[2])
            mps_new[i] = vt.reshape(mps_new[i].shape[0], mps_new[i].shape[1])
        elif i == 1:
            contraction = np.einsum('ai,ijk->ajk', mps_new[i-1], mps_new[i])
            w = contraction.reshape(mps_new[i-1].shape[0], mps_new[i].shape[1]*mps_new[i].shape[2])
            # Compute the SVD
            u, s_diag, vt = truncate_svd(w, chi)
            # Update the MPS tensors
            mps_new[i-1] = (u @ s_diag).reshape(-1, mps_new[i].shape[0])
            mps_new[i] = vt.reshape(mps_new[i].shape[0], mps_new[i].shape[1], -1)
        else:
            contraction = np.einsum('ijk,klm->ijlm', mps_new[i-1], mps_new[i])
            w = contraction.reshape(mps_new[i-1].shape[0]*mps_new[i-1].shape[1], mps_new[i].shape[1]*mps_new[i].shape[2])
            # Compute the SVD
            u, s_diag, vt = truncate_svd(w, chi)
            # Update the MPS tensors
            mps_new[i-1] = (u @ s_diag).reshape(mps_new[i-1].shape[0], mps_new[i-1].shape[1], -1)
            mps_new[i] = vt.reshape(-1, mps_new[i].shape[1], mps_new[i].shape[2])
    mps = mps_new.copy()

    check_right_canonical(mps)
    return mps

def compute_contraction(mod_ket, bra):
    # contract the physical energy_interactions on every tensor to generate a list of 2-tensors
    # travel from site to site and compute the terpene contractions along the way; we will construct the virtual indices along a give list and contract the physical indices between the cat and bra
    L = len(mod_ket)
    # initialize the contraction to the last tensor
    contraction = np.einsum('ij,ki->jk', bra[0], mod_ket[0])
    # loop through the rest of the tensors 
    for i in range(1, L-1):
        # contract the physical indices
        contraction = np.einsum('jk,jpq,kpt->qt', contraction, bra[i], mod_ket[i])
    # we need to treat the last site differentially because it has no right tensor to contract with
    contraction = np.einsum('qt,qp,tp->', contraction, bra[L-1], mod_ket[L-1])
    return contraction
    

        
    
    

        
    
    