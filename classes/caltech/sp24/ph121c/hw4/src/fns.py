import numpy as np
from hw2.src.p5_5_2_2 import check_left_canonical, check_right_canonical
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
    interaction = J * np.kron(sigma_z, sigma_z)
    
    # Create the Trotter gates
    gate_field = np.exp(-1j * t * omega)
    gate_odd = np.exp(-1j * t * interaction).reshape(2, 2, 2, 2)
    gate_even = np.exp(-1j * t * interaction).reshape(2, 2, 2, 2)
    
    return gate_field, gate_odd, gate_even

def trotter_gate_field(mps, gate, site):
    """Apply a single Trotter gate to the MPS tensor at the given site."""
    mps_new = mps.copy()
    mps_new[site] = np.einsum('ij,ajl->ail', gate, mps[site])
    return mps_new

def trotter_gate_interaction(mps, gate, site1, site2):
    """Apply a two-site Trotter gate to the MPS tensors at the given sites."""
    w = np.einsum('abc,bdef,cdg->aefg', mps[site1], gate, mps[site2])
    w = w.reshape(w.shape[0]*w.shape[1], w.shape[2]*w.shape[3])
    # compute the SVD
    U, S, V = np.linalg.svd(w)
    # make a copy of the mps tensors for modification
    mps_new = mps.copy()
    # Update the MPS tensors
    mps_new[site1] = U.reshape(mps[site1].shape[0], 2, -1)
    mps_new[site2] = np.dot(np.diag(S), V.conj().T).reshape(-1, 2, mps[site2].shape[2])
    return mps_new

def apply_trotter_gates(mps, gate_field, gate_odd, gate_even):
    """Apply Trotter gates to the entire MPS."""
    L = len(mps)
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

def enforce_bond_dimension(mps, chi):
    """Enforce left canonical form on the MPS without truncating."""
    L = len(mps)
    mps_new = mps.copy()
    
    for i in range(L-1):
        # Contract the i and i+1 tensors to prepare for SVD
        contraction = np.einsum('ijk,klm->ijlm', mps_new[i], mps_new[i+1])
        # Reshape the contraction to a matrix
        w = contraction.reshape(mps_new[i].shape[0] * mps_new[i].shape[1], mps_new[i+1].shape[1] * mps_new[i+1].shape[2])
        # Compute the SVD
        U, S, V = np.linalg.svd(w, full_matrices=False)
        # Update the MPS tensors
        mps_new[i] = U.reshape(mps_new[i].shape[0], mps_new[i].shape[1], U.shape[-1])
        mps_new[i+1] = np.dot(np.diag(S), V).reshape(U.shape[-1], mps_new[i+1].shape[1], mps_new[i+1].shape[2])

    mps = mps_new.copy()
    mps_new = mps.copy()
    # now throw it in reverse while enforcing the bond dimension
    for i in range(L-1, 0, -1):
        # Contract the i and i-1 tensors to prepare for SVD
        contraction = np.einsum('ijk,klm->ijlm', mps_new[i-1], mps_new[i])
        # Reshape the contraction to a matrix
        w = contraction.reshape(mps_new[i-1].shape[0] * mps_new[i-1].shape[1], mps_new[i].shape[1] * mps_new[i].shape[2])
        # Compute the SVD, now enforcing the truncation
        u, s_diag, v_t = truncate_svd(w, chi)
        # Update the MPS tensors
        mps_new[i-1] = np.dot(u, s_diag).reshape(mps_new[i-1].shape[0], mps_new[i-1].shape[1], u.shape[-1])
        mps_new[i] = v_t.reshape(u.shape[-1], mps_new[i].shape[1], mps_new[i].shape[2])
    return mps_new

        
    
    

        
    
    