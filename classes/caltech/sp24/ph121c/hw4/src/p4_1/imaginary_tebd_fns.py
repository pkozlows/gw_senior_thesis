import numpy as np
from hw2.src.p5_5 import truncate_svd
from scipy.linalg import expm
from hw4.src.contraction_fns import apply_local_hamiltonian
    
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
# this is a test

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
    
    return mps

def create_initial_mps(l, name):
    up_physical = np.array([1, 0])
    down_physical = np.array([0, 1])
    single_sight = np.array([1, -np.sqrt(3)]) / 2 
    mps = []
    for i in range(l):
        if i == 0:
            up_reshape = up_physical.reshape(2, 1)
            if name == 'ferro' or name == 'neel':
                mps.append(up_reshape)
            elif name == 'three':
                mps.append(single_sight.reshape(2, 1))
        elif i == l-1:
            up_reshape = up_physical.reshape(1, 2)
            down_reshape = down_physical.reshape(1, 2)
            if name == 'ferro':
                mps.append(up_reshape)
            elif name == 'neel':
                mps.append(up_reshape if i % 2 == 0 else down_reshape)
            elif name == 'three':
                mps.append(single_sight.reshape(1, 2))
        else:
            up_reshape = up_physical.reshape(1, 2, 1)
            down_reshape = down_physical.reshape(1, 2, 1)
            if name == 'ferro':
                mps.append(up_reshape)
            elif name == 'neel':
                mps.append(up_reshape if i % 2 == 0 else down_reshape)
            elif name == 'three':
                mps.append(single_sight.reshape(1, 2, 1))
    return mps

# make a function that will get red of all of the virtual indices to just leave the physical indices of an mps
def flatten_mps(mps):
    '''Flatten the MPS to remove all virtual indices.'''
    L = len(mps)
    combined = mps[0]

    # Sequentially contract the tensors
    for i in range(1, L-1):
        combined = np.einsum('...a,abc->...bc', combined, mps[i])

    # treat the final case wdifferently
    combined = np.einsum('...a,ab->...b', combined, mps[L-1])

    # Flatten the final combined tensor
    flattened_mps = combined.flatten()
    return flattened_mps

def imaginary_tebd_step(current_mps, chi, time, gates):
    """Imaginary time evolution using TEBD."""
    gate_field, gate_odd, gate_even = gates
    trotterized = apply_trotter_gates(current_mps, gate_field, gate_odd, gate_even)
    mps_enforced = enforce_bond_dimension(trotterized, chi)
    return mps_enforced

        

    

        
    
    

        
    
    