import numpy as np
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