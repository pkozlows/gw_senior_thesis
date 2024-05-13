import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA, dRPA
import numpy as np
from mf import setup_molecule, calculate_mean_field
import numpy as np
from tda import my_dtda, my_drpa

def lin_gw_dm(td, mf):
    '''Calculates the linearized GW self energy for a given molecule and frequency. Returns a matrix of correlation energies for each orbital at the given frequency. All elements are considered.'''
    # from mean field
    orbital_energies = mf.mo_energy
    n_orbitals = mf.mo_energy.shape[0]
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied

    omega, V_pqn = td

    dm = np.zeros((n_orbitals, n_orbitals))

    # started with the occupied block
    # Create the delta matrix
    delta_matrix = np.identity(n_occupied)
    # create the part in the summation
    occ_num = np.einsum('ias,jas->ijas', V_pqn[:n_occupied, n_occupied:, :], V_pqn[:n_occupied, n_occupied:, :])
    occ_denom = (orbital_energies[:n_occupied, None, None] - orbital_energies[None, n_occupied:, None] - omega[None, None, :])
    combined_occ_denom = np.einsum('ias,jas->ijas', occ_denom, occ_denom)
    occ_block = (-1)*np.einsum('ijas,ijas->ij', occ_num, 1/(combined_occ_denom))
    # add the delta matrix
    occ_block += delta_matrix

    # now the virtual block
    virt_num = np.einsum('ais,bis->iabs', V_pqn[n_occupied:, :n_occupied, :], V_pqn[n_occupied:, :n_occupied, :])
    virt_denom = (orbital_energies[:n_occupied, None, None] - orbital_energies[None, n_occupied:, None] - omega[None, None, :])
    combined_virt_denom = np.einsum('ias,ibs->iabs', virt_denom, virt_denom)
    virt_block = np.einsum('iabs,iabs->ab', virt_num, 1/(combined_virt_denom))

    # now the mixed block
    first_num = np.einsum('ias,bas->iabs', V_pqn[:n_occupied, n_occupied:, :], V_pqn[n_occupied:, n_occupied:, :])
    first_denom = (orbital_energies[:n_occupied, None, None] - orbital_energies[None, n_occupied:, None] - omega[None, None, :])
    first_sum = np.einsum('iabs,ias->ib', first_num, 1/first_denom)

    second_num = np.einsum('ijs,bjs->ijbs', V_pqn[:n_occupied, :n_occupied, :], V_pqn[n_occupied:, :n_occupied, :])
    second_denom = (orbital_energies[:n_occupied, None, None] - orbital_energies[None, n_occupied:, None] - omega[None, None, :])
    second_sum = np.einsum('ijbs,jbs->ib', second_num, 1/second_denom)

    mixed_block = (1/(orbital_energies[:n_occupied, None] - orbital_energies[None, n_occupied:]))*(first_sum - second_sum)

    # if the imaginary part is 0, convert each block to the dtype of np.real
    if np.imag(occ_block).all() == 0:
        occ_block = np.real(occ_block)
    if np.imag(virt_block).all() == 0:
        virt_block = np.real(virt_block)
    if np.imag(mixed_block).all() == 0:
        mixed_block = np.real(mixed_block)

    # stack the blocks to form the density matrix
    dm[:n_occupied, :n_occupied] += occ_block
    dm[n_occupied:, n_occupied:] += virt_block
    dm[:n_occupied, n_occupied:] += mixed_block
    dm[n_occupied:, :n_occupied] += mixed_block.T
    

    return 2*dm
# mol = setup_molecule('h2')
# mf = calculate_mean_field(mol, 'hf')
# td = my_dtda(mf)
# dm = lin_gw_dm(td, mf)
# # diagonalize the density matrix
# e, v = np.linalg.eigh(dm)
# print(v.shape)
# # ;rint the natural orbital occupation numbers
# print(e)
# # front the sum of the natural or brutal occupation numbers
# print(np.sum(e))
# # make a rdm_1 using the pyscf implementation and diagonalize it for the smae mf
# rdm_1 = x.make_rdm1()
# e, v = np.linalg.eigh(rdm_1)
# print(e)
# print(np.sum(e))
# # get the trace of this matrix
# print(np.trace(rdm_1))