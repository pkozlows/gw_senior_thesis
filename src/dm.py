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

    # start a loop over the occupied and virtual orbitals

    occ_num = np.einsum('ias,jas->ij', V_pqn[:n_occupied, n_occupied:, :], V_pqn[:n_occupied, n_occupied:, :])

    virt_num = np.einsum('ias,ibs->ab', V_pqn[n_occupied:, :n_occupied, :], V_pqn[n_occupied:, :n_occupied, :])

    for p in range(n_orbitals):
        for q in range(n_orbitals):
            # check if they are both occupied
            if p < n_occupied and q < n_occupied:
                # if so, add the delta function
                dm[p, q] += 1
                # do the sum
                # start with the numerator
                # now the denominator
                numerator = occ_num[p, q]
            if p >= n_occupied and q >= n_occupied:
                dm[p, q] += 1
            if p < n_occupied and q >= n_occupied or p >= n_occupied and q < n_occupied:
                prefactor = 1/(orbital_energies[p] - orbital_energies[q])
                
            
        

    # let's start with the occupied block
    # first add the delta function
    delta_matrix = np.eye(n_occupied)
    dm[:n_occupied, :n_occupied] += delta_matrix
    # prepare for einsum
    # convert the numerator of the excitation vectors into a singular object
    occ_num = np.einsum('ias,jas->ij', R[:n_occupied, n_occupied:, :], R[:n_occupied, n_occupied:, :])
    occ_denom = ()
    return

mol = setup_molecule()
mf, n_orbitals, n_occupied, n_virtual, orbital_energies = calculate_mean_field(mol, 'hf')

td = my_dtda(mf)
lin_gw_dm(td, mf)