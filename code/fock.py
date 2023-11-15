import pyscf
import numpy as np

def fock_matrix(mo_coeffs, mo_occ):
    '''Calculates the fock matrix for a given set of molecular orbitals and occupations.'''
    # find the number of orbitals
    n_orbitals = mo_coeffs.shape[0]
    # first we want to find the charge density matrix
    dm = pyscf.scf.hf.make_rdm1(mo_coeffs, mo_occ)
    # initialize the fock matrix
    fock = np.zeros((n_orbitals, n_orbitals))
    # now we want to loop over all orbitals
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            # first we want to get the core H elements. No further looping is required here.
            fock[p, q] += h_core[p, q]
  
    