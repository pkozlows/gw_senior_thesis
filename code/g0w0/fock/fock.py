import pyscf
from pyscf.dft import rks

import numpy as np

def fock_matrix_hf(molecule):
    '''Calculates the fock matrix for a given set of molecular orbitals and occupations.'''

    # find all of the hopeful parameters for the molecule
    n_orbitals = molecule.nao_nr()
    n_occupied = molecule.nelectron//2
    n_virtual = n_orbitals - n_occupied

    # start the mean field calculations
    
    dft = rks.RKS(molecule)
    dft.xc = 'hf'
    dft.verbose = 0
    dft.kernel()

    hf = pyscf.scf.RHF(molecule)
    hf.verbose = 0
    hf.kernel()
    
    mf = hf
    
    # find the number of orbitals
    orbital_energies = mf.mo_energy
    mo_coeffs = mf.mo_coeff
    mo_occ = mf.mo_occ
    n_orbitals = orbital_energies.shape[0]
    
    # first we want to find the charge density matrix
    # initialize the fock matrix
    fock = np.zeros((n_orbitals, n_orbitals))
    # loop over all orbitals twice
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            # check if we are dealing with a day cando element
            if p == q:
                fock[p,q] += orbital_energies[p]
    return fock
  
molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)
# print(fock_matrix_hf(molecule))

# get me the orbital anergy for the highest occupied molecular orbital and the lowest unoccupied molecular orbital
print(fock_matrix_hf(molecule)[9,9])
print(fock_matrix_hf(molecule)[10,10])