import numpy as np
import pyscf
from mf import setup_molecule, calculate_mean_field
from functools import reduce
def simple_fock(mf):
    '''Calculates the fock matrix for a given set of molecular orbitals and occupations.'''

    
    n_orbitals = mf.mol.nao_nr()
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied
    # get the orbital energies
    orbital_energies = mf.mo_energy
    
    # initialize the fock matrix
    fock = np.zeros((n_orbitals, n_orbitals))
    fock += np.diag(orbital_energies)
    return fock
  
def fock_dft(mf):
    '''Calculates the HF fock matrix using the DFT electron density in AO basis for a given set of molecular orbitals and occupations.

    Parameters:
    mf (object): The object representing the mean-field calculation.

    Returns:
    mo_fock (ndarray): The Fock matrix in the molecular orbital basis.
    '''

    n_orbitals = mf.mol.nao_nr()
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied
    # get the orbital energies
    orbital_energies = mf.mo_energy

    # initialize the ao_fock matrix in the atomic orbital basis
    ao_fock = np.zeros((n_orbitals, n_orbitals))
    h_core = mf.get_hcore()
    # in nationalize the codon matrix
    coulumb_matrix = np.zeros((n_orbitals, n_orbitals))
    coulumb_matrix += np.einsum('rs,pqrs->pq', mf.make_rdm1(), mf.mol.intor('int2e').reshape((n_orbitals, n_orbitals, n_orbitals, n_orbitals)))
    # initialize the exchange term
    exchange_matrix = np.zeros((n_orbitals, n_orbitals))
    exchange_matrix += np.einsum('rs,prqs->pq', mf.make_rdm1(), mf.mol.intor('int2e').reshape((n_orbitals, n_orbitals, n_orbitals, n_orbitals)))
    # add the terms
    ao_fock += (h_core + coulumb_matrix - 0.5*exchange_matrix)
    # convert the fock matrix to the molecular orbital basis
    mo_fock = np.einsum('ia,ij,jb->ab', mf.mo_coeff, ao_fock, mf.mo_coeff)

    
    return mo_fock

def pyscf_fock_dft(mf):
    '''Calculates the HF fock matrix using the DFT electron density in AO basis for a given set of molecular orbitals and occupations.'''

    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    ao_fock = mf.get_fock(vhf=vhf, dm=dm)
    # convert the fock matrix to the molecular orbital basis
    mo_fock = reduce(np.dot, (mf.mo_coeff.conj().T, ao_fock, mf.mo_coeff))
    # mo_fock = np.einsum('ia,ij,jb->ab', mf.mo_coeff, ao_fock, mf.mo_coeff)
    return mo_fock