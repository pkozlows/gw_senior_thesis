#Original PySCF GW (with frequency)
from pyscf import gw
from pyscf import tddft
import pyscf
import numpy as np
from pyscf.dft import rks
from pyscf import scf
from mf import setup_molecule, calculate_mean_field

molecule = setup_molecule()
n_orbitals = molecule.nao_nr()
n_occupied = molecule.nelectron//2
conversion_factor = 27.2114

ks_pbe = rks.RKS(molecule)
ks_pbe.xc = 'pbe'
ks_pbe.kernel()
core_pbe = ks_pbe.get_hcore()
mo_coeffs = ks_pbe.mo_coeff
occ_mos = mo_coeffs[:, :n_occupied]
eri_ao = molecule.intor('int2e').reshape((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
my_density_mat = 2*np.einsum('pi,qi->pq', occ_mos, occ_mos)
pyscf_desity_mat = ks_pbe.make_rdm1()
print(np.allclose(my_density_mat, pyscf_desity_mat))

# initialize the folk matrix
ao_fock = np.zeros((n_orbitals, n_orbitals))
# add the core hamiltonian
ao_fock += core_pbe
coulumb_matrix = np.zeros((n_orbitals, n_orbitals))
coulumb_matrix += np.einsum('rs,pqrs->pq', pyscf_desity_mat, eri_ao)
# initialize the exchange term
exchange_matrix = np.zeros((n_orbitals, n_orbitals))
exchange_matrix += np.einsum('rs,prqs->pq', pyscf_desity_mat, eri_ao)
# add the terms
ao_fock += coulumb_matrix - 0.5*exchange_matrix
mo_fock = np.einsum('pi,ij,qj->pq', mo_coeffs, ao_fock, mo_coeffs)
print(mo_fock.diagonal()*conversion_factor)


