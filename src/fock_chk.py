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
conversion_factor = 27.2114

ks_pbe = rks.RKS(molecule)
ks_pbe.xc = 'pbe'
ks_pbe.kernel()
core_pbe = ks_pbe.get_hcore()
mos = ks_pbe.mo_coeff
mos_transpose = np.transpose(mos)
mo_density_mat = np.einsum('pi,iq->pq', mos, mos_transpose)
dot = np.dot(mos, mos.T)
print(molecule.intor('int1e_ovlp').shape)
print(np.allclose(mo_density_mat, dot))
pyscf_desity_mat = ks_pbe.make_rdm1()
print(np.allclose(2*np.dot(mos, mos.T), pyscf_desity_mat))

eri = molecule.ao2mo(mos, compact=False)
eri = eri.reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)
# initialize the folk matrix
fock = np.zeros((n_orbitals, n_orbitals))
# add the core hamiltonian
fock += core_pbe
# add the coulomb term
print(1)

