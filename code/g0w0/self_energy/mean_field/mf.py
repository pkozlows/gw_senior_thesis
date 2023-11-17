import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA
molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)
# find the number of orbitals
n_orbitals = molecule.nao_nr()
# find the number of occupied orbitals
n_occupied = molecule.nelectron//2
# find the number of virtual orbitals
n_virtual = n_orbitals - n_occupied
mf = rks.RKS(molecule).run()
pyscf_dtda = dTDA(mf).run()
pyscf_dtda.analyze()
orbs = mf.mo_coeff
orbital_energies = mf.mo_energy
eri = molecule.ao2mo(orbs, compact=False)
# we want to reshape them from the packed chemists notation
eri = eri.reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)