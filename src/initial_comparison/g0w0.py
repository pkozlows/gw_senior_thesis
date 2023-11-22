import numpy as np
import pyscf
from pyscf import gto, dft, gw

# Define the water molecule
molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)

# Set up and run the DFT calculation
mf = dft.RKS(molecule)
mf.xc = 'PBE'
mf.kernel()
mf_es = mf.mo_energy
print(mf_es)
print(mf_es.shape)

one_shot = gw.GW(mf)
occupied_orbs = molecule.nelectron//2
one_shot.kernel(orbs=range(occupied_orbs-3,occupied_orbs+3))
gw_es = one_shot.mo_energy
print(gw_es)
print(gw_es.shape)

# Define the range of orbitals for which you've computed the GW QP energies
orbital_range = range(occupied_orbs-3,occupied_orbs+3)

# Calculate the differences only for the specific range of orbitals
differences = mf_es[orbital_range] - gw_es[orbital_range]
print(differences)

