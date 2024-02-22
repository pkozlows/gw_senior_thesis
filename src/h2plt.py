import numpy as np
from pyscf.dft import rks
import pyscf

molecule = pyscf.M(
        atom = 'H  0 0 0; H  0 0 0.735',
        basis = 'ccpvdz',
        symmetry = True,
)
mf = rks.RKS(molecule)
mf.xc = 'hf'
mf.verbose = 0
mf.kernel()
dancey_matrix = mf.make_rdm1()
print(dancey_matrix)

# # make a four lope that goes over bond distances from point five angstroms to six angstroms for the hydrogen molecule h2
# for bond_distance in np.arange(0.5, 6.0, 0.1):
