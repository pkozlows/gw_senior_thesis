'''
G0W0 with k-points sampling
'''

from functools import reduce
import numpy
from pyscf.pbc import gto, scf, gw

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

#
# KDFT and KGW with 2x2x2 k-points
#
kpts = cell.make_kpts([1,1,1])
kmf = scf.KRKS(cell).density_fit()
kmf.kpts = kpts
emf = kmf.kernel()

# Extracting HOMO and LUMO energies
nocc = cell.nelectron // 2  # Number of occupied orbitals
homo = max(kmf.mo_energy[0][:nocc])
lumo = min(kmf.mo_energy[0][nocc:])
band_gap = lumo - homo

print("HOMO energy:", homo)
print("LUMO energy:", lumo)
print("Band gap at Gamma point:", band_gap)


# # Default is AC frequency integration
# mygw = gw.KRGW(emf)
# mygw.kernel()
# print("KRGW energies =", mygw.mo_energy)# dogyank