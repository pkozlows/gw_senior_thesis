import os
import numpy as np
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from ase.lattice import bulk
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.gw import krgw_ac

ase_cell = bulk('GaN', 'wurtzite', a=3.19, c=5.19)

cell = gto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_cell)
cell.a = np.array(ase_cell.cell)
cell.exp_to_discard = 0.1

cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.rcut = 93
cell.build()

kpts = cell.make_kpts([1,1,1])
kmf = dft.KRKS(cell).density_fit()
kmf.kpts = kpts
kmf.xc = 'pbe'
kmf.kernel()

# Default is AC frequency integration
mygw = krgw_ac.KRGWAC(kmf)
mygw.kernel()

# # Extracting HOMO and LUMO energies
nocc = cell.nelectron // 2  # Number of occupied orbitals

au2ev = 27.211

dft_homo = max(kmf.mo_energy[0][:nocc])
dft_lumo = min(kmf.mo_energy[0][nocc:])
dft_gap = dft_lumo - dft_homo
print("DFT HOMO energy:", dft_lumo*au2ev)
print("DFT LUMO energy:", dft_homo*au2ev)
print("DFT Band gap at Gamma point [eV]:", dft_gap*au2ev)

gw_homo = max(mygw.mo_energy[0][:nocc])
gw_lumo = min(mygw.mo_energy[0][nocc:])
gw_gap = gw_lumo - gw_homo
print("GW HOMO energy:", gw_lumo*au2ev)
print("GW LUMO energy:", gw_homo*au2ev)
print("GW Band gap at Gamma point [eV]:", gw_gap*au2ev)
