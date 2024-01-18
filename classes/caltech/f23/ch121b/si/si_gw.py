import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc import gto, scf
import pyscf.pbc.gw as pbcgw
import pyscf.pbc.dft as pbcdft
import numpy as np
import os
from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points
from ase.dft.kpoints import bandpath


ase_cell = bulk('Si', 'diamond', a=5.44)
cell = gto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_cell)
cell.a = np.array(ase_cell.cell)
cell.exp_to_discard = 0.1

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build(None,None)

kmf.__dict__.update(scf.chkfile.load(old_path, 'scf'))
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']

# Choose an appropriate path for the band structure calculation
path = bandpath([L, G, X, W, K, G], ase_cell.cell, npoints=50)
kpts = path.kpts
kpath, sp_points, labels = path.get_linear_kpoint_axis()

old_path = os.path.join(os.getcwd(), 'chunky.chk')
kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
# kmf.xc = 'PBE'
kmf.__dict__.update(scf.chkfile.load(old_path, 'scf'))

# starred AGW calculation based off of the mean field object that is  given in the check fail
new_path = os.path.join(os.getcwd(), 'new.chk')
gw = pbcgw.KRGW(kmf)
gw.chkfile = new_path
gw.kernel()