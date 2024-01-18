import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc import gto, scf
import pyscf.pbc.gw as pbcgw
import pyscf.pbc.dft as pbcdft
import numpy as np
import os
from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points
from ase.dft.kpoints import bandpath


ase_cell = bulk('GaN', 'wurtzite', a=3.19, c=5.19)
cell = gto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_cell)
cell.a = np.array(ase_cell.cell)
cell.exp_to_discard = 0.1

cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build(None,None)

# define the special points here
points = special_points['hexagonal']
G = points['G']
K = points['K']
M = points['M']
A = points['A']
L = points['L']
H = points['H']

# Choose an appropriate path for the band structure calculation
path = bandpath([G, M, K, G, A, L, H, A], ase_cell.cell, npoints=50)
kpts = path.kpts
kpath, sp_points, labels = path.get_linear_kpoint_axis()

old_path = os.path.join(os.getcwd(), 'previous.chk')
kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
kmf.xc = 'PBE'
kmf.__dict__.update(scf.chkfile.load(old_path, 'scf'))
# starred AGW calculation based off of the mean field object that is  given in the check fail
new_path = os.path.join(os.getcwd(), 'new.chk')
gw = pbcgw.KRGW(kmf)
gw.chkfile = new_path
gw.kernel()