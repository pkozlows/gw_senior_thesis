import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import gto, scf
import os

import numpy as np

from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points
from ase.dft.kpoints import bandpath


ase_cell = bulk('C', 'diamond', a=3.5668)

cell = gto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_cell)
cell.a = np.array(ase_cell.cell)
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build(None,None)

kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
kmf.xc = 'lda,vwn_rpa'

points = special_points['fcc']
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
path = bandpath([L, G, X, W, K, G], ase_cell.cell, npoints=50) # note change to bandpath
kpts = path.kpts
kpath, sp_points, labels = path.get_linear_kpoint_axis()

# Set the file path to save the check file in the current working directory
file_path = os.path.join(os.getcwd(), 'previous.chk')
print("File Path:", file_path)
kmf.chkfile = file_path
kmf.kernel()
