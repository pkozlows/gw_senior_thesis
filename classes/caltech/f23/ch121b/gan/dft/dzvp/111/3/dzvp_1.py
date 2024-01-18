import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import gto, scf
import os

import numpy as np

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

# Set the file path to save the check file in the current working directory
file_path = os.path.join(os.getcwd(), 'check_file.chk')
kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
kmf.xc = 'b3pw91'
kmf.chkfile = file_path
kmf.kernel()
