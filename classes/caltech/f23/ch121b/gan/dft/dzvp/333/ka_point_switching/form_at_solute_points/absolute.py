# import statements
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

import matplotlib.pyplot as plt

from ase.build import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath
from ase.visualize import view

import numpy as np

# ase
c = bulk('GaN', 'wurtzite', a=3.19, c=5.19)
# print(c.get_volume())
# view(c)

# pyscf specs
cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = c.cell

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
band_kpts, kpath, sp_points = get_bandpath([G, M, K, G, A, L, H, A], c.cell, npoints=50)
absolute_points = cell.get_abs_kpts(band_kpts)

# dft calculation
kmf = pbcdft.KRKS(cell, cell.make_kpts([3,3,3]))
kmf.kernel()

# get the data from the calculation
e_kn_2 = kmf.get_bands(absolute_points)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

# Saving data to .npy files
np.save('absolute_points.npy', absolute_points)
np.save('band_kpts.npy', band_kpts)
np.save('kpath.npy', kpath)
np.save('sp_points.npy', sp_points)
np.save('e_kn_2.npy', e_kn_2)
np.save('nbands.npy', cell.nao_nr())






