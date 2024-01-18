# import statements
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft


from ase.build import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

import numpy as np

# ase
c = bulk('GaN', 'wurtzite', a=3.19, c=5.19)

# pyscf specs
cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(c)
cell.a = c.cell
cell.exp_to_discard = 0.1

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.build()

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
band_kpts = cell.get_abs_kpts(band_kpts)

# dft calculation
kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
kmf.xc = 'b3pw91'
kmf.kernel()

# get the data from the calculation
e_kn_2 = kmf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

# Saving data to .npy files
np.save('band_kpts.npy', band_kpts)
np.save('kpath.npy', kpath)
np.save('sp_points.npy', sp_points)
np.save('e_kn_2.npy', e_kn_2)
np.save('nbands.npy', cell.nao_nr())







