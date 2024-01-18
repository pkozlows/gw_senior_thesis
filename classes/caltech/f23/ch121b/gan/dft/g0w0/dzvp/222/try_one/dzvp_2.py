# import statements
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

import matplotlib.pyplot as plt

from ase.build import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath
from ase.visualize import view

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
band_kpts = cell.get_abs_kpts(band_kpts)

# dft calculation
kmf = pbcdft.KRKS(cell, cell.make_kpts([2,2,2]))
kmf.kernel()

# get the data from the calculation
e_kn_2 = kmf.get_bands(band_kpts)[0]
vbmax = -99
for en in e_kn_2:
    vb_k = en[cell.nelectron//2-1]
    if vb_k > vbmax:
        vbmax = vb_k
e_kn_2 = [en - vbmax for en in e_kn_2]

# plotting the data
au2ev = 27.21139
emin = -1*au2ev
emax = 1*au2ev
plt.figure(figsize=(5, 6))
nbands = cell.nao_nr()

# Change the color and line width of your bands here
for n in range(nbands):
    plt.plot(kpath, [e[n]*au2ev for e in e_kn_2], color='black', linewidth=1.5)

# Add the Fermi energy level
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label="E Fermi")

# Make your plot look polished
plt.xticks(sp_points, ['$%s$' % n for n in [r'\Gamma', 'M', 'K', r'\Gamma', 'A', 'L', 'H', 'A']], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('E-E$_F$ (Hartree)', fontsize=12)   # Labeling the y-axis

# Adjusting the appearance
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-', linewidth=1)
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector', fontsize=12)
plt.tight_layout()

# Saving the plot
plt.savefig('dzvp_2_modified.png')





