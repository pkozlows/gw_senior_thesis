import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import gto, scf
import os

import numpy as np
import matplotlib.pyplot as plt

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

points = special_points['fcc']
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
path = bandpath([L, G, X, W, K, G], ase_cell.cell, npoints=50) # note change to bandpath
kpts = path.kpts
kpath, sp_points, labels = path.get_linear_kpoint_axis()

kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
kmf.xc = 'lda,vwn_rpa'
kmf.kernel()

# prepare the calculation for plotting
k_points = cell.get_abs_kpts(kpts)
e_kn = kmf.get_bands(k_points)[0]
e_kn = np.array(e_kn)
# Adjust for the Fermi level
vbmax = max(e_kn[:, cell.nelectron//2-1])
e_kn -= vbmax

# unit conversion
au2ev = 27.21139

# set the minimum and maximum energies
emin = -1*au2ev
emax = 1*au2ev

# Plot the bands
plt.figure(figsize=(5, 6))
nbands = cell.nao_nr()
for n in range(nbands):
    plt.plot(kpath, [e[n]*au2ev for e in e_kn], color='#87CEEB')
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')
plt.ylabel('Fermi energy [eV]')
# annotate the energy at the gamma point
# Get the energy of band 7 at the Γ point
gamma_point_indices =[index for index, point in enumerate(k_points) if all(element == 0 for element in point)]
assert(len(gamma_point_indices) == 2)

k_idx = gamma_point_indices[1]
energy_band18_at_Gamma = e_kn[k_idx, 7] * au2ev
# Plot a vertical line from Fermi level to band 7 at Γ point
plt.plot([sp_points[3], sp_points[3]], [0, energy_band18_at_Gamma], color='yellow', linestyle='--')

# Calculate and annotate the energy difference
energy_difference = abs(energy_band18_at_Gamma)  # Absolute value to ensure positive difference
mid_point_energy = energy_band18_at_Gamma / 2  # Midpoint between Fermi energy and band 7 energy
plt.annotate(f'Energy: {energy_difference:.2f} eV',
             xy=(sp_points[3], mid_point_energy),
             xytext=(sp_points[3] + 0.3, mid_point_energy),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='left',
                verticalalignment='center') 
plt.savefig('sample_diamond.png')   

