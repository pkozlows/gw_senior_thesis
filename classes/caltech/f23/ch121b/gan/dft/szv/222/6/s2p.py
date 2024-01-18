import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import gto, scf
import os

import numpy as np
import matplotlib.pyplot as plt

from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points
from ase.dft.kpoints import bandpath


ase_cell = bulk('GaN', 'wurtzite', a=3.19, c=5.19)

cell = gto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_cell)
cell.a = np.array(ase_cell.cell)
cell.exp_to_discard = 0.1

cell.basis = 'gth-szv'
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
new_path = os.path.join(os.getcwd(), 's2p.chk')
kmf = pbcdft.KRKS(cell, cell.make_kpts([2,2,2]))
kmf.xc = 'b3pw91'
kmf.chkfile = new_path
kmf.kernel()

# Calculate bands
k_points = cell.get_abs_kpts(kpts)
e_kn_2 = kmf.get_bands(k_points)[0]
e_kn_2 = np.array(e_kn_2)
# Adjust for the Fermi level
vbmax = max(e_kn_2[:, cell.nelectron//2-1])
e_kn_2 -= vbmax

au2ev = 27.21139
emin = -1*au2ev
emax = 1*au2ev

# Plotting
plt.figure(figsize=(5, 6))
nbands = cell.nao_nr()
for n in range(nbands):
    plt.plot(kpath, [e[n]*au2ev for e in e_kn_2], color='#87CEEB')
# Plot band 18 with a red line
plt.plot(kpath, e_kn_2[:, 18] * au2ev, color='red')

for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in [r'\Gamma', 'M', 'K', r'\Gamma', 'A', 'L', 'H', 'A']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin//2, ymax=emax//2)
plt.xlabel('k-vector')
plt.ylabel('Fermi energy [eV]')

# Get the energy of band 18 at the Γ point
gamma_point_indices =[index for index, point in enumerate(k_points) if all(element == 0 for element in point)]
assert(len(gamma_point_indices) == 2)

k_idx = gamma_point_indices[1]
energy_band18_at_Gamma = e_kn_2[k_idx, 18] * au2ev
# Plot a vertical line from Fermi level to band 18 at Γ point
plt.plot([sp_points[3], sp_points[3]], [0, energy_band18_at_Gamma], color='yellow', linestyle='--')

# Calculate and annotate the energy difference
energy_difference = abs(energy_band18_at_Gamma)  # Absolute value to ensure positive difference
mid_point_energy = energy_band18_at_Gamma / 2  # Midpoint between Fermi energy and band 18 energy
plt.annotate(f'Energy: {energy_difference:.2f} eV',
             xy=(sp_points[3], mid_point_energy),
             xytext=(sp_points[3] + 0.3, mid_point_energy),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             horizontalalignment='left',
             verticalalignment='center')

plt.savefig('szv_222_11-14.png')






