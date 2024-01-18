import os
import numpy as np
import matplotlib.pyplot as plt
from pyscf.lib import chkfile
from pyscf.pbc.lib.chkfile import load_cell
from ase.dft.kpoints import sc_special_points as special_points
from ase.dft.kpoints import bandpath
from ase.lattice import bulk
import pyscf.pbc.dft as pbcdft

# Set the file path for the checkpoint file
file_path = os.path.join(os.getcwd(), 'prv.chk')

# Load the cell object
cell = load_cell(file_path)
cell.pbc = [True, True, True]
cell.build()

# Load the checkpoint file
scf_data = chkfile.load(file_path, 'scf')


# Create and populate the KRKS object
kmf = pbcdft.KRKS(cell, cell.make_kpts([1,1,1]))
kmf.__dict__.update(scf_data)

ase_cell = bulk('C', 'diamond', a=3.5668)

# I want to see the time that it takes to run this plotting script
# Define the special k-points and paths for an FCC structure
points = special_points['fcc']
G = points['G']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
path = bandpath([L, G, X, W, K, G], ase_cell.cell, npoints=50) # note change to bandpath
kpts = path.kpts
kpath, sp_points, labels = path.get_linear_kpoint_axis()



# Calculate bands
k_points = cell.get_abs_kpts(kpts)
# start_time = time.time()
e_kn_2 = kmf.get_bands(k_points)[0]
# end_time = time.time()
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
    plt.plot(kpath, e_kn_2[:, n]*au2ev, color='#4169E1')
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')
plt.ylabel('Energy (eV)')

plt.savefig('reconstructed.png')