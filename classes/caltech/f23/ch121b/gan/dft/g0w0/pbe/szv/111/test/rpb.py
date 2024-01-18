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
file_path = os.path.join(os.getcwd(), 'previous.chk')

# Load the cell object
cell = load_cell(file_path)
cell.pbc = [True, True, True]
cell.build()

# Load the checkpoint file
scf_data = chkfile.load(file_path, 'scf')


# Create and populate the KRKS object
kmf = pbcdft.KRKS(cell)
kmf.__dict__.update(scf_data)

# Extracting HOMO and LUMO energies
nocc = cell.nelectron // 2  # Number of occupied orbitals
homo = max(kmf.mo_energy[0][:nocc])
lumo = min(kmf.mo_energy[0][nocc:])
band_gap = lumo - homo

print("HOMO energy:", homo)
print("LUMO energy:", lumo)
print("Band gap at Gamma point [eV]:", band_gap *27.211)
