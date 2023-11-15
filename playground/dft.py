import numpy as np
from pyscf import gto, dft
import matplotlib.pyplot as plt

# Define the water molecule
mol = gto.Mole()
mol.atom = '''
O  0 0 0
H  0 0.758602 0.504284
H  0 0.758602 -0.504284
'''
mol.unit = 'A'

# List of basis sets
basis_sets = ['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ']

# Placeholder for HOMO energies
homo_energies = []

# Loop over basis sets
for basis in basis_sets:
    mol.basis = basis
    mol.build()
    
    # Set up and run the DFT calculation
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    mf.kernel()

    # Extract HOMO energy
    mo_energy = mf.mo_energy
    homo_energy = mo_energy[mol.nelectron // 2 - 1]  # HOMO is the last occupied orbital
    homo_energies.append(homo_energy)

# Plotting
plt.plot(homo_energies, 'o-')
plt.xticks(ticks=range(len(basis_sets)), labels=basis_sets, rotation=45)
plt.ylabel('HOMO Energy (Hartree)')
plt.title('HOMO Energy vs. Basis Set for Water @DFT')
plt.tight_layout()
plt.grid(True)
# plt.show()
plt.savefig('water_dft.png')
