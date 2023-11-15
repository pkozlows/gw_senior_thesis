import numpy as np
from pyscf import gto, dft, gw
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
dft_energies = []
gw_energies = []

# Loop over basis sets
for basis in basis_sets:
    mol.basis = basis
    mol.build()
    
    # Set up and run the DFT calculation
    mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.kernel()
    print(mol.nelectron)
    occupied_orbs = mol.nelectron//2

    dft_moe = mf.mo_energy
    dft_homo_energy = dft_moe[mol.nelectron // 2 - 1]
    dft_energies.append(dft_homo_energy)
    

    one_shot = gw.GW(mf)
    one_shot.kernel(orbs=range(occupied_orbs-3,occupied_orbs+3))
    print(one_shot.mo_energy)

    # Extract HOMO energy
    gw_moe = one_shot.mo_energy
    # assert(moe.all() != mf.mo_energy.all())
    gw_homo_energy = gw_moe[mol.nelectron // 2 - 1]  # HOMO is the last occupied orbital
    gw_energies.append(gw_homo_energy)

# Plotting
plt.plot(gw_energies, 'o-', label='GW')  # add label here
plt.plot(dft_energies, 'o-', label='DFT')  # add label here
plt.xticks(ticks=range(len(basis_sets)), labels=basis_sets, rotation=45)
plt.ylabel('HOMO Energy (Hartree)')
plt.title('HOMO Energy vs. Basis Set for Water @PBE')
plt.tight_layout()
plt.grid(True)
plt.legend()  # Add the legend
# plt.show()
plt.savefig('water_gw.png')






