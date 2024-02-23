import numpy as np
from pyscf.dft import rks
import pyscf
from pyscf import scf, fci
import matplotlib.pyplot as plt
from dm import lin_gw_dm
from tda import my_dtda, my_drpa

# Initialize lists to hold bond distances and occupations for plotting
bond_distances = []
occupations = {'hf': [], 'fci': [], 'dtda': [], 'drpa': []}

# Loop over bond distances
for bond_distance in np.arange(0.5, 6.0, 0.1):
    bond_distances.append(bond_distance)
    molecule = pyscf.M(
        atom='H  0 0 0; H  0 0 ' + str(bond_distance),
        basis='ccpvdz',
        symmetry=True,
    )
    # start with hf
    mf = scf.RHF(molecule)
    mf.kernel()
    norbs = mf.mo_energy.shape[0]
    hf_dm = np.diag(mf.mo_occ)
    # now do fci
    fci_solver = fci.FCI(molecule, mf.mo_coeff)
    e, fci_wfn = fci_solver.kernel()
    fci_dm = fci_solver.make_rdm1(fci_wfn, norbs, mf.mol.nelectron)
    # now do dtda
    g0w0_dtda_dm = lin_gw_dm(my_dtda(mf), mf)
    # tda_e, tda_wfn = np.linalg.eigh(g0w0_dtda_dm)
    # diag_tda = np.diag(tda_e)
    # now do drpa
    g0w0_drpa_dm = lin_gw_dm(my_drpa(mf), mf)
    # drpa_e, drpa_wfn = np.linalg.eigh(g0w0_drpa_dm)
    # diag_drpa = np.diag(drpa_e)

    # Add occupations for each method
    for key, dm in zip(['hf', 'fci', 'dtda', 'drpa'], [hf_dm, fci_dm, g0w0_dtda_dm, g0w0_drpa_dm]):
        if key in ['dtda', 'drpa']:
            # For 'dtda' and 'drpa', multiply both state 0 and state 1 values by 2
            # Negate the second value as per your requirement
            occupations[key].append((2*dm[0, 0],  -2*dm[1, 1]))
        else:
            # For 'hf' and 'fci', use the values directly without modification
            occupations[key].append((dm[0, 0], dm[1, 1]))

# Plotting
markers = {'hf': '*', 'fci': '+', 'dtda': 'x', 'drpa': '^'}
colors = {1: 'b', 2: 'r'}
for state in [1, 2]:
    for method in ['hf', 'fci', 'dtda', 'drpa']:
        plt.plot(bond_distances, [occ[state-1] for occ in occupations[method]], 
                 marker=markers[method], color=colors[state], linestyle='None', label=f"{method.upper()} State {state}" if state == 1 else None)

# Label the axes
plt.xlabel('Bond Distance (Angstroms)')
plt.ylabel('Natural Occupation')

# Creating a custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='b', lw=4, label='State 1 (Blue)'),
    Line2D([0], [0], color='r', lw=4, label='State 2 (Red)'),
    Line2D([0], [0], marker='*', color='w', label='HF', markerfacecolor='k', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='+', color='w', label='FCI', markerfacecolor='k', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='x', color='w', label='dTDA', markerfacecolor='k', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='dRPA', markerfacecolor='k', markeredgecolor='k', markersize=10),
]

# Now add this updated legend to your plot
plt.legend(handles=legend_elements, loc='best', edgecolor='black')



# Title and save
plt.title('Natural Occupation of the First Two States of H2')
plt.savefig('h2_occupations.png')
