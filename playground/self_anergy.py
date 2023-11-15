import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA
import numpy as np
import matplotlib.pyplot as plt

def real_corr_se(mol, frequency):
    '''Calculates the real part of the correlation self energy for a given molecule and frequency. Returns a matrix of correlation energies for each orbital at the given frequency. Only the diagonal of this matrix is considered.'''

    # find the number of orbitals
    n_orbitals = mol.nao_nr()
    # find the number of occupied orbitals
    n_occupied = mol.nelectron//2
    # find the number of virtual orbitals
    n_virtual = n_orbitals - n_occupied
    
    def calculated_excitation_es(orbital_energies, eri):
        '''Calculates the excitation energies and the R matrix for the molecule.'''
        # initialize the A matrix
        A = np.zeros((n_occupied, n_virtual, n_occupied, n_virtual))
        # loop over all of the relevant indices
        for i in range(n_occupied):
            for a in range(n_virtual):
                for j in range(n_occupied):
                    for b in range(n_virtual):
                        # first we add the relevant component of the electron repulsion integrals
                        A[i, a, j, b] += 4*eri[i, a+n_occupied, j, b+n_occupied]
                        # then we add the differences between orbital energies if they pass the delta function condition
                        if a == b and i == j:
                            A[i, a, j, b] += orbital_energies[a+n_occupied] - orbital_energies[i]
        reshaped_a = A.reshape(n_occupied*n_virtual, n_occupied*n_virtual)
        return np.linalg.eigh(reshaped_a)
    return

molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)
mf = rks.RKS(molecule).run()

pyscf_dtda = dTDA(mf).run()
pyscf_dtda.analyze()
print()
#  # this routine is for finding the orbital energies and the electron repulsion integrals for the molecule
#     mf = rks.RKS(molecule).run()
#     orbs = mf.mo_coeff
#     orbital_energies = mf.mo_energy
#     eri = mol.ao2mo(orbs, compact=False)
#     # we want to reshape them from the packed chemists notation
#     eri = eri.reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)

#     # now we want to calculate the excitation energies and the R matrix
#     omega, R = calculated_excitation_es(orbital_energies, eri)
    
#     # now that us compute the V matrix
#     W_pqia = 4*eri[:, :, n_occupied:, :n_occupied]
#     # now that us reshape it into a form we want
#     W_Ipq = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
#     V_npq = np.einsum('pqi,in->pqn', W_Ipq, R)
#     excitations = n_occupied*n_virtual
#     # initialize the correlation energies
#     correlation_energies = np.zeros((n_orbitals, n_orbitals))
#     # we start out by just getting the diagonal elements
#     for p in range(n_orbitals):
#         for u in range(excitations):
#             for j in range(n_occupied):
#                 correlation_energies[p, p] += V_npq[p, j, u]*V_npq[p, j, u]/(orbital_energies[j] - omega[u] - frequency)
#             for b in range(n_virtual):
#                 correlation_energies[p, p] += V_npq[p, b, u]*V_npq[p, b, u]/(orbital_energies[b] - omega[u] - frequency)

        
#     return np.diag(correlation_energies)
    