import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA
import numpy as np
import matplotlib.pyplot as plt

def real_corr_se(mol, frequency):
    '''Calculates the real part of the correlation self energy for a given molecule and frequency. Returns a matrix of correlation energies for each orbital at the given frequency. Only the diagonal of this matrix is considered.'''
    
    # I start by getting molecule integral pieces for any approximation i want to use

    # find the number of orbitals
    n_orbitals = mol.nao_nr()
    # find the number of occupied orbitals
    n_occupied = mol.nelectron//2
    # find the number of virtual orbitals
    n_virtual = n_orbitals - n_occupied

    # run the mean field calculation
    mf = rks.RKS(mol).run()
    # find the MO coefficients
    orbs = mf.mo_coeff
    # find the MO energies
    orbital_energies = mf.mo_energy
    
    # find the electron repulsion integrals in AO basis
    eri = mol.ao2mo(orbs, compact=False)
    # we want to reshape them from the packed chemists notation
    eri = eri.reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)

    def dtda_excitations(orbital_energies, eri):
        '''Calculates the excitation energies and the R matrix for the molecule in the tda.'''
        # initialize the A matrix
        A = np.zeros((n_occupied, n_virtual, n_occupied, n_virtual))
        # loop over all of the relevant indices
        for i in range(n_occupied):
            for a in range(n_virtual):
                for j in range(n_occupied):
                    for b in range(n_virtual):
                        # first we add the relevant component of the electron repulsion integrals
                        A[i, a, j, b] += np.sqrt(2)*eri[i, a+n_occupied, j, b+n_occupied]
                        # then we add the differences between orbital energies if they pass the delta function condition
                        if a == b and i == j:
                            A[i, a, j, b] += orbital_energies[a+n_occupied] - orbital_energies[i]
        reshaped_a = A.reshape(n_occupied*n_virtual, n_occupied*n_virtual)
        # converted from hartress to electron volts
        return np.linalg.eigh(reshaped_a) 


    # now we want to calculate the excitation energies and the R matrix
    omega, R = dtda_excitations(orbital_energies, eri)
    
    # now that us compute the V matrix
    W_pqia = np.sqrt(2)*eri[:, :, n_occupied:, :n_occupied]
    # now that us reshape it into a form we want
    W_Ipq = W_pqia.reshape(n_orbitals, n_orbitals, n_occupied*n_virtual)
    V_npq = np.einsum('pqi,in->pqn', W_Ipq, R)
    excitations = n_occupied*n_virtual
    # initialize the correlation energies
    correlation_energies = np.zeros((n_orbitals, n_orbitals))
    # we start out by just getting the diagonal elements
    for p in range(n_orbitals):
        for u in range(excitations):
            for j in range(n_occupied):
                correlation_energies[p, p] += V_npq[p, j, u]*V_npq[p, j, u]/(orbital_energies[j] - omega[u] - frequency)
            for b in range(n_virtual):
                correlation_energies[p, p] += V_npq[p, b, u]*V_npq[p, b, u]/(orbital_energies[b] - omega[u] - frequency)

    # returned the diagonal converting from her hartrees to electron volts
    return np.diag(correlation_energies)*27.2114    
    
molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)
mf = rks.RKS(molecule).run()
pyscf_dtda = dTDA(mf).run()
pyscf_dtda.analyze()

# Define a range of frequencies
frequencies = np.linspace(0, 0.01, 5)

# Compute correlation energies for each frequency
all_correlation_energies = np.zeros((molecule.nao, len(frequencies)))

for idx, freq in enumerate(frequencies):
    correlation_energies = real_corr_se(molecule, freq)
    all_correlation_energies[:, idx] = correlation_energies
# the frequencies are currently in harte's have start change then to electron volts
frequencies *= 27.2114

print(all_correlation_energies)
