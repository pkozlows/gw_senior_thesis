import numpy as np
from tda import real_corr_se
from mf import setup_molecule, calculate_mean_field
from fock import fock_matrix_hf
from pyscf import gw
from pyscf import tddft
import pyscf
from pyscf.dft import rks
from pyscf import scf
    # implement the iterative procedure to do g0w0
def g0w0(orbital_number, fock_matrix, real_corr_se):
    # Initial guess
    fock_element = fock_matrix[orbital_number, orbital_number]
    # the initial guess is the fog element
    initial_guess = fock_element
    # Initialize the qpe
    qpe = initial_guess
    iter = 0
    tol = 1e-9
    while True:
        # Update the self energy using the current guess as the frequency
        new_qpe = fock_element + real_corr_se(qpe)[orbital_number]

        # Check if the convergence criterion is met
        if np.abs(new_qpe - qpe) <= tol:
            break

        qpe = new_qpe
        iter += 1
        
    return qpe
molecule = setup_molecule()
my_fock = fock_matrix_hf(molecule)
# # find the number of orbitals
n_orbitals = molecule.nao_nr()
# # find the number of occupied orbitals
n_occupied = molecule.nelectron//2
# # find the number of virtual orbitals
n_virtual = n_orbitals - n_occupied
homo_index = n_occupied-1
lumo_index = n_occupied
conversion_factor = 27.2114
print(g0w0(homo_index, my_fock, real_corr_se) * conversion_factor)

# #Original PySCF GW (with frequency)

# mf = rks.RKS(molecule)
# mf.xc = 'hf'
# mf.verbose = 0
# mf.kernel()
# # IP and EA
# nocc = molecule.nelectron//2
# nmo = mf.mo_energy.size
# nvir = nmo - nocc

# orbs = [nocc-2, nocc-1, nocc, nocc+1]

# td = tddft.dTDA(mf)
# td.nstates = nocc*nvir
# e, xy = td.kernel()
# # Make a fake Y vector of zeros
# td_xy = list()
# for e,xy in zip(td.e,td.xy):
#     x,y = xy
#     td_xy.append((x,0*x))
# td.xy = td_xy

# mygw = gw.GW(mf, freq_int='exact', tdmf=td)
# mygw.kernel(orbs=orbs)
# print(mygw.mo_energy*conversion_factor)