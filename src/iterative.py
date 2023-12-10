import numpy as np
from tda import real_corr_se
from mf import setup_molecule, calculate_mean_field
from fock import fock_matrix_hf, fock_dft
from pyscf import gw
from pyscf import tddft
import pyscf
from pyscf.dft import rks
from pyscf import scf

molecule = setup_molecule()
conversion_factor = 27.2114
# #Original PySCF GW (with frequency)

mf = rks.RKS(molecule)
mf.xc = 'hf'
mf.verbose = 0
mf.kernel()
# IP and EA
nocc = molecule.nelectron//2
nmo = mf.mo_energy.size
nvir = nmo - nocc

orbs = [nocc-1, nocc]

td = tddft.dRPA(mf)
td.nstates = nocc*nvir
e, xy = td.kernel()
# Make a fake Y vector of zeros
# td_xy = list()
# for xy in td.xy:
#     x,y = xy
#     td_xy.append((x,y))
# td.xy = td_xy

mygw = gw.GW(mf, freq_int='exact', tdmf=td)
# mygw.kernel(orbs=orbs)
# print(mygw.mo_energy*conversion_factor)

    # implement the iterative procedure to do g0w0
def g0w0(orbital_number, fock_mo, real_corr_se, mf):
    '''Calculates the G0W0 correction for a given orbital number, fock matrix in the atomic orbital basis, and real part of the correlation self energy.'''
    # convert the fock matrix to the molecular orbital basis

    # Initial guess
    fock_element = fock_mo[orbital_number, orbital_number]
    # the initial guess is the fog element
    initial_guess = fock_element
    # Initialize the qpe
    qpe = initial_guess
    iter = 0
    tol = 1e-7
    while True:
        # Update the self energy using the current guess as the frequency
        new_qpe = fock_element + real_corr_se(qpe, mf)[orbital_number]

        # Check if the convergence criterion is met
        if np.abs(new_qpe - qpe) <= tol:
            break

        qpe = new_qpe
        iter += 1
        
    return qpe
mf = rks.RKS(molecule)
mf.xc = 'hf'
mf.verbose = 0
mf.kernel()

my_fock = fock_matrix_hf(molecule)
# my_fock = fock_dft(molecule)
# find the number of orbitals
n_orbitals = molecule.nao_nr()
# # find the number of occupied orbitals
n_occupied = molecule.nelectron//2
# # find the number of virtual orbitals
n_virtual = n_orbitals - n_occupied
homo_index = n_occupied-1
lumo_index = n_occupied
print(g0w0(homo_index, my_fock, real_corr_se, mf) * conversion_factor)
