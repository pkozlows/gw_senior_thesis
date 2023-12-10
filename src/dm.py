import pyscf
from pyscf.dft import rks
from pyscf.tdscf.rks import dTDA, dRPA
import numpy as np
from mf import setup_molecule, calculate_mean_field
import numpy as np

def lin_gw_dm(td, mf):
    '''Calculates the linearized GW self energy for a given molecule and frequency. Returns a matrix of correlation energies for each orbital at the given frequency. All elements are considered.'''
    e, xy = td.kernel()
    # Make a fake Y vector of zeros
    td_xy = list()
    for e,xy in zip(td.e,td.xy):
        x,y = xy
        td_xy.append((x,0*x))
    td.xy = td_xy
    # from mean field
    orbital_energies = mf.mo_energy
    n_orbitals = mf.mo_energy.shape[0]
    n_occupied = mf.mol.nelectron//2
    n_virtual = n_orbitals - n_occupied

    dm = np.zeros((n_orbitals, n_orbitals))

    # let's start with the occupied block
    # first add the delta function
    delta_matrix = np.eye(n_occupied)
    dm[:n_occupied, :n_occupied] += delta_matrix
    # prepare for einsum
    # convert the numerator of the excitation vectors into a singular object
    occ_num = np.einsum('ias,jas->ij', td.xy[:n_occupied, n_occupied:, :], td.xy[:n_occupied, n_occupied:, :])
    
    return

    print(m)
mol = setup_molecule()
mf, n_orbitals, n_occupied, n_virtual, orbital_energies = calculate_mean_field(mol, 'dft')

td = dTDA(mf)
td.nstates = n_occupied*n_virtual
lin_gw_dm(td, mf)