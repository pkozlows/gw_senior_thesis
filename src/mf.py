import pyscf
from pyscf.dft import rks

def setup_molecule():
    '''Sets up the water molecule.'''
    molecule = pyscf.M(
    atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
    basis = 'ccpvdz',
    symmetry = True,
)
    return molecule

def calculate_mean_field(molecule, method):
    '''Calculates the mean field for a given molecule and method.'''

    # I want to get basic information like the number of orbitals, occupied orbitals, and virtual orbitals
    n_orbitals = molecule.nao_nr()
    n_occupied = molecule.nelectron//2
    n_virtual = n_orbitals - n_occupied

    # run the mean field calculation
    if method == 'dft':
        mf = rks.RKS(molecule)
        mf.xc = 'pbe'
        mf.verbose = 0
        mf.kernel()
    elif method == 'hf':
        mf = pyscf.scf.RHF(molecule)
        mf.xc = 'hf'
        mf.verbose = 0
        mf.kernel()
    # get the orbital energies
    orbital_energies = mf.mo_energy
            
    return mf, n_orbitals, n_occupied, n_virtual, orbital_energies

