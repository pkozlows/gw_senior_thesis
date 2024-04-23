import pyscf
from pyscf import gto  # Importing gto module for molecule creation
from pyscf.dft import rks  # Importing rks module for mean field calculation

def setup_molecule(name):
    '''Sets up the molecule.'''
    if name == 'water':
        molecule = gto.Mole()  # Creating a Mole object
        molecule.build(
            atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
            basis = 'ccpvdz',
            symmetry = True,
        )
    elif name == 'h2':
        molecule = gto.Mole()  # Creating a Mole object
        molecule.build(
            atom = 'H  0 0 0; H  0 0 0.735',
            basis = 'ccpvdz',
            symmetry = True,
        )
    elif name == 'methane':
        molecule = gto.Mole()  # Creating a Mole object
        molecule.build(
            atom = 'C  0 0 0; H  0 0 1.08; H  0 1.02 -0.36; H  -0.88 -0.51 -0.36; H  0.88 -0.51 -0.36',
            basis = 'ccpvdz',
            symmetry = True,
        )
    elif name == 'hcl':
        molecule = gto.Mole()
        molecule.build(
            atom = 'H  0 0 0; Cl  0 0 1.27',
            basis = 'ccpvdz',
            symmetry = True,
        )
    elif name == 'nh3':
        molecule = gto.Mole()
        molecule.build(
            atom = 'N  0 0 0; H  0 0 1.02; H  0.885 0 -0.34; H  -0.885 0 -0.34',
            basis = 'ccpvdz',
            symmetry = True,
        )
    elif name == 'lih':
        molecule = gto.Mole()
        molecule.build(
            atom = 'Li  0 0 0; H  0 0 1.6',
            basis = 'ccpvdz',
            symmetry = True,
        )
    elif name == 'co':
        molecule = gto.Mole()
        molecule.build(
            atom = 'C  0 0 0; O  0 0 1.1',
            basis = 'ccpvdz',
            symmetry = True,
        )
    return molecule


def calculate_mean_field(molecule, method):
    '''Calculates the mean field for a given molecule and method.'''

    # run the mean field calculation
    if method == 'dft':
        mf = rks.RKS(molecule)
        mf.xc = 'pbe'
        mf.verbose = 0
        mf.kernel()
    elif method == 'hf':
        mf = rks.RKS(molecule)
        mf.xc = 'hf'
        mf.verbose = 0
        mf.kernel()
            
    return mf



