import pyscf
from pyscf.dft import rks

def setup_molecule(name):
    '''Sets up the molecule.'''
    if name == 'water':
        molecule = pyscf.M(
        atom = 'O  0 0 0; H  0 0.758602 0.504284; H  0 0.758602 -0.504284',
        basis = 'ccpvdz',
        symmetry = True,
    )
    elif name == 'h2':
        molecule = pyscf.M(
        atom = 'H  0 0 0; H  0 0 0.735',
        basis = 'ccpvdz',
        symmetry = True,
    )
    elif name == 'methane':
        molecule = pyscf.M(
        atom = 'C  0 0 0; H  0 0 1.08; H  0 1.02 -0.36; H  -0.88 -0.51 -0.36; H  0.88 -0.51 -0.36',
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



