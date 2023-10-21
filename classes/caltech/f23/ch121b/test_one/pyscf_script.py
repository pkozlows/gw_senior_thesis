from pyscf import gto, scf

# Define the molecule
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74'  # Hydrogen molecule with a bond length of 0.74 angstroms
mol.basis = 'sto-3g'             # Basis set
mol.build()

# Perform a Hartree-Fock (HF) calculation
mf = scf.RHF(mol)
mf.verbose = 0  # Set verbosity to minimal (0)
mf.scf()

# Get the HF energy
energy = mf.e_tot

print("Hartree-Fock Energy: {:.6f} Hartrees".format(energy))
