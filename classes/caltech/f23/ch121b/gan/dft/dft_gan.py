import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.bandstructure import HighSymmKpath 
from pyscf.pbc import gto, scf, dft
# 1. Obtain the GaN structure from MP-804
API_KEY = "O40Y6VhzaK0YlL03wiN6kOTaQ1HJTLkP"


def structure_to_pyscf_cell(structure: Structure):
    cell = gto.Cell()
    data = zip(structure.species, structure.cart_coords)
    # for species, coords in data:
    #     print(Element(species).Z)
    cell.atom = [(Element(species).Z, coords) for species, coords in data]
    cell.a = structure.lattice.matrix
    cell.unit = 'Bohr'  # Pymatgen uses Angstrom by default, but PySCF often uses Bohr; adjust as needed
    cell.build()
    return cell

# Using the function:
with MPRester(API_KEY) as m:
    structure = m.get_structure_by_material_id("mp-804")

cell = structure_to_pyscf_cell(structure)
cell.basis = {'N': '6-311++G**', 'Ga': '6-31G'} 


# 2. Set Up DFT Calculation in PySCF
kmf = scf.KRKS(cell, kpts=cell.make_kpts([3, 3, 3]))  # 3x3x3 k-points grid as an example
kmf.xc = 'b3pw91'
kmf.kernel()

# 3. Extract data after DFT calculation and save to disk
eigenvalues = kmf.mo_energy
coefficients = kmf.mo_coeff

np.save('eigenvalues.npy', eigenvalues)
# np.save('coefficients.npy', coefficients)

