import math
molecules_in_system = 1539
molecules_per_mol = 6.022e23
molecules_in_system_mols = molecules_in_system / molecules_per_mol
v4 = 4000
v5 = 5000
v7 = 7000
v9 = 9000
v11 = 11000
v13 = 13000
def convert_ms_to_kjmol(v):
    wait_of_water_molecules = 18.01
    weight_water_per_mol = wait_of_water_molecules / molecules_per_mol
    ke = 0.5 * wait_of_water_molecules * v**2
    return ke / 1000000
print(convert_ms_to_kjmol(v4))
print(convert_ms_to_kjmol(v5))
print(convert_ms_to_kjmol(v7))
print(convert_ms_to_kjmol(v9))
print(convert_ms_to_kjmol(v11))
print(convert_ms_to_kjmol(v13))