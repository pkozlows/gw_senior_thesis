import math
small = 4e-10
bag = 15e-10
def volume_of_box(long):
    return long**3        
# convert molecule weight of one molecule of carbon monoxide in grams
molecular_weight = 28.01
molecules_per_mol = 6.022e23
molecular_weight_kilograms = (molecular_weight / molecules_per_mol) * 1e-3
print(molecular_weight_kilograms / volume_of_box(small))
print(molecular_weight_kilograms / volume_of_box(bag))
# calculations for the absorption energies
pristine = -.12089257e3
top = -.13809963e3
second_layer = -.13657701e3
long_bridge = -.13744195e3
for_bridge = -.13807358e3
# using the calculation results from carbon monoxide from 15 A, because this is more realistic to the actual density of air
carbon_monoxide = -.14767794e2
def change(before, fino):
    return fino - before
def reactants(carbonMonoxide, pristine):
    return carbonMonoxide + pristine
print(change(reactants(carbon_monoxide, pristine), top))
print(change(reactants(carbon_monoxide, pristine), second_layer))
print(change(reactants(carbon_monoxide, pristine), for_bridge))