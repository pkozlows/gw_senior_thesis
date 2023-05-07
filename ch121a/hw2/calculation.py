import math

def conversion(e):
    hartree = 627.51
    return e * hartree

reluctant = -270.082447
product = -270.113477
ts = -270.044434

def change(before, fino):
    return fino - before

change_from_transition = conversion(change(reluctant, ts))
print(change_from_transition)

boltzmann_constant = 1.38064852e-23
plank_constant = 6.62607004e-34
temperature = 298.15
gas_constant = 1.987e-3
equation = (boltzmann_constant * temperature) / (plank_constant) * math.exp(-change_from_transition / (gas_constant * temperature))
print(-change_from_transition / (gas_constant * temperature))
print(equation)


# I DNS on
second_regarded = -1.669886
second_transition_stated = -1.662442
# print(conversion(change(second_regarded, second_transition_stated)))