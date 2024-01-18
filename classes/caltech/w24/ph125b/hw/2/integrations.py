import sympy as sp
# Define the symbols
x, L, A1, A2, hbar, m = sp.symbols('x L A1 A2 hbar m')
# first normalization constant
fn1 = A1 * ((L/2)**2 - x**2)
first_integral = sp.integrate(fn1**2, (x, -L/2, L/2))
first_equation = sp.Eq(first_integral, 1)
first_A1 = sp.solve(first_equation, A1)
# second normalization constant
fn2 = A2 * x * ((L/2)**2 - x**2)
second_integral = sp.integrate(fn2**2, (x, -L/2, L/2))
second_equation = sp.Eq(second_integral, 1)
second_A2 = sp.solve(second_equation, A2)
# lope over all of the possibilities
for bra in [1, 2]:
    for ket in [1, 2]:
        # define the wave function
        if bra == 1:
            bra_function = fn1
        elif bra == 2:
            bra_function = fn2
        if ket == 1:
            ket_function = fn1
        elif ket == 2:
            ket_function = fn2
        # Define the Hamiltonian operator part: -hbar^2 / (2m) * d^2/dx^2
        Hamiltonian_operator = (-hbar**2 / (2*m)) * sp.diff(ket_function, x, 2)

        # Calculate the integral
        integral_result_Hamiltonian = sp.integrate(bra_function * Hamiltonian_operator, (x, -L/2, L/2))
        integral_result_Hamiltonian.simplify()
        # Print the result
        print('bra = {}, ket = {}:'.format(bra, ket))
        print(integral_result_Hamiltonian)
#  # calculation for 5 (eV)
# rydberg = 13.6 # eV
# # make a dictionary that contains the atomic number and number of electrons for li, be, and n
# numbers = {'li': [3, 3], 'be': [4, 4], 'n': [7, 7]}
# # my variational formula
# def variational_formula(Z, n):
#     value = n * (Z - (5/16)*(n-1))**2
#     return value * rydberg
# # loop over the dictionary
# for key in numbers:
#     # get the values
#     Z = numbers[key][0]
#     n = numbers[key][1]
#     full = variational_formula(Z, n)
#     ion = variational_formula(Z, n-1)
#     # front the differences
#     difference = full - ion
#     # print the resultst
#     print('For {}:'.format(key))
#     print('The full energy is {} eV'.format(full))
#     print('The ion energy is {} eV'.format(ion))
#     print('The difference is {} eV'.format(difference))

# # make a script to evalue the integral from 6a
# x_0, x_e, m, beta, E = sp.symbols('x_0 x_e m beta z E')
# gamma = 2*sp.integrate(sp.sqrt(2*m*((beta/z)-E)), (z, x_0, x_e))
# print(gamma)
