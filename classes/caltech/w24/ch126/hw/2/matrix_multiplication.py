from sympy import Matrix, I, sqrt, symbols
hbar = symbols('hbar')
j_x = (hbar*sqrt(2)/2)*Matrix([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]])

j_y = (hbar*sqrt(2)/(2*I))*Matrix([[0, 1, 0],
                    [-1, 0, 1],
                    [0, -1, 0]])

j_z = hbar*Matrix([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, -1]])

# evaluate the commutators
print("Commutator of Jx and Jy:")
print((j_x*j_y - j_y*j_x).doit())
print("Commutator of Jy and Jz:")
print((j_y*j_z - j_z*j_y).doit())
print("Commutator of Jz and Jx:")
print((j_z*j_x - j_x*j_z).doit())

