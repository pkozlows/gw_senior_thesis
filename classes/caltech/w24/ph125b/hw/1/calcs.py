import sympy as sp

# Define symbols
m, Z, a0, alpha, hbar = sp.symbols('m Z a0 alpha hbar', real=True, positive=True)
r, rn = sp.symbols('r rn', real=True, positive=True)

# Define the hydrogen-like wave function
psi_hydrogen = sp.sqrt(Z**3 / (sp.pi * a0**3)) * sp.exp(-Z * r / a0)

# Define the trial wave function for N particles
N = sp.symbols('N', integer=True, positive=True)
n = sp.symbols('n', integer=True, positive=True)
psi_trial = sp.Product(psi_hydrogen.subs(r, rn), (n, 1, N))

# Define the momentum operator squared
p_squared = -hbar**2 * sp.diff(psi_trial, rn, rn)

# Kinetic energy operator
T = p_squared / (2 * m)

# Kinetic energy integral
kinetic_energy = sp.integrate(T * psi_trial**2, (rn, 0, sp.oo))

# Potential energy integral (Nuclear potential)
potential_energy = sp.integrate(alpha / rn * psi_trial**2, (rn, 0, sp.oo))

# Print the results
print("Kinetic Energy:", kinetic_energy)
print("Potential Energy:", potential_energy)