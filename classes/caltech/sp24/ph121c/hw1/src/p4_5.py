import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from hw1.src.hw1 import sparse_hamiltonian, binary_string

def calculate_ground_state(H):
    """Calculate the ground state of a Hamiltonian."""
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return eigenvectors[:, 0], eigenvalues[0]

def correlation_function(psi_gs, L):
    """Calculate the correlation function of a state."""
    correlations = np.zeros(L)
    for i in range(2**L):
        # Convert the state index to binary
        binary_state = format(i, '0{}b'.format(L))
        for r in range(L):
            # Get the spins at site 1 and site 1+r from the binary representation
            spin1 = 1 if binary_state[0] == '1' else -1  # First site spin
            spin2 = 1 if binary_state[r % L] == '1' else -1  # (1+r)th site spin considering periodic boundary
            # Add the product of the spins to the correlation value, weighted by the probability of this basis state
            correlations[r] += spin1 * spin2 * np.abs(psi_gs[i])**2

    # The correlations array now contains the expectation value of the product of spins at positions 1 and 1+r for all r
    return correlations

def expectation_value_M_squared(psi_gs, L):
    """
    Calculate the expectation value of (M / L)^2 for a ground state.
    """
    # Calculate the correlation function
    correlations = correlation_function(psi_gs, L)
    
    # Compute the expectation value <M^2>
    # Under periodic boundary conditions, <M^2> simplifies to L times the sum of all correlations
    M_squared = L * np.sum(correlations)
    return M_squared

L_values = [8, 10, 12]
# Choose some representative values for h
h_values = [0.3, 1, 1.7]
ground_states = {}  # Dictionary to store ground states and energies for each (h, L) pair

# Calculate ground states and correlation functions for various L and h
for L in L_values:
    ground_states[L] = {}  # Initialize a sub-dictionary for each L
    for h in h_values:
        H = sparse_hamiltonian(L, h, periodic=True).tocsr()  # Get CSR format
        H = H.astype(np.float64)  # Ensure the matrix is of type float64
        
        psi_gs, E_gs = calculate_ground_state(H)
        
        # Compute the correlation function for the given ground state
        correlations = correlation_function(psi_gs, L)
        
        # Store psi_gs, E_gs, and correlations as a tuple
        ground_states[L][h] = (psi_gs, E_gs, correlations)

# Now ground_states is a nested dictionary with structure {L: {h: (psi_gs, E_gs, correlations)}}
for L in L_values:
    plt.figure(figsize=(10, 6))
    for h in h_values:
        correlations = ground_states[L][h][2]  # Retrieve correlations from the stored tuple
        r_values = range(L)  # Site separations
        plt.plot(r_values, correlations, 'o-', label=f'h={h}')
    
    plt.title(f'Correlation Function for L={L}')
    plt.xlabel('Site Separation r')
    plt.ylabel('Correlation $C^{zz}(r)$')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'4-5_correlation_L{L}.png')

# Plot setup
plt.figure(figsize=(10, 6))

# Loop over each L and plot the expectation values for all h
for L in L_values:
    M_squared_per_L = []  # To store the calculated <(M/L)^2> for each h
    for h in h_values:
        psi_gs, E_gs, _ = ground_states[L][h]
        M_squared = expectation_value_M_squared(psi_gs, L)
        M_squared_per_L.append(M_squared / L**2)
    
    # Plot for the current L
    plt.plot(h_values, M_squared_per_L, 'o-', label=f'L={L}')

# Enhance the plot
plt.title('Expectation Value of $(M / L)^2$ Across Different System Sizes')
plt.xlabel('Transverse Field Strength $h$')
plt.ylabel('Expectation Value $(M / L)^2$')
plt.legend()
plt.grid(True)
plt.savefig('4-5_M_squared.png')
    
        



