import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.optimize import curve_fit

def sparse_hamiltonian(L, h, J=1, periodic=False):
    row = []
    col = []
    data = []
    def binary_string(n, L):
        binary = bin(n)[2:]
        while len(binary) < L:
            binary = '0' + binary
        return binary
    # loop over the index i
    for i in range(2**L):
        # turn that index into a binary string
        bi = binary_string(i, L)

        # to the diagonal element first
        total_interaction = 0
        # initialize the loop range based on the boundary conditions
        loop_range = L if periodic else L - 1
        for k in range(loop_range):
            # Handle periodic boundary by wrapping the index
            next_k = (k + 1) % L if periodic else k + 1
            # Add neighbor interaction based on the spin values
            total_interaction += -J * (1 if bi[k] == bi[next_k] else -1)
        row.append(i)
        col.append(i)
        data.append(total_interaction)

        # now consider the off-diagonal elements connected by a single spin flip
        # make a function that takes the bit i and returns a list of basis states connected by a single flip
        def flip(i):
            connected_bases_states = []
            for j in range(L):
                flip_i = i ^ (1 << j)
                connected_bases_states.append(flip_i)
            return connected_bases_states
        
        # loop over the connected basis states
        for flip_i in flip(i):
            row.append(i)
            col.append(flip_i)
            data.append(-h)       

    # Create a sparse matrix from the lists
    H = scipy.sparse.coo_matrix((data, (row, col)), shape=(2**L, 2**L))
    return H
# This function should return a sparse matrix representing the Hamiltonian


def entanglement_entropy(rho):
    """Calculate the entanglement entropy of a reduced density matrix."""
    # Normalize rho
    rho /= np.trace(rho)
    # Eigenvalues of the reduced density matrix
    eigvals = np.linalg.eigh(rho)[0]
    # Filter out zero eigenvalues to avoid log(0)
    nonzero_eigvals = eigvals[eigvals > 1e-10]
    # Entanglement entropy calculation
    entropy = -np.sum(nonzero_eigvals * np.log(nonzero_eigvals))
    return entropy

def calculate_reduced_density_matrix(state, L, ell):
    """Calculate the reduced density matrix for segment A of length ell."""
    # Reshape the state vector into a matrix
    state_matrix = state.reshape(2**ell, 2**(L - ell))
    # Calculate the reduced density matrix
    rho_A = np.dot(state_matrix, state_matrix.conj().T)
    return rho_A


# L = [4, 6, 8]
# h = [0.3]

# entropies = {l: [] for l in L}
# # Initialize data collection
# S_L2_vs_L = {hi: [] for hi in h}  # This will hold entropy values for each h across all L


# for l in L:
#     for hi in h:
#         # Calculate the ground state using the sparse_hamiltonian
#         H = sparse_hamiltonian(l, hi, periodic=False)
#         eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H.astype(np.float64), k=1, which='LA')
#         ground_state = eigenvectors[:, 0]
        
#         for ell in range(1, l):
#             rho_A = calculate_reduced_density_matrix(ground_state, l, ell)
#             entropy = entanglement_entropy(rho_A)
#             entropies[l].append(entropy)
        
#         rho_A_L2 = calculate_reduced_density_matrix(ground_state, l, l//2)
#         entropy_L2 = entanglement_entropy(rho_A_L2)
#         S_L2_vs_L[hi].append((l, entropy_L2))  # Store the tuple (L, entropy)



# # Define line styles for different values of h and colors for L
# line_styles = ['-', '--', ':']
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend if needed
# markers = ['o', '^', 's']  # Example: circle, triangle, square

# if len(colors) < len(L):
#     raise ValueError("Not enough colors for the number of L values")

# # Plot S(ell; L) for 1 <= ell <= L-1 for various L and h
# plt.figure(figsize=(10, 6))
# for l_index, l in enumerate(L):
#     color = colors[l_index]  # Color for each L
#     for h_index, hi in enumerate(h):
#         start_index = h_index * (l - 1)
#         end_index = start_index + l - 1
#         line_style = line_styles[h_index % len(line_styles)]
#         plt.plot(range(1, l), entropies[l][start_index:end_index], label=f'L = {l}, h = {hi}',
#                  linestyle=line_style, color=color)

# plt.xlabel(r'Segment length $\ell$')
# plt.ylabel(r'Entanglement Entropy $S(\ell; L)$')
# plt.legend()
# plt.title(r'Entanglement Entropy $S(\ell; L)$ vs. Segment length $\ell$')
# plt.grid(True)
# plt.savefig('entanglement_entropy_LA.png')

# # Create the plot
# plt.figure(figsize=(10, 6))
# for h_index, hi in enumerate(h):
#     color = colors[h_index % len(colors)]
#     # Unpack L and entropy values for plotting
#     L_vals, entropies = zip(*S_L2_vs_L[hi])
#     plt.plot(L_vals, entropies, label=f'h = {hi}', color=color, linestyle='-')

# plt.xlabel(r'System size $L$')
# plt.ylabel(r'Entanglement Entropy $S(L/2; L)$')
# plt.title(r'Entanglement Entropy at $L/2$ vs. System size $L$')
# plt.legend()
# plt.grid(True)
# plt.savefig('entanglement_entropy_L2_LA.png')

# # convert the current entropis list to the format # entropies = {l: [] for l in L}
# entropies = {l: [] for l in L}
# for l in L:
    

# # # Constants
# # L = [4, 6, 8]
# # h = [1]  # Includes a critical value assumed to be 1 for illustration
# # entropies = {l: [] for l in L}
# # S_L2_vs_L = {hi: [] for hi in h}

# # for l in L:
# #     entropies[l] = []  # Initialize a list for each system size
# #     for hi in h:
# #         H = sparse_hamiltonian(l, hi, periodic=True)  # Corrected to periodic=True as needed
# #         eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H.astype(np.float64), k=1, which='SA')
# #         ground_state = eigenvectors[:, 0]

# #         # You might need to store entropy for each segment length here
# #         for ell in range(1, l):
# #             rho_A = calculate_reduced_density_matrix(ground_state, l, ell)
# #             entropy = entanglement_entropy(rho_A)
# #             entropies[l].append(entropy)  # Append entropy for each segment length

# #         rho_A_L2 = calculate_reduced_density_matrix(ground_state, l, l//2)
# #         entropy_L2 = entanglement_entropy(rho_A_L2)
# #         S_L2_vs_L[hi].append((l, entropy_L2))

# # # Define plot aesthetics
# # line_styles = ['-', '--', ':']
# # colors = ['b', 'g', 'r']  # One color for each h value

# # # Entanglement entropy S(ell; L) vs. Segment length ell
# # plt.figure(figsize=(10, 6))
# # for l_index, l in enumerate(L):
# #     for h_index, hi in enumerate(h):
# #         start_index = h_index * (l - 1)
# #         end_index = start_index + l - 1
# #         plt.plot(range(1, l), entropies[l][start_index:end_index], label=f'L = {l}, h = {hi}',
# #                  linestyle=line_styles[h_index % len(line_styles)], color=colors[h_index])

# # plt.xlabel(r'Segment length $\ell$')
# # plt.ylabel(r'Entanglement Entropy $S(\ell; L)$')
# # plt.legend()
# # plt.title(r'Periodic Entanglement Entropy $S(\ell; L)$ vs. Segment length $\ell$')
# # plt.grid(True)
# # plt.savefig('entanglement_entropy_periodic.png')

# # # Entanglement entropy S(L/2; L) vs. System size L for periodic boundaries
# # plt.figure(figsize=(10, 6))
# # for h_index, hi in enumerate(h):
# #     color = colors[h_index % len(colors)]
# #     L_vals, entropies = zip(*S_L2_vs_L[hi])
# #     plt.plot(L_vals, entropies, label=f'h = {hi}', color=color, linestyle='-')

# # plt.xlabel(r'System size $L$')
# # plt.ylabel(r'Entanglement Entropy $S(L/2; L)$')
# # plt.title(r'Periodic Entanglement Entropy at $L/2$ vs. System size $L$')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('entanglement_entropy_L2_periodic.png')

# # Correct the definition of the fit function
#     def fit_function(ell, c, C):
#         # Assuming max(L) is the largest value in your L array
#         largest_L = max(L)  
#         return (c / 3) * np.log((largest_L / np.pi) * np.sin(np.pi * ell / largest_L)) + C

# # Correct the extraction of critical data
# # This assumes that entropies[max(L)] contains the entropies for the largest system size at the critical point h=1
# critical_ell = np.array(range(1, max(L)))  # This should generate 8 values if max(L) is 8
# critical_ee = np.array(entropies[max(L)])  # Make sure this has the same number of values as critical_ell


# # # Fit the data to the equation
# # params, params_covariance = curve_fit(fit_function, critical_ell, critical_ee)

# # # Plot the fit results
# # plt.figure()
# # plt.plot(critical_ell, critical_ee, 'bo', label='Data')
# # plt.plot(critical_ell, fit_function(critical_ell, *params), 'r-', label=f'Fit: ${params[0]:.2f}/3 \log((L/\pi) \sin(\pi \ell / L)) + {params[1]:.2f}$')
# # plt.xlabel(r'Segment length $\ell$')
# # plt.ylabel(r'Entanglement Entropy $S(\ell; L)$')
# # plt.legend()
# # plt.title('Fit of Entanglement Entropy at Critical Point')
# # plt.grid(True)
# # plt.savefig('entanglement_entropy_fit_LA.png')

# # # Output the fit parameters
# # print(f"Fitted c: {params[0]}")
# # print(f"Fitted C: {params[1]}")

