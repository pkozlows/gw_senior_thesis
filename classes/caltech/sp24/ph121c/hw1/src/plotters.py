import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from hw1 import sparse_hamiltonian


# Representative values of h
h_values = [0.3, 1.7]
# Range of L values to study
L_range = range(8, 16, 2)  # Example: from 8 to 16, in steps of 2

# Initialize storage for energies
energies = {'open': {}, 'periodic': {}}

# Define a consistent interval for y-axis ticks
tick_interval = 0.01  # Adjust this based on the expected range of energy values

for h in h_values:
    energies['open'][h] = []
    energies['periodic'][h] = []

    for bc in ['open', 'periodic']:
        for L in L_range:
            H = sparse_hamiltonian(L, h, periodic=(bc == 'periodic')).asformat('csr')
            energy_per_site = scipy.sparse.linalg.eigsh(H, k=1, which='SA', return_eigenvectors=False)[0] / L
            energies[bc][h].append(energy_per_site)

    # Plotting after collecting all data for the current h
    plt.figure(figsize=(14, 7))
    plt.plot(list(L_range), energies['open'][h], 'o-', label=f'Open, h={h}')
    plt.plot(list(L_range), energies['periodic'][h], 's--', label=f'Periodic, h={h}')

    plt.xlabel('System Size L')
    plt.ylabel('Ground State Energy per Site $E_0 / L$')
    plt.title(f'Convergence with System Size for h = {h}')
    plt.legend()
    plt.grid(True)

    # Determine the min and max energies for this plot to set y-limits appropriately
    min_energy = min(energies['open'][h] + energies['periodic'][h])
    max_energy = max(energies['open'][h] + energies['periodic'][h])

    # Set y-axis limits based on the smallest and largest energies, adjusted to the nearest tick
    plt.ylim((min_energy // tick_interval * tick_interval, 
              (max_energy // tick_interval + 1) * tick_interval))
    plt.yticks(np.arange(min_energy // tick_interval * tick_interval, 
                         (max_energy // tick_interval + 1) * tick_interval, 
                         tick_interval))

    plt.savefig(f'convergence_h{h}.png')
h_values = [0.3, 1.7]
L_range = range(8, 20, 2)  # From L=8 to L=18, in steps of 2

# Dictionary to store energies for each h value
energies_per_h = {h: [] for h in h_values}

for L in L_range:
    for h in h_values:
        H_open = sparse_hamiltonian(L, h, periodic=False).asformat('csr')
        E_open = scipy.sparse.linalg.eigsh(H_open, k=1, which='SA', return_eigenvectors=False)[0]
        energies_per_h[h].append((L, E_open))

# Now, plot the energy differences for each h value
for h in h_values:
    plt.figure(figsize=(10, 6))
    plt.title(f'Bulk Energy Density per Site for h = {h}')
    Ls, Es = zip(*energies_per_h[h])  # Unpack the energies and L values
    # Calculate the differences and plot
    energy_differences = [(Es[i] - Es[i-1]) / 2 for i in range(1, len(Es))]
    plt.plot(Ls[1:], energy_differences, 'o-', label=f'h = {h}')

    plt.xlabel('System Size L')
    plt.ylabel('Bulk Energy Density per Site')
    plt.grid(True)
    plt.savefig(f'bulk_energy_density_h{h}.png')

# 4.4
h = np.linspace(1, 1.2, 10)
L = 16
# Lists to store excitation energies
first_excitation_energies = []
second_excitation_energies = []
third_excitation_energies = []
fourth_excitation_energies = []

plt.figure()

for h_value in h:
    H = sparse_hamiltonian(L, h_value, periodic=True).asformat('csr')
    es = scipy.sparse.linalg.eigsh(H, k=5, which='SA', return_eigenvectors=False)
    # Compute the excitation energy gaps
    first_excitation_energy = es[1] - es[0]
    second_excitation_energy = es[2] - es[0]
    third_excitation_energy = es[3] - es[0]
    fourth_excitation_energy = es[4] - es[0]
    # Append energies to their respective lists
    first_excitation_energies.append(first_excitation_energy)
    second_excitation_energies.append(second_excitation_energy)
    third_excitation_energies.append(third_excitation_energy)
    fourth_excitation_energies.append(fourth_excitation_energy)

# Plot the excitation energies after collecting all data
plt.plot(h, first_excitation_energies, 'o-', label='First Excitation Energy')
plt.plot(h, second_excitation_energies, 's-', label='Second Excitation Energy')
plt.plot(h, third_excitation_energies, '^-', label='Third Excitation Energy')
plt.plot(h, fourth_excitation_energies, 'v-', label='Fourth Excitation Energy')
plt.xlabel('Transverse Field Strength h')
plt.ylabel('Excitation Energy')
plt.legend()

plt.savefig('excitation_energies.png')










# for l in L:
#     plt.figure()
#     plt.title(f'Sparse L={l}')
#     open_ground_state_energies = []  # To store ground state energies for different h
#     periodic_ground_state_energies = []  # To store ground state energies for different h
#     for hi in h_range:
#         # for the given hi, comupte gs energy for both periodic and non-periodic
#         H = hw1.sparse_hamiltonian(l, hi)
#         H_periodic = hw1.sparse_hamiltonian(l, hi, periodic=True)
#         # Compute the ground state energy using the lowest eigenvalue
#         open_ground_state_energies.append(min(scipy.sparse.linalg.eigsh(H, k=1, which='SA')[0]))
#         periodic_ground_state_energies.append(min(scipy.sparse.linalg.eigsh(H_periodic, k=1, which='SA')[0]))
#     # plot the ground state energy as a function of h for both periodic and non-periodic
#     plt.plot(h_range, open_ground_state_energies, label='Non-periodic')
#     plt.plot(h_range, periodic_ground_state_energies, label='Periodic')
#     plt.xlabel('h')
#     plt.ylabel('Ground state energy')
#     plt.legend()
#     # save a tout for the given value of L
#     plt.savefig(f'sparse_ising_model_L{l}.png')

# for l in L:
#     plt.figure()
#     plt.title(f'Dense L={l}')
#     open_ground_state_energies = []  # To store ground state energies for different h
#     periodic_ground_state_energies = []  # To store ground state energies for different h
#     for hi in h_range:
#         # for the given hi, comupte gs energy for both periodic and non-periodic
#         H = open_dense_hamiltonian_explicit(l, hi)
#         print(f'H: hi={hi}\n{H}')
#         H_periodic = periodic_dense_hamiltonian_explicit(l, hi)
#         # Compute the ground state energy using the lowest eigenvalue
#         open_ground_state_energies.append(min(np.linalg.eigvalsh(H)))
#         periodic_ground_state_energies.append(min(np.linalg.eigvalsh(H_periodic)))
#     # plot the ground state energy as a function of h for both periodic and non-periodic
#     plt.plot(h_range, open_ground_state_energies, label='Non-periodic')
#     plt.plot(h_range, periodic_ground_state_energies, label='Periodic')
#     plt.xlabel('h')
#     plt.ylabel('Ground state energy')
#     plt.legend()
#     # save a tout for the given value of L
#     plt.savefig(f'dense_ising_model_L{l}.png')

for l in L:

    plt.figure(figsize=(12, 8))
    plt.title(f'System Size L={l}')
    
    # Initialize storage for energies
    energies = {'dense': {'open': ([], []), 'periodic': ([], [])},
                'sparse': {'open': ([], []), 'periodic': ([], [])}}

    for hi in h_range:
        # Dense Hamiltonian calculations
        for bc in ['open', 'periodic']:
            H_func = open_dense_hamiltonian_explicit if bc == 'open' else periodic_dense_hamiltonian_explicit
            H = H_func(l, hi)
            eigvals = np.linalg.eigvalsh(H)
            energies['dense'][bc][0].append(eigvals[0])  # Ground state energy
            energies['dense'][bc][1].append(eigvals[1])  # First excited state energy

        # Sparse Hamiltonian calculations
        for bc in ['open', 'periodic']:
            periodic_flag = True if bc == 'periodic' else False
            H_sparse = hw1.sparse_hamiltonian(l, hi, periodic=periodic_flag).asformat('csr')
            sparse_eigvals = scipy.sparse.linalg.eigsh(H_sparse, k=2, which='SA', return_eigenvectors=False)
            energies['sparse'][bc][0].append(sparse_eigvals[0])  # Ground state energy
            energies['sparse'][bc][1].append(sparse_eigvals[1])  # First excited state energy

    # Plot configurations
    configs = [('dense', 'open', 'o', '-'), ('dense', 'periodic', 's', '-'),
               ('sparse', 'open', 'o', '--'), ('sparse', 'periodic', 's', '--')]
    colors = ['blue', 'red']  # Ground state and first excited state

    for method, bc, marker, linestyle in configs:
        ground_states = energies[method][bc][0]
        excited_states = energies[method][bc][1]
        label_base = f"{method.capitalize()} {bc.capitalize()}"

        plt.plot(h_range, ground_states, marker=marker, linestyle=linestyle, color=colors[0],
                 label=f'{label_base} Ground State')
        plt.plot(h_range, excited_states, marker=marker, linestyle=linestyle, color=colors[1],
                 label=f'{label_base} First Excited', alpha=0.7)

    plt.xlabel('Transverse Field Strength h')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig(f'enhanced_compare_ising_model_L{l}.png')

    L = 12  # System size
h_values = np.linspace(0.6, 1.3, 10)  # Range of h values to scan
delta_h = 0.01 # Small increment in h

fidelities = []  # To store fidelity values

for i, h in enumerate(h_values[:-1]):  # Skip the last value since we look at h and h + delta_h
    H1 = sparse_hamiltonian(L, h, periodic=True)
    H2 = sparse_hamiltonian(L, h + delta_h, periodic=True)

    def calculate_ground_state(H):
        """Calculate the ground state of a Hamiltonian."""
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
        return eigenvectors[:, 0]
    
    psi_gs_h = calculate_ground_state(H1)
    psi_gs_h_dh = calculate_ground_state(H2)
    
    # Calculate fidelity and add to the list
    fidelity = np.abs(np.dot(psi_gs_h.conj().T, psi_gs_h_dh))
    fidelities.append(fidelity)

plt.figure(figsize=(10, 6))
plt.plot(h_values[:-1], fidelities, label='Fidelity', marker='o', linestyle='-', markersize=4)
plt.xlabel('Transverse Field Strength $h$')
plt.ylabel('Fidelity')
plt.title('Ground State Fidelity Across Transverse Field Strength')
plt.grid(True)
plt.legend()
plt.savefig('fidelity_vs_h.png')
# # 4.2
# # Define the range of h values
# h_range = np.linspace(0, 2, 4)
# # Define the range of L values
# L = [16, 18, 20, 22]
# # just plot the ground state energy for now
# for l in L:
#     plt.figure()
#     plt.title(f'System Size L={l}')
#     gs_energies = []  # To store ground state energies for different h
#     for hi in h_range:
#         # for the given hi, comupte gs energy
#         H = sparse_hamiltonian(l, hi, periodic=False).asformat('csr')
#         # Compute the ground state energy using the lowest eigenvalue
#         gs_energies.append(min(scipy.sparse.linalg.eigsh(H, k=1, which='SA')[0]))
#     # plot the ground state energy as a function of h
#     plt.plot(h_range, gs_energies)
#     plt.xlabel('h')
#     plt.ylabel('Ground state energy')
#     # save a tout for the given value of L
#     plt.savefig(f'sparse_ising_model_L{l}.png')
