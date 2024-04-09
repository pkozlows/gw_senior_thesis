import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import hw1.src.hw1 as hw1
# Representative values of h
h_values = [0.3, 1.7]
# Range of L values to study
L_range = range(8, 16, 2)  # Example: from 8 to 20, in steps of 2

# Initialize storage for energies
energies = {'open': {}, 'periodic': {}}
for h in h_values:
    energies['open'][h] = []
    energies['periodic'][h] = []
    for L in L_range:
        # Calculate for open boundary conditions
        H_open = hw1.sparse_hamiltonian(L, h, periodic=False).asformat('csr')
        E_open = scipy.sparse.linalg.eigsh(H_open, k=1, which='SA', return_eigenvectors=False)[0]
        energies['open'][h].append(E_open / L)
        
        # Calculate for periodic boundary conditions
        H_periodic = hw1.sparse_hamiltonian(L, h, periodic=True).asformat('csr')
        E_periodic = scipy.sparse.linalg.eigsh(H_periodic, k=1, which='SA', return_eigenvectors=False)[0]
        energies['periodic'][h].append(E_periodic / L)
plt.figure(figsize=(14, 7))
for h in h_values:
    plt.plot(L_range, energies['open'][h], 'o-', label=f'Open, h={h}')
    plt.plot(L_range, energies['periodic'][h], 's--', label=f'Periodic, h={h}')

plt.xlabel('System Size L')
plt.ylabel('Ground State Energy per Site $E_{gs}(L) / L$')
plt.title('Convergence with System Size')
plt.legend()
plt.grid(True)
plt.savefig('convergence.png')







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