import scipy.sparse.linalg
import matplotlib.pyplot as plt
import numpy as np
import os
from hw1 import sparse_hamiltonian, binary_string


import scipy.sparse as sp

def create_sparse_hamiltonian_x_basis(L, J, h):

  odd_sector = {}
  even_sector = {}

  dim = 2 ** L
  odd_ind = 0
  even_ind = 0

  for i in range(dim):
    if bin(i).count('1') % 2 != 0:
      odd_sector[i] = odd_ind
      odd_ind += 1
    else:
      even_sector[i] = even_ind
      even_ind += 1
  print('here 1')

  coo_odd = []
  coo_even = []

  for i, alpha in enumerate(odd_sector.keys()):

    sigma_x_term = 0
    for k in range(L):
      sigma_x_term += (2 * int((alpha >> k) & 1) - 1)

    coo_odd.append([i, i, -h * sigma_x_term])

    x_terms = []
    for k in range(L):
      ind1 = odd_sector[alpha]
      ind2 = odd_sector[(alpha ^ (1 << k)) ^ (1 << (k+1)%L)]
      x_terms.append([ind1, ind2, -J])

    coo_odd += x_terms

    #print(coo_odd)

  print('here 2')

  for i, alpha in enumerate(even_sector.keys()):

    sigma_x_term = 0
    for k in range(L):
      sigma_x_term += (2 * int((alpha >> k) & 1) - 1)

    coo_even.append([i, i, -h * sigma_x_term])

    x_terms = []
    for k in range(L):
      ind1 = even_sector[alpha]
      ind2 = even_sector[(alpha ^ (1 << k)) ^ (1 << (k+1)%L)]
      x_terms.append([ind1, ind2, -J])

    coo_even += x_terms

  print('here 3')

  row_even, col_even, data_even = zip(*coo_even)
  H_even = scipy.sparse.coo_matrix((data_even, (row_even, col_even)), shape=(2**(L-1), 2**(L-1)))

  row_odd, col_odd, data_odd = zip(*coo_odd)
  H_odd = scipy.sparse.coo_matrix((data_odd, (row_odd, col_odd)), shape=(2**(L-1), 2**(L-1)))

  return H_even, H_odd

def sparse_hamiltonian_x_basis(L, h, J=1, periodic=False):
    size = 2**L
    index_even = []  # To track indices of even parity states
    index_odd = []   # To track indices of odd parity states

    # Initialize lists for even and odd parity sectors
    row_even, col_even, data_even = [], [], []
    row_odd, col_odd, data_odd = [], [], []

    def count_ones(n):
        """Helper function to count '1's in binary representation."""
        return bin(n).count('1')
    # Populate even and odd indices
    for i in range(size):
        if count_ones(i) % 2 == 0:
            index_even.append(i)
        else:
            index_odd.append(i)

    # Map original indices to new compacted indices
    map_even = {idx: n for n, idx in enumerate(index_even)}
    map_odd = {idx: n for n, idx in enumerate(index_odd)}

    # Construct the Hamiltonian for each state
    for i in range(size):
        x_basis_state = binary_string(i, L)
        parity = count_ones(i) % 2  # Calculate parity of the state

        # Select the correct lists and mapping based on the parity
        row = row_even if parity == 0 else row_odd
        col = col_even if parity == 0 else col_odd
        data = data_even if parity == 0 else data_odd
        mapping = map_even if parity == 0 else map_odd

        # Diagonal contributions from σ^x (magnetic field)
        row.append(mapping[i])
        col.append(mapping[i])
        data.append(-h * (x_basis_state.count('1') - x_basis_state.count('0')))

        # Off-diagonal contributions from σ^z σ^z interaction
        loop_range = L if periodic else L - 1
        for j in range(loop_range):
            flipped_index = i ^ (1 << j) ^ (1 << ((j + 1) % L))
            if flipped_index in mapping:  # Check if flipped index is in the same parity
                row.append(mapping[i])
                col.append(mapping[flipped_index])
                data.append(-J)

    # Create sparse matrices for each parity sector
    H_even = sp.coo_matrix((data_even, (row_even, col_even)), shape=(len(index_even), len(index_even)), dtype=float).tocsr()
    H_odd = sp.coo_matrix((data_odd, (row_odd, col_odd)), shape=(len(index_odd), len(index_odd)), dtype=float).tocsr()

    return H_even, H_odd

L =[8, 10, 12, 14, 16]
h = 0.3
plt.figure(figsize=(10, 6))
plt.title(f'Dependence of ground state splitting on L for h={h} in ordered phase')
energies = []
excitation_energy = []
for L_val in L:
    H_even, H_odd = create_sparse_hamiltonian_x_basis(L_val, 1, h)
    eigvals_even, _ = scipy.sparse.linalg.eigsh(H_even, k=2, which='SA')
    eigvals_odd, _ = scipy.sparse.linalg.eigsh(H_odd, k=2, which='SA')
    # append all of the energies to a list
    energies.append([eigvals_even[0], eigvals_odd[0]])
    # energies.append([eigvals_even[1], eigvals_odd[1]])
    energies.sort()
    # determine the excitation energy
    excitation_energy.append(energies[0][1] - energies[0][0])

# fought the excitation and energy for each the value of L
plt.plot(L, excitation_energy, label=f'h={h}')
plt.xlabel('L')
plt.ylabel('Energy splitting')
plt.legend()
plt.grid(True)
plt.savefig('4-6_L_dependence.png')




    
L = 21
h_vals = [0.3, 1, 1.7]  # This should probably be h_vals to avoid confusion with h in plt.plot
energies_ground_even = []
energies_ground_odd = []
energies_first_excited_even = []
energies_first_excited_odd = []
energies_second_excited_even = []
energies_second_excited_odd = []

plt.figure(figsize=(10, 6))
plt.title(f'Sparse Hamiltonian in x-basis for L={L}')
for h_val in h_vals:
    H_even, H_odd = sparse_hamiltonian_x_basis(L, h_val, periodic=True)
    eigvals_even, _ = scipy.sparse.linalg.eigsh(H_even, k=3, which='SA')
    eigvals_odd, _ = scipy.sparse.linalg.eigsh(H_odd, k=3, which='SA')
    
    energies_ground_even.append(eigvals_even[0])
    energies_ground_odd.append(eigvals_odd[0])

    energies_first_excited_even.append(eigvals_even[1])
    energies_first_excited_odd.append(eigvals_odd[1])

    energies_second_excited_even.append(eigvals_even[2])
    energies_second_excited_odd.append(eigvals_odd[2])

# Now plot using the accumulated lists
plt.plot(h_vals, energies_ground_even, label='Ground state even')
plt.plot(h_vals, energies_ground_odd, label='Ground state odd')
plt.plot(h_vals, energies_first_excited_even, label='First excited state even')
plt.plot(h_vals, energies_first_excited_odd, label='First excited state odd')
plt.plot(h_vals, energies_second_excited_even, label='Second excited state even')
plt.plot(h_vals, energies_second_excited_odd, label='Second excited state odd')

plt.xlabel('h')
plt.ylabel('Energy')
plt.legend()
plt.grid(True)
plt.savefig('4-6_parity_resolved_energies.png')

# Example usage:
L = [8, 10, 12]
h_values = np.linspace(0, 2.0, 20)  # Range of h values to scan

for L_val in L:
    plt.figure(figsize=(10, 6))
    plt.title(f'Ground State Energies for L={L_val}')
    
    # Lists to store the lowest three unique energies for each h value
    lowest_three_parity = []
    lowest_three_z_basis = []

    for h_val in h_values:
        H_even, H_odd = sparse_hamiltonian_x_basis(L_val, h_val, periodic=True)
        H_z = sparse_hamiltonian(L_val, h_val, periodic=True)

        # Diagonalize and collect the lowest three energies for even sector
        eigvals_even = scipy.sparse.linalg.eigsh(H_even, k=3, which='SA', return_eigenvectors=False)
        
        # Diagonalize and collect the lowest three energies for odd sector
        eigvals_odd = scipy.sparse.linalg.eigsh(H_odd, k=3, which='SA', return_eigenvectors=False)
        
        # Combine and sort the eigenvalues from even and odd sectors, then take the lowest three
        combined_parity_eigvals = np.union1d(eigvals_even, eigvals_odd)
        combined_parity_eigvals.sort()
        
        # Append to the list of lowest three energies for the parity sectors
        lowest_three_parity.append(combined_parity_eigvals[:3])

        # Diagonalize and collect the lowest four energies for the z-basis for comparison
        eigvals_z = scipy.sparse.linalg.eigsh(H_z, k=4, which='SA', return_eigenvectors=False)
        eigvals_z.sort()

        # Append to the list of lowest three energies for the z-basis
        lowest_three_z_basis.append(eigvals_z[:3])

    # Reshape the lists for plotting
    lowest_three_parity = np.array(lowest_three_parity).T  # Transpose to match h_values shape
    lowest_three_z_basis = np.array(lowest_three_z_basis).T

    # Plot for parity sectors
    plt.plot(h_values, lowest_three_parity[0], label='Ground state (parity)')
    plt.plot(h_values, lowest_three_parity[1], label='1st excited state (parity)')
    plt.plot(h_values, lowest_three_parity[2], label='2nd excited state (parity)')

    # Plot for z-basis; using dashed lines for distinction
    plt.plot(h_values, lowest_three_z_basis[0], label='Ground state (z)', linestyle='--')
    plt.plot(h_values, lowest_three_z_basis[1], label='1st excited state (z)', linestyle='--')
    plt.plot(h_values, lowest_three_z_basis[2], label='2nd excited state (z)', linestyle='--')

    plt.xlabel('Transverse Field h')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'4-6_L{L_val}_energies.png')
