from hw1 import sparse_hamiltonian
import numpy as np
import matplotlib.pyplot as plt

def schmidt_decomposition(psi, L):
    """Performs Schmidt decomposition at the middle of the chain."""
    cut_psi = psi.reshape(+2**(L//2), -1)
    U, s, Vt = np.linalg.svd(cut_psi, full_matrices=False)
    return cut_psi, U, s, Vt

def truncated_state(U, S, Vh, k):
    """Reconstructs the state using only the first k singular values."""
    Sk = np.zeros_like(S)
    Sk[:k] = S[:k]
    flat_psi_k = U @ np.diag(Sk) @ Vh
    return flat_psi_k, flat_psi_k.flatten()

h_values = [0.3, 1, 1.7]
L = 8
errors = {hi: [] for hi in h_values}

# Single loop to handle everything per each h
for hi in h_values:
    # Compute the ground state
    H = sparse_hamiltonian(L, hi, periodic=False)
    eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
    ground_state = eigenvectors[:, 0]
    
    # Perform Schmidt decomposition
    original, U, s, Vt = schmidt_decomposition(ground_state, L)
    
    # Loop through all possible k values for truncation
    for k in range(1, 2**(L//2) + 1):
        mat_psi_k, flat_psi_k = truncated_state(U, s, Vt, k)
        norm_error = np.linalg.norm(original - mat_psi_k, 'fro')
        psi_k_energy = (flat_psi_k.conj().T @ H @ flat_psi_k) / (flat_psi_k.conj().T @ flat_psi_k)
        energy_error = np.abs(psi_k_energy - eigenvalues[0])
        errors[hi].append((norm_error, energy_error))

# Plotting the results
for hi, error_list in errors.items():
    norm_errors, energy_errors = zip(*error_list)
    plt.plot(norm_errors, energy_errors, label=f'h/J = {hi:.1f}')

plt.xlabel('Frobenius Norm Error')
plt.ylabel('Energy Error')
plt.title('Energy Error vs Frobenius Norm Error')
plt.legend()
plt.grid()
plt.savefig('energy_vs_frobenius.png')
