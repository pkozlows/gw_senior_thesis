import numpy as np
import matplotlib.pyplot as plt
from hw3.src.p4_3.fns import periodic_dense_hamiltonian_scars
from hw2.src.p5_2 import entanglement_entropy, calculate_reduced_density_matrix

# Define system parameters
L_values = [6, 8, 10]
omega = 5

# Initialize the plotting environment outside the loop
plt.figure(figsize=(10, 6))
plt.title("Entropic Signature of Thermalization for Scar Hamiltonian")
plt.xlabel(f'Eigenvalue ($\epsilon_n$)')
plt.ylabel(f'Half-System Entanglement Entropy ($S_n$)')

# Generate and analyze Hamiltonians for each system size
for L in L_values:
    # Generate the Hamiltonian
    H = periodic_dense_hamiltonian_scars(L, omega)
    # Diagonalize the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Compute entanglement entropy for each eigenstate
    entropies = [entanglement_entropy(calculate_reduced_density_matrix(vec, L, L // 2)) for vec in eigenvectors]

    # Plot entanglement entropy
    plt.plot(np.array(eigenvalues), np.array(entropies), label=f'L={L}')

    # annotate the energy valdes of the scars on the plot; that is mark evidently far less entangled than the typical eigenstate. For this simple model, the number and exact energies of the scar states are known, indexed by $m=\{-L / 2,-L / 2+1, \ldots, L / 2-1, L / 2\}$ (that is, the $S^{z}$ spin states of an overall spin $s=L / 2$ system), with harmonically spaced energies $E_{m}=\Omega m$. Indicate these energy values on your plot.
    for m in range(-L // 2, L // 2 + 1):
        plt.axvline(x=m * omega, color='black', linestyle='--', alpha=0.5)

# Add legend and grid to the plot
plt.legend()
plt.grid(True)
plt.savefig("hw3/docs/images/p4_3_scars_entropic_signature.png")
