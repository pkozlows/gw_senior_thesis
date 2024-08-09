import numpy as np
import matplotlib.pyplot as plt
def plot_energy_vs_time(energies, name, L, time_step, gs_e=None):
    """Takes in the ground state energy, the converging energies, and the name of the initial state. Returns a plot."""
    plt.figure()
    plt.title(f"Energy vs. imaginary time for L={L}, time step={time_step}, name={name}")
    plt.xlabel("Imaginary time")
    plt.ylabel("Energy")
    # I need these pas to be done with the same vertical scale: converged_energy -1 -> converged_energy + 5
    print(energies.keys(), energies.values())
    plt.plot(energies.keys(), energies.values(), label="Energy")
    if gs_e is not None:
        plt.ylim(gs_e - 1, gs_e + 2)
        plt.axhline(y=gs_e, color='r', linestyle='--', label="ED ground state")
    plt.legend()
    plt.savefig(f"hw4/docs/images/p4_1_energy_L_{L}_time_step_{time_step}_name_{name}.png")
    return

def plot_ground_state_density(ground_state_energies, name):
    plt.figure()
    plt.title("Ground state density as a function of system size")
    plt.xlabel("System Size (L)")
    plt.ylabel(r"Ground State Density $\frac{E(L+x) - E(L)}{x}$")
    
    system_sizes = sorted(ground_state_energies.keys())
    
    ground_state_densities = {}
    for i in range(1, len(system_sizes)):
        # Extract the energies for the given time step
        time_step = list(ground_state_energies[system_sizes[i]].keys())[0]
        energy_current = list(ground_state_energies[system_sizes[i]][time_step].values())[-1]
        energy_previous = list(ground_state_energies[system_sizes[i-1]][time_step].values())[-1]
        
        size_difference = system_sizes[i] - system_sizes[i-1]
        ground_state_densities[system_sizes[i]] = (energy_current - energy_previous) / size_difference
    
    
    plt.plot(system_sizes[1:], list(ground_state_densities.values()), 'o-')
    plt.savefig(f"hw4/docs/images/ground_state_density_name_{name}.png")
    return

def get_correlations(mps):
    L = len(mps)
    correlations = {}
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigmas = [sigma_x, sigma_y, sigma_z]

    for s, sigma in enumerate(sigmas):
        # the first pauli matrix is always at the first site
        correlations[s] = {}
        first_tensor = mps[0]
        first_sigma_contraction = np.einsum('bc,bd,de->ce', first_tensor.conj(), sigma, first_tensor)
        # move to the right to determine the second point for the pauli matrix
        for i in range(1, L):
            second_tensor = mps[i]
            if i == L-1:
                second_sigma_contraction = np.einsum('ab,bd,cd->ac', second_tensor.conj(), sigma, second_tensor)
            else:
                second_sigma_contraction = np.einsum('abc,bd,jdc->aj', second_tensor.conj(), sigma, second_tensor)
            # now the train is going to the left
            for j in range(i-1, 0, -1):
                second_sigma_contraction = np.einsum('akc,jkl,cl->aj', mps[j].conj(), mps[j], second_sigma_contraction)
            
            correlations[s][i] = np.einsum('ab,ab->', first_sigma_contraction, second_sigma_contraction)

    return correlations


def plot_correlations(correlations, name, L, dt):
    sigma_labels = [r'\sigma_x', r'\sigma_y', r'\sigma_z']
    sigma_file_labels = ['sigma_x', 'sigma_y', 'sigma_z']  # Labels without LaTeX formatting for filenames
    for s, (sigma_label, sigma_file_label) in enumerate(zip(sigma_labels, sigma_file_labels)):
        plt.figure()
        plt.title(f"Correlation function for {sigma_label} for L={L}, dt={dt}, {name}")
        plt.xlabel("Distance")
        plt.ylabel(rf"Correlation $\langle {sigma_label}^1 \cdot {sigma_label}^r \rangle$")
        plt.plot(list(correlations[s].keys()), list(correlations[s].values()), 'o-')
        plt.savefig(f"hw4/docs/images/p4_1_correlation_{sigma_file_label}_L_{L}_dt_{dt}_name_{name}.png")
    return

