import numpy as np
import matplotlib.pyplot as plt

from hw2.src.p5_5_1_2 import compute_mps
from hw4.src.fns import open_dense_hamiltonian, create_trotter_gates, apply_trotter_gates, enforce_bond_dimension, compute_contraction
from hw3.src.p4_1.fns import make_product_state
from hw2.src.p5_5_2_2 import check_left_canonical, check_right_canonical

def create_initial_mps(l):
    up_physical = np.array([1, 0])
    down_physical = np.array([0, 1])
    
    ferro_mps = []
    neel_mps = []
    for i in range(l):
        if i == 0:
            up_reshape = up_physical.reshape(2, 1)
            ferro_mps.append(up_reshape)
            neel_mps.append(up_reshape)
        elif i == l-1:
            up_reshape = up_physical.reshape(1, 2)
            ferro_mps.append(up_reshape)
            neel_mps.append(up_reshape if i % 2 == 0 else down_physical.reshape(1, 2))
        else:
            up_reshape = up_physical.reshape(1, 2, 1)
            ferro_mps.append(up_reshape)
            neel_mps.append(up_reshape if i % 2 == 0 else down_physical.reshape(1, 2, 1))
    
    return {'ferro': ferro_mps, 'neel': neel_mps}

def compute_ground_state(L, total_time, time_steps):
    ground_state_energies = {}
    ground_states = {}
    
    for l in L:
        H = open_dense_hamiltonian(l)
        eigenvalues, _ = np.linalg.eigh(H)
        initial_ferro = make_product_state(np.array([1, 0]), l)
        initial_energy = initial_ferro.T @ H @ initial_ferro
        print(f"Initial energy for L={l} is {initial_energy}")

        initial_mps = create_initial_mps(l)
        ground_state = {}
        for key, mps_list in initial_mps.items():
            current_mps = mps_list
            
            for i in time_steps:
                times = np.linspace(0, total_time, int(total_time / i))
                energies = {}
                
                for time in times:
                    gate_field, gate_odd, gate_even = create_trotter_gates(i)
                    trotterized = apply_trotter_gates(current_mps, gate_field, gate_odd, gate_even)
                    chi = 4
                    mps_enforced = enforce_bond_dimension(trotterized, chi)
                    bra_mps = [np.conj(mps) for mps in mps_list]
                    energy = compute_contraction(mps_enforced, bra_mps)
                    print(energy)
                    
                    # if time > 0:
                    #     if (np.abs(energy - energies[time - i]) / np.abs(energy)) < 1e-6:
                    #         if key == 'ferro':
                    #             ground_state_energies[l] = energy
                    #         ground_states[l][key] = mps_enforced
                    #         break

                    energies[time] = energy
                    current_mps = mps_enforced

        plt.figure()
        plt.title(f"Energy as a function of time for different imaginary time steps for L={l}")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.plot(energies.keys(), energies.values(), label="Energy")
        plt.axhline(y=eigenvalues[0], color='r', linestyle='--', label="ED Solution")
        plt.legend()
        plt.savefig(f"hw4/docs/images/p4_1_energy_L_{l}.png")
    
    return ground_state, ground_state_energies

def plot_ground_state_density(ground_state_energies):
    plt.figure()
    plt.title("Ground state density as a function of system size")
    plt.xlabel("System Size (L)")
    plt.ylabel(r"Ground State Density $\frac{E(L+x) - E(L)}{x}$")
    
    ground_state_densities = []
    L = sorted(ground_state_energies.keys())
    
    for i in range(1, len(L)):
        delta_E = ground_state_energies[L[i]] - ground_state_energies[L[i - 1]]
        ground_state_densities.append(delta_E / (L[i] - L[i - 1]))
    
    plt.plot(L[1:], ground_state_densities, 'o-')
    plt.savefig("hw4/docs/images/ground_state_density.png")

def plot_correlations_fns(mps):
    L = len(mps)
    correlations = {}
    # define the three different pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigmas = [sigma_x, sigma_y, sigma_z]
    # now loop over them to determine the separate correlations
    for sigma in sigmas:
        correlations[sigma] = {}
        # first tensor
        first_tensor = mps[0]
        # contact the first tensor with the sigma_z matrix
        first_sigma_contraction = np.einsum('bc,bd,de->ce', first_tensor, sigma, first_tensor.conj())
        # dope offer the remaining once
        for i in range(1, L):
            second_tensor = mps[i]
            # compute constructions of the second dancer with the one directly to its left until we reach the first site to get a scaler
            second_sigma_contraction = np.einsum('abc,bd,jdc->aj', second_tensor, sigma, second_tensor.conj())
            # contract the remaining tensors to the left of the second tensor until we reach the first site
            for j in range(i-1, 0, -1):
                contracted_tensor_new = np.einsum('akc,jkl,cl->aj', mps[j], mps[j].conj(), second_sigma_contraction)
                second_sigma_contraction = contracted_tensor_new
            # the last case will give a scaler
            correlations[sigma][i] = np.einsum('ab,ab->', first_sigma_contraction, second_sigma_contraction)
            # now plot the correlations as a function of distance
            plt.figure()
            plt.title(f"Correlation function for {sigma} as a function of distance")
            plt.xlabel("Distance")
            plt.ylabel(rf"Correlation $\langle  {sigma}^1 \cdot {sigma}^{i+1} \rangle$")
            plt.plot(correlations[sigma].keys(), correlations[sigma].values(), 'o-')
            plt.savefig(f"hw4/docs/images/correlation_{sigma}.png")
        

    
    
def main():
    L = [10]
    total_time = 1
    time_steps = [0.1, 0.01]
    
    gs, gs_es = compute_ground_state(L, total_time, time_steps)
    plot_ground_state_density(gs_es)
    plot_correlations_fns(gs['ferro'])
    plot_correlations_fns(gs['neel'])

if __name__ == "__main__":
    main()
