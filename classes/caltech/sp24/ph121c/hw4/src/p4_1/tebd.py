import numpy as np
import matplotlib.pyplot as plt

from hw4.src.fns import create_trotter_gates, apply_trotter_gates, enforce_bond_dimension, compute_contraction, apply_local_hamiltonian, open_dense_hamiltonian*

def create_initial_mps(l, name):
    up_physical = np.array([1, 0])
    down_physical = np.array([0, 1])
    mps = []
    for i in range(l):
        if i == 0:
            up_reshape = up_physical.reshape(2, 1)
            if name == 'ferro' or name == 'neel':
                mps.append(up_reshape)
        elif i == l-1:
            up_reshape = up_physical.reshape(1, 2)
            down_reshape = down_physical.reshape(1, 2)
            if name == 'ferro':
                mps.append(up_reshape)
            elif name == 'neel':
                mps.append(up_reshape if i % 2 == 0 else down_reshape)
        else:
            up_reshape = up_physical.reshape(1, 2, 1)
            down_reshape = down_physical.reshape(1, 2, 1)
            if name == 'ferro':
                mps.append(up_reshape)
            elif name == 'neel':
                mps.append(up_reshape if i % 2 == 0 else down_reshape)
    
    return mps

def compute_ground_state(L, chi, total_time, time_step, name):
    
    H = open_dense_hamiltonian(L)
    # front the lowest eigenvalue of the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    print(f'Lowest eigenvalue of the Hamiltonian is {eigenvalues[0]}')
    if name == 'ferro':
        initial_mps = create_initial_mps(L, name)
        # initial_ferro = make_product_state(np.array([1, 0]), L)
        # initial_energy = initial_ferro.T @ H @ initial_ferro
        # print(f"Initial energy ferro for L={L} is {initial_energy}")
    elif name == 'neel':
        initial_mps = create_initial_mps(L, name)
        # make the initial nil state by alternating the up and down spins in a loop over the length of the system
        up_physical = np.array([1, 0])
        down_physical = np.array([0, 1])
        initial_neel = up_physical
        for i in range(1, L):
            if i % 2 == 0:
                initial_neel = np.kron(initial_neel, up_physical)
            else:
                initial_neel = np.kron(initial_neel, down_physical)
        # initial_energy = initial_neel.T @ H @ initial_neel
        # print(f"Initial energy for neel L={L} is {initial_energy}")
            
            
    times = np.arange(0, total_time, time_step)
    energies = {}
    ground_states = {}
    current_mps = initial_mps
    # create the charter gate for the Kevin time stop
    gate_field, gate_odd, gate_even = create_trotter_gates(time_step*1j)
    for time in times:
        trotterized = apply_trotter_gates(current_mps, gate_field, gate_odd, gate_even)
        mps_enforced = enforce_bond_dimension(trotterized, chi)
        top = apply_local_hamiltonian(mps_enforced)
        normalization = compute_contraction(mps_enforced, mps_enforced)
        print(f'At time {time}, the top is {top} and the normalization is {normalization}')
        energy = top/normalization
        print(f'Energy at time {time} is {energy}')
            

        if time > 0:
            prev_time = time - time_step
            if prev_time in energies and (np.abs(energy - energies[prev_time]) / np.abs(energy)) < 1e-5:
                fino_energy = energy
                final_gs = mps_enforced
                break
        energies[time] = energy
        current_mps = mps_enforced

    # plt.figure()
    # plt.title(f"Energy vs. imaginary time for L={L}, time step={time_step}, name={name}")
    # plt.xlabel("Imaginary time")
    # plt.ylabel("Energy")
    # # I need these pas to be done with the same vertical scale: converged_energy -1 -> converged_energy + 5
    # plt.ylim(eigenvalues[0] - 1, eigenvalues[0] + 5)
    # plt.plot(energies.keys(), energies.values(), label="Energy")
    # plt.axhline(y=eigenvalues[0], color='r', linestyle='--', label="ED ground state")
    # plt.legend()
    # plt.savefig(f"hw4/docs/images/p4_1_energy_L_{L}time_step_{time_step}name_{name}.png")
    
    return final_gs, fino_energy

def plot_ground_state_density(ground_state_energies):
    plt.figure()
    plt.title("Ground state density as a function of system size")
    plt.xlabel("System Size (L)")
    plt.ylabel(r"Ground State Density $\frac{E(L+x) - E(L)}{x}$")
    
    ground_state_densities = []
    system_sizes = sorted(ground_state_energies.keys())
    
    for i in range(1, len(system_sizes)):
        if i == 0:
            continue
        for time_step in ground_state_energies[system_sizes[i]]:
            for name in ground_state_energies[system_sizes[i]][time_step]:
                ground_state_density = (ground_state_energies[system_sizes[i]][time_step][name] - ground_state_energies[system_sizes[i-1]][time_step][name])/1
                ground_state_densities.append(ground_state_density)
    
    plt.plot(system_sizes[1:], ground_state_densities, 'o-')
    plt.savefig("hw4/docs/images/ground_state_density_tst.png")
    return

def get_correlations(mps):
    L = len(mps)
    correlations = {}
    # define the three different pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigmas = [sigma_x, sigma_y, sigma_z]
    # now loop over them to determine the separate correlations
    for s in range(len(sigmas)):
        correlations[s] = {}
        # first tensor
        first_tensor = mps[0]
        # contact the first tensor with the sigma_z matrix
        first_sigma_contraction = np.einsum('bc,bd,de->ce', first_tensor.conj(), sigmas[s], first_tensor)
        # dope offer the remaining once
        for i in range(1, L):
            second_tensor = mps[i]
            # compute constructions of the second dancer with the one directly to its left until we reach the first site to get a scaler
            if i == L-1:
                second_sigma_contraction = np.einsum('ab,bd,cd->ac', second_tensor.conj(), sigmas[s], second_tensor)
            else:
                second_sigma_contraction = np.einsum('abc,bd,jdc->aj', second_tensor.conj(), sigmas[s], second_tensor)
            # contract the remaining tensors to the left of the second tensor until we reach the first site
            for j in range(i-1, 0, -1):
                contracted_tensor_new = np.einsum('akc,jkl,cl->aj', mps[j].conj(), mps[j], second_sigma_contraction)
                second_sigma_contraction = contracted_tensor_new
            # the last case will give a scaler
            correlations[s][i] = np.einsum('ab,ab->', first_sigma_contraction, second_sigma_contraction)
    return correlations

        

    
    
def main():
    L = [6,8]
    total_time = 10
    time_steps = [0.1]
    chi = 16
    # initialize the ground state
    gs = {}
    gs_es = {}
    for l in L:
        gs[l] = {}
        gs_es[l] = {}
        for time_step in time_steps:
            gs[l][time_step] = {}
            gs_es[l][time_step] = {}
            for name in ['ferro']:
                ground_state, energy = compute_ground_state(l, chi, total_time, time_step, name)
                gs[l][time_step][name] = ground_state
                gs_es[l][time_step][name] = energy
    # now plot the ground state density using the energies just obtained
    # plot_ground_state_density(gs_es)
    for l in L:
        for time_step in time_steps:
            for name in ['ferro']:
                correlations = get_correlations(gs[l][time_step][name])
                for s in range(len(correlations)):
                    if s == 0:
                        sigma_name = 'x'
                    elif s == 1:
                        sigma_name = 'y'
                    else:
                        sigma_name = 'z'
                    plt.figure()
                    plt.title(f"Correlation function for {sigma_name} as a function of distance")
                    plt.xlabel("Distance")
                    plt.ylabel(rf"Correlation $\langle  {sigma_name}^1 \cdot {sigma_name} \rangle$")
                    plt.plot(correlations[s].keys(), correlations[s].values(), 'o-')
                    plt.savefig(f"hw4/docs/images/correlation_{sigma_name}.png")
    # plot_correlations_fns(gs['neel'])

if __name__ == "__main__":
    main()
