import numpy as np
import matplotlib.pyplot as plt

from hw4.src.p4_1.imaginary_tebd_fns import create_trotter_gates, apply_trotter_gates, enforce_bond_dimension, create_initial_mps, flatten_mps
from hw4.src.contraction_fns import apply_local_hamiltonian, compute_contraction
from hw4.src.p4_1.ed_fns import open_dense_hamiltonian
from hw4.src.p4_1.plt_fns import plot_energy_vs_time, plot_ground_state_density, get_correlations, plot_correlations





def main():
    H = open_dense_hamiltonian(8)
    eigvals, eigvecs = np.linalg.eigh(H)
    gs_e_12 = eigvals[0]
    system_sizes = [8]
    total_time = 8
    time_steps = [0.01]
    initial_states = ['ferro', 'neel']
    chi = 10

    ground_states = {}
    ground_state_energies = {}

    for name in initial_states:
        ground_states[name] = {}
        ground_state_energies[name] = {}
        
        for L in system_sizes:
            if L == 12 and name == 'ferro':
                gs_e = gs_e_12
            else:
                gs_e = None
            ground_states[name][L] = {}
            ground_state_energies[name][L] = {}
            
            for dt in time_steps:
                ground_state, energies = compute_ground_state(L, chi, total_time, dt, name)
                ground_states[name][L][dt] = ground_state
                # compute the inner protect of this determined ground state with the true one from ed
                tebd_gs = flatten_mps(ground_state)
                
                inner_product = np.abs(np.vdot(tebd_gs, eigvecs[:, 0]))
                print(f'Inner product for {name} with L={L} and dt={dt} is {inner_product}')
                
                ground_state_energies[name][L][dt] = energies
                
                plot_energy_vs_time(energies, name, L, dt, gs_e)
        
        plot_ground_state_density(ground_state_energies[name], name)
        # only measure the correlations for the largest system size in the list
        for L in system_sizes[-1:]:
            # also do this only for the small list time step coma which is at the end of the list
            for dt in time_steps[-1:]:
                correlations = get_correlations(ground_states[name][L][dt])
                plot_correlations(correlations, name, L, dt)

if __name__ == "__main__":
    main()
