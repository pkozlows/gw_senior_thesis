import numpy as np
import matplotlib.pyplot as plt
from hw4.src.p4_1.ed_fns import open_dense_hamiltonian
from hw4.src.p4_1.imaginary_tebd_fns import create_initial_mps, create_trotter_gates, apply_trotter_gates, enforce_bond_dimension
from hw4.src.contraction_fns import apply_local_hamiltonian
from hw4.src.p4_2.observable_fns import measure_observable, plot_observables, entanglement_entropy, plot_entanglement_entropy, plot_combined_entanglement_entropy
from hw4.src.p4_3.dynamics_fns import apply_observable

def real_tebd(L, chi, total_time, dt, name):
    """Real time evolution using TEBD."""
    initial_mps = create_initial_mps(L, name)
    # measure some observables of the initial state
    first_observable = ('sigma_x', int(L/2))
    second_observable = ('sigma_x', 1)
    third_observable = ('sigma_z', int(L/2))
    observables = [first_observable, second_observable, third_observable]
    observable_values = {obs: {} for obs in observables}

    times = np.arange(0, total_time, dt)
    energies = {}
    entropies = {}
    current_mps = initial_mps
    gate_field, gate_odd, gate_even = create_trotter_gates(-dt)

    for time in times:
        for observable in observables:
            observable_values[observable][time] = measure_observable(current_mps, observable[0], observable[1])
        # compute the entitlement and copy for the half system
        ee = entanglement_entropy(current_mps, int(L/2))
        entropies[time] = ee
        trotterized = apply_trotter_gates(current_mps, gate_field, gate_odd, gate_even)
        mps_enforced = enforce_bond_dimension(trotterized, chi)
        energy = apply_local_hamiltonian(mps_enforced)
        print(f'Energy at time {time} is {energy}')

        energies[time] = energy

        if time > 0:
            prev_time = time - dt
            if prev_time in energies and (np.abs(energy - energies[prev_time]) / np.abs(energy)) < 1e-8:
                final_gs = mps_enforced
                break

        current_mps = mps_enforced

    return observable_values, entropies, times





# Main execution for different initial states and chi values
L = 16
total_time = 5
dt = 0.2
initial_states = ['neel', 'three']
chi_values = [8]


for chi in chi_values:
    observable_values_dict = {}
    entropies_dict = {}
    for state in initial_states:
        print(f'Running TEBD for state={state} with chi={chi}')
        observable_values, entropies, times = real_tebd(L, chi, total_time, dt, state)
        observable_values_dict[state] = observable_values
        entropies_dict[state] = entropies

    plot_combined_entanglement_entropy(entropies_dict, L, chi)

    
