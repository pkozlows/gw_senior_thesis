import numpy as np
import matplotlib.pyplot as plt
from hw4.src.p4_1.imaginary_tebd_fns import create_trotter_gates, create_initial_mps, imaginary_tebd_step
from hw4.src.contraction_fns import apply_local_hamiltonian, compute_contraction
from hw4.src.p4_1.plt_fns import get_correlations, plot_correlations
L = 20
total_time = 10
dt = 0.01
times = np.arange(0, total_time, dt)
chi_values = [16]
conv_tol = 1e-4
names = ['neel', 'ferro']
# create the inderal state
for name in names:
    initial_state = create_initial_mps(L, name)

    # start by creating the gates for the imaginary time evolution
    gate_field, gate_odd, gate_even = create_trotter_gates(1j*dt, h_z=0.5)

    for time in times:
        state = imaginary_tebd_step(initial_state, chi_values[0], time, [gate_field, gate_odd, gate_even])
        # compute the energy of the state
        energy = apply_local_hamiltonian(state, h_z=0.5)
        print(f'Energy at time {time} is {energy}')
        if time > 0:
            if np.abs(energy - previous_energy) / np.abs(energy) < conv_tol:
                gs = state
                break
        # set the values equal to the value from the previous alteration
        previous_energy = energy
        initial_state = state
    # now compute correlation functions from the converged ground state
    correlations = get_correlations(gs)
    plot_correlations(correlations, name, L, dt)


