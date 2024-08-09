import numpy as np
import matplotlib.pyplot as plt
from hw4.src.p4_1.imaginary_tebd_fns import create_trotter_gates, create_initial_mps, imaginary_tebd_step
from hw4.src.p4_3.dynamics_fns import real_tebd_step, plot_correlation_functions
from hw4.src.contraction_fns import apply_local_hamiltonian, compute_contraction
L = 12
total_time = 10
dt = 0.1
times = np.arange(0, total_time, dt)
initial_state = create_initial_mps(L, 'neel')
chi_values = [16]
conv_tol = 1e-6

for model in ['non_integrable', 'integrable']:
    if model == 'non_integrable':
        h_z_val = 0.5
    elif model == 'integrable':
        h_z_val = 0.0
    # first we want to do an imagine time tebd two find the ground state
    gate_field, gate_odd, gate_even = create_trotter_gates(1j*dt, h_z=h_z_val)
    
    for time in times:
        state = imaginary_tebd_step(initial_state, chi_values[0], time, [gate_field, gate_odd, gate_even])
        # compute the energy of the state
        energy = apply_local_hamiltonian(state, h_z=h_z_val)
        print(f'Energy at time {time} is {energy}')
        if time > 0:
            if np.abs(energy - previous_energy) / np.abs(energy) < conv_tol:
                gs = state
                break
        # set the values equal to the value from the previous alteration
        previous_energy = energy
        initial_state = state    
    
    # Now, use real-time TEBD to calculate the correlation functions
    # we must create new gates first
    gate_field, gate_odd, gate_even = create_trotter_gates(-dt, h_z=h_z_val)
    correlation_functionS_time_evolution = {}
    for coordinate in ['x', 'y', 'z']:
        bra = gs
        ket = gs

        correlation_function_time_evolution = {}

        for time in times:
            for braket in ['bra', 'ket']:
                if braket == 'bra':
                    transformed_bra = real_tebd_step(bra, chi_values[0], time, braket, [gate_field, gate_odd, gate_even], coordinate)
                    bra = transformed_bra
                if braket == 'ket':
                    transformed_ket = real_tebd_step(ket, chi_values[0], time, braket, [gate_field, gate_odd, gate_even], coordinate)
                    ket = transformed_ket
            correlation_function_time_evolution[time] = compute_contraction(transformed_bra, transformed_ket)


        correlation_functionS_time_evolution[coordinate] = correlation_function_time_evolution

    plot_correlation_functions(correlation_functionS_time_evolution, model)
        




