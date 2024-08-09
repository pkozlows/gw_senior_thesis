import numpy as np
import matplotlib.pyplot as plt
from hw4.src.p4_1.imaginary_tebd_fns import create_trotter_gates, apply_trotter_gates, enforce_bond_dimension
# from hw4.src.p4_2.observable_fns import orthogonalize
def apply_observable(state, coordinate):
    """Compute the correlations for a given state."""
    L = len(state)
    half_sys_py = int(L/2) - 1
    # define the pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    if coordinate == 'x':
        observable = sigma_x
    elif coordinate == 'y':
        observable = sigma_y
    elif coordinate == 'z':
        observable = sigma_z
    state[half_sys_py] = np.einsum('ijk,jl->ilk', state[half_sys_py], observable)
    return state

def real_tebd_step(current_mps, chi, time, braket, gates, coordinate):
    """Real time evolution using TEBD."""
    gate_field, gate_odd, gate_even = gates
    transformed_state = []
    if braket == 'ket':
        transformed_state = apply_observable(current_mps.copy(), coordinate)
    trotterized = apply_trotter_gates(current_mps, gate_field, gate_odd, gate_even)
    mps_enforced = enforce_bond_dimension(trotterized, chi)
    if braket == 'bra':
        transformed_state = apply_observable(mps_enforced.copy(), coordinate)
    return transformed_state

# Function to plot the correlation functions
def plot_correlation_functions(correlation_functionS_time_evolution, model):
    plt.figure()
    times = sorted(next(iter(correlation_functionS_time_evolution.values())).keys())
    coordinates = ['x', 'y', 'z']
    labels = {'x': r'$C^{xx}(t)$', 'y': r'$C^{yy}(t)$', 'z': r'$C^{zz}(t)$'}

    for coordinate in coordinates:
        values = [correlation_functionS_time_evolution[coordinate][time] for time in times]
        plt.plot(times, values, label=labels[coordinate])

    plt.xlabel('Time')
    plt.ylabel('Correlation Function')
    plt.title(f'Time Evolution of Correlation Functions for {model} model')
    plt.legend()
    plt.savefig(f'hw4/docs/images/{model}_correlation_functions.png')
    