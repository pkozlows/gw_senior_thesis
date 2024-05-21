import numpy as np
import matplotlib.pyplot as plt

from hw2.src.p5_5_1_2 import compute_mps
from hw4.src.fns import open_dense_hamiltonian, create_trotter_gates, apply_trotter_gates, enforce_bond_dimension, compute_contraction
from hw3.src.p4_1.fns import make_product_state
from hw2.src.p5_5_2_2 import check_left_canonical, check_right_canonical

# Define the parameters
L = [6, 8, 10]
times = [0.1, 0.01, 0.001]
# Define the sample ferromagnet product state for a single site
up_physical = [1, 0]
down_physical = [0, 1]


for l in L:
    # Open the dense Hamiltonian for the current system size
    H = open_dense_hamiltonian(l)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # take the tensor critic of the state on a single site to form the initial state as a product state
    non_mps_initial = make_product_state(np.array([1, 0]), l)

    initial_energy = non_mps_initial.T @ H @ non_mps_initial
    normalisation = non_mps_initial.T @ non_mps_initial
    assert np.isclose(normalisation, 1)
    print(f"Initial energy for L={l}: {initial_energy}")
    
    # Translate the full product state into MPS form
    mps_list = []
    for i in range(l):
        up_physical=np.array(up_physical).reshape(1, 2, 1)
        mps_list.append(up_physical)

    
    for t in times:
        gate_field, gate_odd, gate_even = create_trotter_gates(t)
        # Apply the gates to the MPS
        trotterized = apply_trotter_gates(mps_list, gate_field, gate_odd, gate_even)
        # now we want to enforce nonincreasing bond dimensions
        chi = 4
        mps_enforced = enforce_bond_dimension(trotterized, chi)
        # now compute the energy of this mps
        bra_mps = [t.conj().T for t in mps_list]
        energy = compute_contraction(mps_enforced, bra_mps)
        # normalization = compute_contraction(mps_enforced, mps_enforced)
        # print(normalization)
        # assert np.isclose(normalization, 1)
        print(f"Energy for L={l}, t={t}: {energy }")
        

            