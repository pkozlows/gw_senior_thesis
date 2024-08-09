import numpy as np
import matplotlib.pyplot as plt
from hw4.src.p4_1.imaginary_tebd_fns import check_left_canonical, check_right_canonical


def orthogonalize(mps, position):
    L = len(mps)
    for i in range(position):
        if i == 0:
            contraction = np.einsum('ij,jab->iab', mps[i], mps[i+1])
            w = contraction.reshape(mps[i].shape[0], mps[i+1].shape[1]*mps[i+1].shape[2])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            mps[i] = U.reshape(mps[i].shape[0], -1)
            assert check_left_canonical(mps[i])
            mps[i+1] = (np.diag(S) @ V).reshape(-1, mps[i+1].shape[1], mps[i+1].shape[2])
        else:
            contraction = np.einsum('ijk,klm->ijlm', mps[i], mps[i+1])
            w = contraction.reshape(mps[i].shape[0]*mps[i].shape[1], mps[i+1].shape[1]*mps[i+1].shape[2])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            mps[i] = U.reshape(mps[i].shape[0], mps[i].shape[1], -1)
            assert check_left_canonical(mps[i])
            mps[i+1] = (np.diag(S) @ V).reshape(-1, mps[i+1].shape[1], mps[i+1].shape[2])
    # now move in reverse and make sure pitting is right canonical
    for i in range(L-1, position, -1):
        if i == L-1:
            contraction = np.einsum('ijk,kl->ijl', mps[i-1], mps[i])
            w = contraction.reshape(mps[i-1].shape[0]*mps[i-1].shape[1], mps[i].shape[1])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            mps[i-1] = (U @ np.diag(S)).reshape(mps[i-1].shape[0], mps[i-1].shape[1], -1)
            mps[i] = V.reshape(-1, mps[i].shape[1])
            assert check_right_canonical(mps[i])
        else:
            contraction = np.einsum('ijk,klm->ijlm', mps[i-1], mps[i])
            w = contraction.reshape(mps[i-1].shape[0]*mps[i-1].shape[1], mps[i].shape[1]*mps[i].shape[2])
            U, S, V = np.linalg.svd(w, full_matrices=False)
            mps[i-1] = (U @ np.diag(S)).reshape(mps[i-1].shape[0], mps[i-1].shape[1], -1)
            mps[i] = V.reshape(-1, mps[i].shape[1], mps[i].shape[2])
            assert check_right_canonical(mps[i])
    return mps

def measure_observable(state, pauli, position):
    """Measure an observable on a state."""
    L = len(state)
    # define the pauli matrices
    if pauli == 'sigma_x':
        pauli = np.array([[0, 1], [1, 0]])
    elif pauli == 'sigma_z':
        pauli = np.array([[1, 0], [0, -1]])

    state = orthogonalize(state, position)

    # now we can just measure the observable at the orthogonality center/position
    expectation = np.einsum('ijk,jl,ilk->', state[position].conj(), pauli, state[position])


    return expectation



def plot_observables(observable_values, L, chi):
    for observable, values in observable_values.items():
        times_sorted = sorted(values.keys())
        values_sorted = [values[time] for time in times_sorted]
        
        plt.figure()
        plt.plot(times_sorted, values_sorted, label=f'{observable[0]} at site {observable[1]}')
        plt.xlabel('Time')
        plt.ylabel('Observable Value')
        plt.title(f'Observable {observable[0]} at site {observable[1]}, L={L}, chi={chi}')
        plt.legend()
        plt.savefig(f"hw4/docs/images/observable_{observable[0]}_site_{observable[1]}_L_{L}_chi_{chi}.png")
    return



def entanglement_entropy(mps, position):
    # put the orthogonality center at the position
    mixed_canonical = orthogonalize(mps, position)
    L = len(mps)
    # now compute the svd at the position
    center = mixed_canonical[position]
    U, S, V = np.linalg.svd(center, full_matrices=False)
    # calculate entanglement entropy from the singular values
    entropy = -np.sum(S**2 * np.log(S**2))
    return entropy

def plot_entanglement_entropy(entropies, L, chi):
    entropies_sorted = [entropies[time] for time in sorted(entropies.keys())]
    plt.figure()
    plt.plot(sorted(entropies.keys()), entropies_sorted)
    plt.xlabel('Time')
    plt.ylabel(rf'Half system Entanglement Entropy ($S_{{L/2}}$))')
    plt.title(rf'Entanglement Entropy for $L={L}$ and $\chi={chi}$')
    plt.savefig(f"hw4/docs/images/hs_ee_L_{L}_chi_{chi}.png")
    return

def plot_combined_entanglement_entropy(entropies_dict, L, chi):
    plt.figure()
    for state, entropies in entropies_dict.items():
        entropies_sorted = [entropies[time] for time in sorted(entropies.keys())]
        plt.plot(sorted(entropies.keys()), entropies_sorted, label=f'State={state}')
    plt.xlabel('Time')
    plt.ylabel(rf'Half system Entanglement Entropy ($S_{{L/2}}$)')
    plt.title(rf'Entanglement Entropy for $L={L}$ and $\chi={chi}$')
    plt.legend()
    plt.savefig(f"hw4/docs/images/combined_hs_ee_L_{L}_chi_{chi}.png")
