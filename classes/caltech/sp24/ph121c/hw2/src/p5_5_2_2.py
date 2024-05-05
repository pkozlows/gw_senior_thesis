import numpy as np
from p5_5 import truncate_svd  # Assuming truncate_svd performs SVD and truncates based on some criterion

def initialize_mps(L):
    mps_tensors = []
    A_1_up = np.array([1, 2])
    A_1_down = np.array([2, -1])
    mps_tensors.append(np.stack((A_1_up, A_1_down), axis=0).reshape(1, 2, 2))

    for site in range(1, L-1):
        A_ell_up = np.array([[1, 1], [0, 0]]) / np.sqrt(2)
        A_ell_down = np.array([[0, 0], [1, -1]]) / np.sqrt(2)
        mps_tensors.append(np.stack((A_ell_up, A_ell_down), axis=0))

    A_L_up = np.array([1, 3])
    A_L_down = np.array([3, 1])
    mps_tensors.append(np.stack((A_L_up, A_L_down), axis=0).reshape(2, 2, 1))
    return mps_tensors

def shift_orthogonality_center(mps_tensors, target):
    L = len(mps_tensors)
    
    # Make left-canonical up to the target site
    for i in range(target):
        tensor_matrix = mps_tensors[i].reshape(-1, mps_tensors[i].shape[-1])
        u, s, vt = truncate_svd(tensor_matrix, min(tensor_matrix.shape))
        mps_tensors[i] = u.reshape(mps_tensors[i].shape[0], mps_tensors[i].shape[1], u.shape[-1])
        if i + 1 < L:
            next_tensor_matrix = mps_tensors[i+1].reshape(s.shape[0], -1)
            mps_tensors[i+1] = np.dot(s @ vt, next_tensor_matrix).reshape(u.shape[-1], *mps_tensors[i+1].shape[1:])
    
    # Make right-canonical from the end down to target+1
    for i in range(L-1, target, -1):
        tensor_matrix = mps_tensors[i].reshape(mps_tensors[i].shape[0], -1)
        u, s, vt = truncate_svd(tensor_matrix, min(tensor_matrix.shape))
        mps_tensors[i] = vt.reshape(vt.shape[0], *mps_tensors[i].shape[1:])
        if i > 0:
            prev_tensor_matrix = mps_tensors[i-1].reshape(-1, u.shape[0])
            mps_tensors[i-1] = np.dot(prev_tensor_matrix, u @ s).reshape(*mps_tensors[i-1].shape[:-1], vt.shape[0])

def check_canonical(mps_tensors):
    for idx, tensor in enumerate(mps_tensors):
        # Left canonical check
        reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])
        if np.allclose(reshaped_tensor.conj().T @ reshaped_tensor, np.eye(reshaped_tensor.shape[-1])):
            print(f"Tensor at site {idx+1} is left-canonical.")
        else:
            print(f"Tensor at site {idx+1} is NOT left-canonical.")
        
        # Right canonical check
        reshaped_tensor = tensor.reshape(tensor.shape[0], -1)
        if np.allclose(reshaped_tensor @ reshaped_tensor.conj().T, np.eye(reshaped_tensor.shape[0])):
            print(f"Tensor at site {idx+1} is right-canonical.")
        else:
            print(f"Tensor at site {idx+1} is NOT right-canonical.")

# Example usage
L = 6
mps_tensors = initialize_mps(L)
shift_orthogonality_center(mps_tensors, 3)  # Move the orthogonality center to site 3
check_canonical(mps_tensors)
