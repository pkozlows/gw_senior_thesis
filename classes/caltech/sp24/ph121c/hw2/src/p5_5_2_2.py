import numpy as np
from p5_5 import truncate_svd  # Assuming truncate_svd performs SVD and truncates based on some criterion

L = 6  # Number of sites
mps_tensors = []

# First tensor initialization
A_1_up = np.array([1, 2])
A_1_down = np.array([2, -1])
mps_tensors.append(np.stack((A_1_up, A_1_down), axis=0).reshape(1, 2, 2))

# Middle tensors
for site in range(1, L-1):
    A_ell_up = np.array([[1, 1], [0, 0]]) / np.sqrt(2)
    A_ell_down = np.array([[0, 0], [1, -1]]) / np.sqrt(2)
    mps_tensors.append(np.stack((A_ell_up, A_ell_down), axis=0))

# Last tensor
A_L_up = np.array([1, 3])
A_L_down = np.array([3, 1])
mps_tensors.append(np.stack((A_L_up, A_L_down), axis=0).reshape(2, 2, 1))

# Canonicalization process
for i in range(L-1):
    current_tensor = mps_tensors[i].reshape(-1, mps_tensors[i].shape[-1])  # Flatten the tensor for SVD
    u, s, vt = truncate_svd(current_tensor, min(current_tensor.shape))  # Perform SVD
    # print out the schmidt values at the cut
    print(f"Schmidt values between site {i+1} and site {i+2}: {s}")
    u = u.reshape(mps_tensors[i].shape[0], mps_tensors[i].shape[1], u.shape[-1])  # Reshape U
    mps_tensors[i] = u  # Store the left-canonical form of the tensor

    # Update the next tensor by absorbing the S part of SVD and the entire next tensor
    if i < L-1:
        next_tensor = mps_tensors[i+1].reshape(mps_tensors[i+1].shape[0], -1)
        mps_tensors[i+1] = np.dot(s @ vt, next_tensor).reshape(-1, mps_tensors[i+1].shape[1], mps_tensors[i+1].shape[2])

# The last tensor absorbs the remaining transformation
last_shape = mps_tensors[-1].shape
mps_tensors[-1] = np.dot(s @ vt, mps_tensors[-1].reshape(last_shape[0], -1)).reshape(last_shape)

def check_left_canonical(mps_tensors):
    for idx, tensor in enumerate(mps_tensors):
        # Reshape the tensor for easier multiplication
        # We consider the left and physical indices as 'rows' and the right index as 'columns'
        tensor_reshaped = tensor.reshape(-1, tensor.shape[-1])
        # Compute the product of the tensor and its conjugate transpose
        product = np.dot(tensor_reshaped.conj().T, tensor_reshaped)
        # Check if the product is close to the identity matrix
        identity = np.eye(tensor_reshaped.shape[-1])
        if np.allclose(product, identity):
            print(f"Tensor at site {idx+1} is left-canonical.")
        else:
            print(f"Tensor at site {idx+1} is NOT left-canonical.")

check_left_canonical(mps_tensors)
