import numpy as np
from p5_5 import truncate_svd

# Define MPS matrices for a chain
L = 6  # Number of sites
mps_tensors = []

# # Initial tensor setup
# A_1_up = np.array([1, 1]) / np.sqrt(2)
# A_1_down = np.array([1, -1]) / np.sqrt(2)
# mps_tensors.append(np.stack((A_1_up, A_1_down), axis=0).reshape(1, 2, 2))  # First tensor

# initial tensor setup for second part
A_1_up = np.array([1, 2])
A_1_down = np.array([2, -1])
mps_tensors.append(np.stack((A_1_up, A_1_down), axis=0).reshape(1, 2, 2))  # First tensor

# Middle tensors
for site in range(1, L-1):
    A_ell_up = np.array([[1, 1], [0, 0]]) / np.sqrt(2)
    A_ell_down = np.array([[0, 0], [1, -1]]) / np.sqrt(2)
    mps_tensors.append(np.stack((A_ell_up, A_ell_down), axis=0))

# # Last tensor
# A_L_up = np.array([1, 0])
# A_L_down = np.array([0, 1])
# mps_tensors.append(np.stack((A_L_up, A_L_down), axis=0).reshape(2, 2, 1))  # Last tensor

# last tensor for second part
A_L_up = np.array([1, 3])
A_L_down = np.array([3, 1])
mps_tensors.append(np.stack((A_L_up, A_L_down), axis=0).reshape(2, 2, 1))  # Last tensor


# Display tensor shapes
for idx, tensor in enumerate(mps_tensors):
    print(f"Site {idx+1}, Tensor Shape: {tensor.shape}")

# Calculate Schmidt values at each cut
for cut in range(1, L):
    # Contract tensors to the left of the cut
    left_state = mps_tensors[0].reshape(-1, 2)  # Start with the first tensor reshaped for matrix multiplication
    # first make a loop from 1 to cut - 1
    for i in range(1, cut):
        left_state = np.einsum('ab,bcd->acd', left_state, mps_tensors[i])
        left_state = left_state.reshape(-1, mps_tensors[i].shape[2])  # Ensure correct shape

    # Contract tensors to the right of the cut
    right_state = mps_tensors[cut].reshape(2, -1)  # Start with the tensor at the cut
    # now make a lope from cut+1 to L-1
    for j in range(cut + 1, L):
        right_state = np.einsum('abc,ad->cbd', mps_tensors[j], right_state)
        right_state = right_state.reshape(mps_tensors[j].shape[0], -1)  # Ensure correct shape

    # Compute the singular values directly from reshaped left and right states
    schmidt_matrix = np.dot(left_state, right_state)
    u, s, vt = truncate_svd(schmidt_matrix, 2)  # Adjust the truncation
    print(f"Cut {cut}: {s}")
