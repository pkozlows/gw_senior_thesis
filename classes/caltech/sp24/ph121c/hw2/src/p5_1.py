import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.color import rgb2gray
import os



def load_and_convert_image(image_path):
    image = plt.imread(image_path)
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = image[:, :, :3]  # Drop the alpha channel
    if len(image.shape) == 3:  # Now, it should have only three channels
        image = rgb2gray(image)  # Convert it to grayscale
    return img_as_float(image)


# define the image paths
image_path = 'hw2/src/landscape.png'

# load and convert the image
image = load_and_convert_image(image_path)
print(image.shape)
# perform the svd on the image


def reconstruct_image(U, s, Vt, k):
    S_k = np.diag(s[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    # Print shapes for debugging
    print(f"U_k shape: {U_k.shape}, S_k shape: {S_k.shape}, Vt_k shape: {Vt_k.shape}")
    return np.dot(U_k, np.dot(S_k, Vt_k))


def calculate_frobenius_norms(image_path, ranks):
    image = load_and_convert_image(image_path)
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    originals = [reconstruct_image(U, s, Vt, k) for k in ranks]
    frobenius_norms = [np.linalg.norm(original - image, 'fro') for original in originals]
    return originals, frobenius_norms

smaller_dim = min(image.shape)
# the ranks go like smaller_dim/(2^i) for i in range(0, 8)
ranks = [smaller_dim // (2**i) for i in range(8)]
reconstructions, frobenius_norms = calculate_frobenius_norms(image_path, ranks)
# show the images corresponding to the reconstructions
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(reconstructions[i])
    ax.set_title(f'k = {ranks[i]}')
    ax.axis('off')
plt.savefig('reconstructions.png')

# plot the frobenius norms as a function of k
plt.figure(figsize=(10, 5))
plt.plot(ranks, frobenius_norms)
plt.xlabel('k')
plt.ylabel('Frobenius norm')
plt.title('Frobenius norm as a function of k')
plt.savefig('frobenius_norms.png')
