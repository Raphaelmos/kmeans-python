import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the image path
image_path = 'SILENCER.png'  # Ensure this path is correct

# Load the image
image = np.copy(mpimg.imread(image_path))

# Check if the image has an alpha channel and remove it if necessary
if image.shape[2] == 4:
    image = image[:, :, :3]  # Keep only the RGB channels

def init_centroids(num_clusters, image):
    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    centroids_init = np.empty([num_clusters, 3])

    for i in range(num_clusters):
        rand_row = random.randint(0, H - 1)  # Use H - 1 to prevent index out of range
        rand_col = random.randint(0, W - 1)  # Use W - 1 to prevent index out of range
        centroids_init[i] = image[rand_row, rand_col]

    return centroids_init

def update_centroids(centroids, image, max_iter=30):
    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for i in range(max_iter):
        centroid_rgbs = {}

        for row in range(H):
            for col in range(W):
                centroid = np.argmin(np.linalg.norm(centroids - image[row, col], axis=1))
                if centroid in centroid_rgbs:
                    centroid_rgbs[centroid] = np.append(centroid_rgbs[centroid], [image[row, col]], axis=0)
                else:
                    centroid_rgbs[centroid] = np.array([image[row, col]])

        for k in centroid_rgbs:
            centroids[k] = np.mean(centroid_rgbs[k], axis=0)

    return centroids

def update_image(image, centroids):
    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for row in range(H):
        for col in range(W):
            nearest_centroid = np.argmin(np.linalg.norm(centroids - image[row, col], axis=1))
            image[row, col] = centroids[nearest_centroid]

    return image

num_clusters = 16
initial_centroids = init_centroids(num_clusters, image)
final_centroids = update_centroids(initial_centroids, image, max_iter=30)
image_compressed = update_image(image, final_centroids)

# Display the compressed image
plt.imshow(image_compressed.astype(np.uint8))  # Ensure the image is in the correct format
plt.axis('off')  # Hide axis
plt.savefig(fname='sea_turtle_compressed.png', format='png', dpi=300)
plt.show()  # Show the image
