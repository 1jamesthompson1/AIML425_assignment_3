# This code will be to generate some graphs as well as performance test the model.
import matplotlib.pyplot as plt
from jax import random, numpy as jnp
import math

def vis_grid(images, grid_shape=None):
    num_images = len(images)
    if grid_shape is None:
        # Calculate grid shape to fit all images
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))
        grid_shape = (rows, cols)
    
    # Calculate figsize, capping at 10 width and 15 height
    width_per_subplot = 2
    height_per_subplot = 2
    figsize = (min(10, grid_shape[1] * width_per_subplot), min(15, grid_shape[0] * height_per_subplot))
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape((28,28)), cmap='gray')
            ax.grid(True, which='both')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def sample_and_generate(trained_model, latent_dim, num_samples=5, rng_key=None):
    z = random.normal(rng_key, (num_samples, latent_dim))
    generated = trained_model.generate(z)
    return generated

def visualize_reconstruction(trained_model, batch, rng_key=None, num_images=5):
    x = batch["input"][:num_images]
    if rng_key is None:
        rng_key = random.PRNGKey(1)
    recon_x, _, _ = trained_model(x, rng_key)
    fig, axes = plt.subplots(num_images, 2, figsize=(5, 2 * num_images))
    for i in range(num_images):
        axes[i, 0].imshow(x[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(recon_x[i].reshape(28, 28), cmap='gray')
        axes[i, 1].set_title("Reconstruction")
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    
def final_generation_performance_evaluation(
        trained_model, training_data, latent_dim, num_samples=100, rng_key=None, distance='euclidean'
    ):
    '''
    This will get the final performance of generations.
    
    It will do this by finding the closest distance between generated images and any theoretical image possible in the training data.
    
    The distance is currently euclidean. However instead it could be changed to other metrics like using athreshold to mark a pixel as on or off then calculate hamming distance. Or instead use a threshold to mark a match or not and count the number of unknowns.
    
    It will then average these distances to get a final performance metric.
    '''
    
    generated_imgs = sample_and_generate(trained_model, latent_dim, num_samples, rng_key)
    
    # Flatten training data: (samples, 28, 28) -> (samples, 784)
    training_flat = training_data.reshape(training_data.shape[0], -1)

    
    min_distances = []
    for gen_img in generated_imgs:
        distances = None

        match distance:
            case 'euclidean':
                distances = jnp.linalg.norm(training_flat - gen_img, axis=1)
            case _:
                raise ValueError(f"Unsupported distance metric: {distance}")
        
        min_distances.append(jnp.min(distances))
    
    # Average the minimum distances
    avg_distance = jnp.mean(jnp.array(min_distances))

    return avg_distance

