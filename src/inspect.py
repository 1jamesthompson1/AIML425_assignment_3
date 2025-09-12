# This code will be to generate some graphs as well as performance test the model.
import matplotlib.pyplot as plt
from jax import random, numpy as jnp
import pandas as pd
import seaborn as sns
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

def sample_and_generate(trained_model, num_samples=5, rng_key=None):
    z = random.normal(rng_key, (num_samples, trained_model.latent_dim))
    generated = trained_model.generate(z)
    return generated

def visualize_reconstruction(trained_model, batch, rng_key=None, num_images=5):
    x = batch["input"][:num_images]
    if rng_key is None:
        rng_key = random.PRNGKey(1)
    recon_x, *_ = trained_model(x, rng_key)
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

def visualize_latent_space(trained_model, batch, group_size=16):
    
    # x = batch["input"]
    # z = trained_model(x, z_rng=None, deterministic=True)[1]

    # z_dim = z.shape[1]
    # fig, axes = plt.subplots(1, z_dim, figsize=(5 * z_dim, 5))
    # for i in range(z_dim):
    #     axes[i].hist(z[:, i], bins=50, density=True)
    #     axes[i].set_title(f"Latent Dimension {i + 1}")
    #     axes[i].grid(True)

    # plt.tight_layout()
    # plt.show()
    x = batch["input"]
    z = trained_model(x, z_rng=None, deterministic=True)[1]  # (N, z_dim)

    z_dim = z.shape[1]

    # Put into tidy DataFrame
    df = pd.DataFrame(z, columns=[f"dim_{i+1}" for i in range(z_dim)])
    df_long = df.melt(var_name="dimension", value_name="value")


    plt.figure(figsize=(8, 0.5*len(df_long['dimension'].unique())))
    sns.violinplot(
        data=df_long,
        x="value", y="dimension",
        scale="width", inner=None,
        orient="h", linewidth=0
    )
    plt.tight_layout()
    plt.show()

def coverage_estimation(trained_model, num_samples=1000, rng_key=None, threshold=0.5):
    generated = sample_and_generate(trained_model, num_samples, rng_key)
    binary_generated = (generated > threshold).astype(jnp.float32)
    unique_images = jnp.unique(binary_generated, axis=0)
    coverage = unique_images.shape[0] / num_samples
    return coverage

def nearest_neighbor_performance_evaluation(
        trained_model, training_data, num_samples=100, rng_key=None, distance='euclidean'
    ):
    '''
    This will get the final performance of generations.
    
    It will do this by finding the closest distance between generated images and any theoretical image possible in the training data.
    
    The distance is currently euclidean. However instead it could be changed to other metrics like using a threshold to mark a pixel as on or off then calculate hamming distance. Or instead use a threshold to mark a match or not and count the number of unknowns.
    
    It will then average these distances to get a final performance metric.
    '''
    
    generated_imgs = sample_and_generate(trained_model, num_samples, rng_key)
    
    # Flatten both datasets: (samples, 28, 28) -> (samples, 784)
    training_flat = training_data.reshape(training_data.shape[0], -1)
    generated_flat = generated_imgs.reshape(generated_imgs.shape[0], -1)
    
    # Compute pairwise distances in a vectorized way
    if distance == 'euclidean':
        # Expand dims for broadcasting: (num_gen, 1, 784) - (1, num_train, 784) -> (num_gen, num_train, 784)
        diff = generated_flat[:, None, :] - training_flat[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)  # Shape: (num_gen, num_train)
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")
    
    # Find min distance for each generated image and average
    min_distances = jnp.min(distances, axis=1)
    avg_distance = jnp.mean(min_distances)
    
    return avg_distance

    return avg_distance

