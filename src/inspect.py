# This code will be to generate some graphs as well as performance test the model.
import matplotlib.pyplot as plt
from jax import random, numpy as jnp
import pandas as pd
import seaborn as sns
import math

def sample_and_generate(trained_model, num_samples=5, rng_key=None):
    '''
    Samples from the prior and generates images using the trained model.
    '''
    z = random.normal(rng_key, (num_samples, trained_model.latent_dim))
    generated = trained_model.generate(z)
    return generated

################################################################################
# 
# ---------- Simple visualization function to see the outputs ----------
#
################################################################################
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

def visualize_latent_space(trained_model, batch):
    '''
    Create a violin plot for each dimension in the latent space to visualize the distribution of latent variables. They are stacked vertically
    '''
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

def plot_training_history(history):
    '''
    Plot the train and val loss over epochs.
    Will make three plots one for regular loss, and one each for the loss_recon and loss_reg components.    Will make three plots one for regular loss, and one each for the loss_recon and loss_reg components.
    
    Args:
        history: A dictionary with keys 'train_loss' and 'val_loss', each containing a list of loss values per epoch.

    Returns:
        None (displays plots)
    '''

    plt.figure(figsize=(12, 8))

    # Plot training losses
    plt.plot(history['train_epochs'], history['train_loss'], label='Train Total Loss', color='blue')
    plt.plot(history['train_epochs'], history['train_loss_recon'], label='Train Recon Loss', linestyle='--', color='green')
    plt.plot(history['train_epochs'], history['train_loss_reg'], label='Train Reg Loss', linestyle='--',color='red')

    # Plot validation losses
    plt.plot(history['val_epochs'], history['val_loss'], label='Val Total Loss', color='cyan')
    plt.plot(history['val_epochs'], history['val_loss_recon'], label='Val Recon Loss', linestyle='--', color='lime')
    plt.plot(history['val_epochs'], history['val_loss_reg'], label='Val Reg Loss', linestyle='--', color='magenta')

    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


    

################################################################################
# 
# ---------- Measuring the generative performance of the model ----------
#
################################################################################

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

def estimate_information_rate(trained_model, batch, rng_key=None, num_bins=30):
    x = batch["input"]
    z = trained_model(x, z_rng=None, deterministic=True)[1]  # (N, z_dim)

    information = 1/2 * jnp.sum(jnp.log2(1 + (jnp.var(z, axis=0) / trained_model.latent_noise_scale)), axis=0)

    return information
    

################################################################################
# 
# ---------- Understanding the information that the reconstruction  ----------
# ---------- uses from the latent space                             ----------
#
################################################################################

# Ideas
# Check for correlation between known features and latent dimensions
# Change the latent dimensions and see how the reconsutrction changes
# Zero out latent dimensions and see how reconstruction changes similar to above.
