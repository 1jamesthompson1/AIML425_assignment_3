# This code will be to generate some graphs as well as performance test the model.
import matplotlib.pyplot as plt
from jax import random, numpy as jnp
import pandas as pd
import seaborn as sns
import math

def sample_and_generate(trained_model, num_samples=5, rng_key=None):
    '''
    Samples from the prior and generates images using the trained model.

    Will batch them into no more than 100 images
    '''
    batches = math.ceil(num_samples / 100)
    generated = []
    for i in range(batches):
        rng_key, subkey = random.split(rng_key)
        z = random.normal(subkey, (min(100, num_samples - i * 100), trained_model.latent_dim))
        generated.append(trained_model.generate(z))
    return jnp.concatenate(generated)

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

def calculate_distance(
    generated_images, reference_images, distance
):
    '''
    Calculate the distance between each generated image and all of the reference images.
    
    Args:
        generated_images: JAX array of shape (N, 28, 28) or (N, 784).
        reference_images: JAX array of shape (M, 28, 28) or (M, 784).
        distance: Distance metric to use ('euclidean' supported).

    Returns:
        JAX array of shape (N, M) containing the pairwise distances.
    '''
    generated_images = generated_images.reshape(generated_images.shape[0], -1)
    reference_images = reference_images.reshape(reference_images.shape[0], -1)

    
    batch_size = 100
    distances = []
    
    for i in range(0, generated_images.shape[0], batch_size):
        batch_end = min(i + batch_size, generated_images.shape[0])
        batch_generated = generated_images[i:batch_end]

        if distance == 'euclidean':
            # Compute distances for this batch
            diff = batch_generated[:, None, :] - reference_images[None, :, :]
            batch_distances = jnp.linalg.norm(diff, axis=-1)
        elif distance == 'hamming':
            # Binarize images at 0.5 threshold
            bin_generated = (batch_generated > 0.5).astype(jnp.float32)
            bin_reference = (reference_images > 0.5).astype(jnp.float32)
            # Compute Hamming distances
            diff = bin_generated[:, None, :] != bin_reference[None, :, :]
            batch_distances = jnp.sum(diff, axis=-1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        
        # Find min distance for each image in this batch
        distances.append(batch_distances)

    dists = jnp.concatenate(distances, axis=0)
    
    return dists

def visualize_neighbors(trained_model, num_images, training_data, show_histograms=False, k=5, max_dist=7, distance='euclidean', rng_key=None):
    '''
    Helper function to understand what euclidean distance means visually.

    This will show the input image and k neighbors from the closest to the 25th, quartile, median, 3rd quartile and furthest for each input image.

    Args:
        trained_model: The trained autoencoder model.
        num_images: The number of images to visualize.
        training_data: The training data to find neighbors from. Shape (M, 28, 28) or (M, 784).
        k: The number of neighbors to display.
        max_dist: The maximum distance to consider for neighbors.
        distance: Distance metric to use ('euclidean' supported).
        rng_key: JAX random key for reproducibility.

    Returns:
        None (displays plots)
    '''
    
    images = sample_and_generate(trained_model, num_images, rng_key)

    images_flat = images.reshape(images.shape[0], -1)
    training_data_flat = training_data.reshape(training_data.shape[0], -1)

    dists = calculate_distance(images_flat, training_data_flat, distance=distance)

    # Setup plots
    fig_neighbors, axes_neighbors = plt.subplots(num_images, k + 1, figsize=(10, 2 * num_images), squeeze=False)
    if show_histograms:
        fig_hist, axes_hist = plt.subplots(num_images, 1, figsize=(8, 4 * num_images), squeeze=False)


    for img_idx in range(num_images):
        image = images_flat[img_idx]

        # Only grab the indices of images within max_dist
        within_max_mask = dists[img_idx] <= max_dist
        dists_masked = dists[img_idx, within_max_mask]
        training_data_masked = training_data_flat[within_max_mask]

        if len(dists_masked) == 0:
            print(f"Image {img_idx+1}: No training images found within the specified max_dist.")
            # Display input image and blank neighbors
            ax_row = axes_neighbors[img_idx]
            ax_row[0].imshow(image.reshape(28, 28), cmap='gray')
            ax_row[0].set_title(f"Input {img_idx+1}")
            ax_row[0].axis('off')
            for i in range(k):
                ax_row[i+1].set_title("No neighbor")
                ax_row[i+1].axis('off')
            continue

        print(f"Image {img_idx+1}: Found {len(dists_masked)} training images within distance {max_dist}.")
        
        sorted_indices_masked = jnp.argsort(dists_masked)
        
        # Get indices for quantiles
        quantile_positions = jnp.linspace(0, len(sorted_indices_masked) - 1, k).astype(jnp.int32)
        selected_indices_masked = sorted_indices_masked[quantile_positions]

        # --- Plot Neighbors ---
        ax_row = axes_neighbors[img_idx]
        ax_row[0].imshow(image.reshape(28, 28), cmap='gray')
        ax_row[0].set_title(f"Input {img_idx+1}")
        ax_row[0].axis('off')

        for i, neighbor_idx in enumerate(selected_indices_masked):
            ax_row[i + 1].imshow(training_data_masked[neighbor_idx].reshape(28, 28), cmap='gray')
            ax_row[i + 1].set_title(f"Neighbor {i+1}\n(Dist: {dists_masked[neighbor_idx]:.2f})")
            ax_row[i + 1].axis('off')

        # --- Plot Distance Histogram ---
        if show_histograms:
            ax_hist = axes_hist[img_idx, 0]
            ax_hist.hist(dists_masked, bins=50, alpha=0.7, edgecolor='black')
            ax_hist.axvline(dists_masked[selected_indices_masked].min(), color='red', linestyle='--', label='Min distance')
            ax_hist.axvline(dists_masked[selected_indices_masked].max(), color='red', linestyle='--', label='Max distance')
            ax_hist.set_xlabel('Distance')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title(f'Distances for Input Image {img_idx+1}')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)

    fig_neighbors.tight_layout()
    if show_histograms:
        fig_hist.tight_layout()
    plt.show()


def coverage_estimation(trained_model, all_possible_images, num_samples=1000, rng_key=None, distance='euclidean', neighbor_threshold=3.0):
    '''
    Estimate the coverage of the generated images in the latent space.
    
    This is done by generating a number of images, binarizing them with a threshold, finding their nearest valid image in the training data (using threshold), and then calculating the proportion of unique images generated.

    Args:
        trained_model: The trained autoencoder model.
        all_possible_images: JAX array of shape (M, 28, 28) or (M, 784) containing all valid images.
        num_samples: Number of images to generate for the estimation. This should be the size of the valid possible images or more.
        rng_key: JAX random key for reproducibility.
        neighbor_threshold: Distance threshold to consider two images as the same.
    Returns:
        coverage: Proportion of unique images generated.
    '''
    generated = sample_and_generate(trained_model, num_samples, rng_key)

    all_flat = all_possible_images.reshape(all_possible_images.shape[0], -1)

    dists = calculate_distance(generated, all_flat, distance)
    
    within_threshold_mask = dists <= neighbor_threshold

    dists_masked = jnp.where(within_threshold_mask, dists, jnp.inf)

    nearest_indices = jnp.argmin(dists_masked, axis=1)

    found_neighbor_mask = jnp.any(within_threshold_mask, axis=1)

    unique_indices = jnp.unique(nearest_indices[found_neighbor_mask])
    
    # Compute coverage as the proportion of unique images generated
    coverage = len(unique_indices) / all_possible_images.shape[0]
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

    dists = calculate_distance(generated_imgs, training_data, distance=distance)
    
    min_dists = jnp.min(dists, axis=1)  # (N,)
    avg_distance = jnp.mean(min_dists)
    
    return avg_distance


def kl_divergence(data, trained_model, rng_key=None):
    '''
    Given the trained_model and the true data distribution, this will compute the KL divergence between the two distributions.
    The data provides the true distribution and the trained model provides the learned distribution.
    
    Args:
        data: The complete set of true data (N, 28, 28)
        trained_model: The trained autoencoder model.
        rng_key: JAX random key for reproducibility.
    '''
    generated = sample_and_generate(trained_model, data.shape[0], rng_key)
    generated = generated.reshape(generated.shape[0], -1)  # Flatten the generated

    data = data.reshape(data.shape[0], -1)  # Flatten the true data
    
    dists = calculate_distance(generated, data, distance='euclidean')

    assigned_nearests = jnp.argmin(dists, axis=1)  # (N,)

    # Count occurrences of each true data point being assigned
    counts = jnp.bincount(assigned_nearests, length=data.shape[0])

    p_x = 1 / data.shape[0]  # Uniform distribution over true data

    q_x = counts / generated.shape[0]  # Empirical distribution from generated data
    
    return jnp.sum(p_x * jnp.log2(p_x / (q_x + 1e-10)))
    


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


########################################
# Latent space correlation with known features
########################################

def latent_space_correlation(trained_model, all_images, known_features):
    '''
    This will compute the correlation between each latent dimension and each known feature.
    
    Args:
        trained_model: The trained auto encoder model.
        all_images: This should be all the images that would be input. (N, 28, 28)
        known_features: A DataFrame or array of known features corresponding to the input data. (N, num_features)
    '''
    flattended_images = all_images.reshape(all_images.shape[0], -1)  # (N, 784)
    
    # Encode the input batch to obtain the latent representations
    z = trained_model(flattended_images, z_rng=None, deterministic=True)[1]  # (N, z_dim)

    latent_dim = z.shape[1]
    num_features = known_features.shape[1]

    # Combine latent variables and known features
    combined_data = jnp.concatenate([z, known_features], axis=1)

    # Compute the full correlation matrix
    # rowvar=False means that each column represents a variable
    full_corr_matrix = jnp.corrcoef(combined_data, rowvar=False)

    # Extract the cross-correlation matrix between latent dims and features
    # This is the top-right block of the full correlation matrix
    cross_corr_matrix = full_corr_matrix[:latent_dim, latent_dim:]

    return cross_corr_matrix


def visualize_latent_correlation(correlation_matrix, latent_dim_names=None, feature_names=None):
    """
    Visualizes the correlation matrix between latent dimensions and known features using a heatmap.

    Args:
        correlation_matrix: A 2D JAX or NumPy array of shape (latent_dim, num_features).
        latent_dim_names: Optional list of names for the latent dimensions.
        feature_names: Optional list of names for the features.
    """
    if latent_dim_names is None:
        latent_dim_names = [f"Latent {i+1}" for i in range(correlation_matrix.shape[0])]
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(correlation_matrix.shape[1])]

    df_corr = pd.DataFrame(correlation_matrix, index=latent_dim_names, columns=feature_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation between Latent Dimensions and Known Features')
    plt.ylabel('Latent Dimensions')
    plt.xlabel('Known Features')
    plt.show()

def visualize_latent_by_category(trained_model, all_images, shape):
    """
    Visualizes the distribution of each latent dimension, grouped by a categorical feature.

    Args:
        trained_model: The trained autoencoder model.
        all_images: All images to be encoded. (N, 28, 28)
        shape: An array (N,) indicating the shape category for each image (0: square, 1: circle, 2: triangle).
    """
    flattended_images = all_images.reshape(all_images.shape[0], -1)
    z = trained_model(flattended_images, z_rng=None, deterministic=True)[1]

    latent_df = pd.DataFrame(z, columns=[f"Latent {i+1}" for i in range(z.shape[1])])
    
    # Map shape index to a name for better plotting
    shape_map = {0: 'Square', 1: 'Circle', 2: 'Triangle'}
    shape_names = [shape_map[int(s)] for s in shape]

    # Melt the DataFrame for plotting with seaborn
    melted_df = latent_df.copy()
    melted_df['shape_name'] = shape_names
    melted_df = melted_df.melt(id_vars=['shape_name'], var_name='Latent Dimension', value_name='Value')

    # Create the plot
    g = sns.catplot(
        data=melted_df,
        x='Value',
        y='Latent Dimension',
        hue='shape_name',
        kind='violin',
        orient='h',
        height=max(6, z.shape[1] * 0.5),
        aspect=1.5,
        inner='quartile',
        split=True,
        palette='viridis'
    )
    g.fig.suptitle('Latent Dimension Distributions by Shape', y=1.02)
    plt.tight_layout()
    plt.show()

########################################
# Modifying the latent space and seeing how the reconstruction changes
########################################

