# %% Start up
import importlib
import jax.numpy as jnp
from flax import nnx

from jax import random, vmap
from importlib import reload
import utils
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

from src import model, train, data

key = random.key(42)

reload(utils)
reload(model)
reload(train)
reload(data)

# %% [markdown]

# # Generate data

# The data generated will be images of size 28x28 of shapes at different locations. A single shape per image.

# %%
reload(data)
def vis(img):
    plt.imshow(img, cmap='gray')
    plt.grid(True, which='both')
    plt.show()


# database = data.generate_database(1000, key)

# vis(database[23])

train_batches = partial(data.create_batches, data.generate_database(10000, key))

valid_batches = partial(data.create_batches, data.generate_database(2000, key))

# %% [markdown]

# Train model with training loop and database

# %%
reload(train)
reload(data)

trained_state = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    learning_rate=0.001,
    minibatch_size=256,
    num_epochs=50,
    eval_every=5,
)

# %% Inspecting the output

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Assume trained_state is returned from do_complete_experiment
model = nnx.merge(trained_state.graphdef, trained_state.params, trained_state.counts)

# 1. Generate new images from random latent vectors
def sample_and_generate(model, latent_dim, num_samples=5, rng_key=None):
    if rng_key is None:
        rng_key = j# %% [markdown]ax.random.PRNGKey(0)
    z = jax.random.normal(rng_key, (num_samples, latent_dim))
    generated = model.generate(z)
    return generated

# 2. Visualize reconstruction error
def visualize_reconstruction(model, batch, rng_key=None, num_images=5):
    x = batch["input"][:num_images]
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)
    recon_x, _, _ = model(x, rng_key)
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

# Example usage:
# Generate images
generated_imgs = sample_and_generate(model, latent_dim=500, num_samples=5)
for img in generated_imgs:
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()

# Visualize reconstruction
batch = next(train_batches(key=key, minibatch_size=5))
visualize_reconstruction(model, batch, rng_key=key, num_images=5)