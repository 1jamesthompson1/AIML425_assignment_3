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
reload(model)

trained_state = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    learning_rate=0.001,
    minibatch_size=256,
    latent_dim=32,
    num_epochs=400,
    eval_every=5,
)

# %% Inspecting the output

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

trained_model = nnx.merge(trained_state.graphdef, trained_state.params, trained_state.counts)

def sample_and_generate(trained_model, latent_dim, num_samples=5, rng_key=None):
    z = jax.random.normal(rng_key, (num_samples, latent_dim))
    generated = trained_model.generate(z)
    return generated

def visualize_reconstruction(trained_model, batch, rng_key=None, num_images=5):
    x = batch["input"][:num_images]
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)
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

generated_imgs = sample_and_generate(trained_model, latent_dim=32, num_samples=5, rng_key=key)
for img in generated_imgs:
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()

batch = next(train_batches(key=key, minibatch_size=5))
visualize_reconstruction(trained_model, batch, rng_key=key, num_images=5)