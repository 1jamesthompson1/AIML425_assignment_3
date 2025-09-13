# %% Start up
from jax import random
from jax import numpy as jnp
from importlib import reload
import utils
from functools import partial

from src import model, train, data, inspect

key = random.key(42)

reload(utils)
reload(model)
reload(train)
reload(data)
reload(inspect)

# %% [markdown]

# # Generate data

# The data generated will be images of size 28x28 of shapes at different locations. A single shape per image. Assuming black and white so no anti-aliasing.

# %%
reload(data)
reload(inspect)

dim = (28, 28)
size_bounds = (7, 14)

database = data.generate_database(100, random.key(4), dim=dim, size_bounds=size_bounds)

inspect.vis_grid(database[:25])

all_possible_images = data.generate_all_possible_images(dim=dim, size_bounds=size_bounds)

print(f"Number of possible images: {len(all_possible_images)}, with shape {all_possible_images.shape}")

train_batches = partial(data.create_batches, data.generate_database(10000, key, dim=dim, size_bounds=size_bounds))

valid_batches = partial(data.create_batches, data.generate_database(2000, key, dim=dim, size_bounds=size_bounds))

# %% [markdown]

# # Train a VAE model

# %%
reload(train)
reload(data)
reload(model)
reload(inspect)

vae_trained_model, vae_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.VAE,
    loss_fn=train.vae_loss_fn,
    learning_rate=0.001,
    minibatch_size=256,
    latent_dim=32,
    # encoder_arch=[2000, 2000, 2000],
    # decoder_arch=[2000, 2000, 2000],
    num_epochs=200,
    eval_every=5,
    # dropout=0.2,
)

inspect.plot_training_history(vae_history)

# %% [markdown]
# ## Understand the performance of the model

# %% Understand latent space distribution

reload(inspect)
inspect.visualize_latent_space(vae_trained_model, next(train_batches(key=key, minibatch_size=1000)))

# %% Inspecting the output
reload(inspect)


batch = next(train_batches(key=key, minibatch_size=10))
inspect.visualize_reconstruction(vae_trained_model, batch, rng_key=key, num_images=10)


# %% [markdown]
# Understanding the performance of the generation

# %%
reload(inspect)
generated_imgs = inspect.sample_and_generate(vae_trained_model, num_samples=9, rng_key=key)

inspect.vis_grid(generated_imgs)

# %%
reload(inspect)

performance = inspect.nearest_neighbor_performance_evaluation(
    vae_trained_model, training_data=all_possible_images, num_samples=1000, rng_key=key
)

print(f"Nearest neighbor performance evaluation: {performance:.4f}")

coverage = inspect.coverage_estimation(vae_trained_model, num_samples=10000, rng_key=key)
print(f"Coverage estimate: {coverage:.4f}")

# %% [markdown]
# # Train an AutoEncoder model

reload(train)
reload(data)
reload(model)
reload(inspect)

ae_trained_model, ae_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.AutoEncoder,
    model_kwargs={
        "latent_noise_scale": 0.1
    },
    loss_fn=train.ae_loss_fn,
    learning_rate=0.001,
    minibatch_size=256,
    latent_dim=10,
    # encoder_arch=[2000, 2000, 2000],
    # decoder_arch=[2000, 2000, 2000],
    num_epochs=100,
    eval_every=10,
    # dropout=0.2,
)

inspect.plot_training_history(ae_history)

# %% [markdown]
# ## Understand the performance of the model

# %% Visualize latent space

reload(inspect)
inspect.visualize_latent_space(ae_trained_model, next(train_batches(key=key, minibatch_size=1000)))


# %% Inspecting the output
reload(inspect)


batch = next(train_batches(key=key, minibatch_size=10))
inspect.visualize_reconstruction(ae_trained_model, batch, rng_key=key, num_images=10)


# %% [markdown]
# Understanding the performance of the generation

# %%
reload(inspect)
generated_imgs = inspect.sample_and_generate(ae_trained_model, num_samples=9, rng_key=key)

inspect.vis_grid(generated_imgs)

# %%
reload(inspect)
reload(data)

performance = inspect.nearest_neighbor_performance_evaluation(
    ae_trained_model, training_data=all_possible_images, num_samples=1000, rng_key=key
)

print(f"Nearest neighbor performance evaluation: {performance:.4f}")

# %% Estimate information rate

reload(inspect)

info_rate = inspect.estimate_information_rate(ae_trained_model, next(train_batches(key=key, minibatch_size=1000)))

print(f"Estimated information rate: {info_rate:.4f} bits")
print(f"Which allows for {2**info_rate:.1f} distinct numbers to be represented, which is {2**info_rate/len(all_possible_images):.1f} numbers per possible image")
real_count = 2 ** jnp.ceil(jnp.log2(info_rate)).astype(int)
print(f"Next power of two: {real_count} bits or {real_count // 8} bytes")


# %% Explanation of what the latent space means for AE

