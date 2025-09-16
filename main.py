# %% [markdown]
# # AIML 425 - Assignment 3
# ## Problem 2: Variational Auto Encoders and Auto Encoders

# This notebook is provided for ease of use for marking. However it sohuld be noted that the development was conducted with the notebook as script percent format. The assignment repository can be found at [my gitea instance](https://gitea.james-server.duckdns.org/james/AIML425_assignment_3)  
# 
# Most of the interesting good stuff is in the `src/` directory this file just runs the experiments and call the implementations.

# ## Global imports
# %% Start up
from jax import random
from jax import numpy as jnp
from importlib import reload
from functools import partial

from src import model, train, data, inspect

# This is the main key used for all random operations.
key = random.key(42)

reload(model)
reload(train)
reload(data)
reload(inspect)

################################################################################
################################################################################

# ------------ Data generation -----------------
# %% [markdown]
################################################################################
# # Generate data

# %%
reload(data)
reload(inspect)

dim = (28, 28)
size_bounds = (7, 14)

database = data.generate_database(100, random.key(4), dim=dim, size_bounds=size_bounds)

inspect.vis_grid(database[:25])

all_possible_images, parameters = data.generate_all_possible_images(dim=dim, size_bounds=size_bounds)

num_possible = len(all_possible_images)

print(f"Number of possible images: {num_possible}, with shape {all_possible_images.shape}")



# Randomly sample 6,000 images from possible images to be training
key, subkey = random.split(key)
train_indices = random.choice(subkey, num_possible, shape=((9 * num_possible)//10,), replace=False)
key, subkey = random.split(subkey)
mask = jnp.ones(num_possible, dtype=bool).at[train_indices].set(False)
valid_indices = jnp.where(mask)[0]

print(f"Training on {len(train_indices)} images, validating on {len(valid_indices)} images")

train_batches = partial(data.create_batches, all_possible_images[train_indices])

valid_batches = partial(data.create_batches, all_possible_images[valid_indices])

# train_batches = partial(data.create_batches, data.generate_database(10000, key, dim=dim, size_bounds=size_bounds))

# valid_batches = partial(data.create_batches, data.generate_database(2000, key, dim=dim, size_bounds=size_bounds))

# %%

################################################################################
################################################################################

# ------------ VARIATIONAL AUTO ENCODER SECTION -----------------
# %% [markdown]
################################################################################
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
    loss_fn=partial(
        train.vae_loss_fn,
        kl_beta=0.65
    ),
    learning_rate=0.001,
    minibatch_size=512,
    latent_dim=32,
    encoder_arch=[1000, 1000, 1000, 500 ],
    decoder_arch=[500, 1000, 1000, 1000],
    num_epochs=1000,
    eval_every=20,
    dropout=0.1, 
)

inspect.plot_training_history(vae_history)
inspect.final_performance_information(vae_trained_model, all_possible_images, key)
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
# This is simply to help intuitively select the threshold for what counts as an attempt
inspect.visualize_neighbors(vae_trained_model, 15, all_possible_images, k=8, max_dist=50, rng_key=key, distance='euclidean')


################################################################################
################################################################################

# ------------ AUTO ENCODER SECTION -----------------
# %% [markdown]
# # Train an AutoEncoder model
################################################################################
################################################################################

# %%
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
    loss_fn=partial(
        train.ae_loss_fn,
        regularization_weight=0.1,
        mmd_sigma=(0.5, 1, 3, 5) # Use basic mean and variance control
    ),
    learning_rate=0.001,
    minibatch_size=64,
    latent_dim=10,
    encoder_arch=[1000, 1000, 500],
    decoder_arch=[500, 1000, 1000],
    num_epochs=500,
    eval_every=10,
    dropout=0.1,
)

inspect.plot_training_history(ae_history)
inspect.final_performance_information(ae_trained_model, all_possible_images, key)

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

# %% Estimate information rate

reload(inspect)

info_rate = inspect.estimate_information_rate(ae_trained_model, all_possible_images)

print(f"Estimated information rate: {info_rate:.4f} bits")
print(f"Which allows for {2**info_rate:.1f} distinct numbers to be represented, which is {2**info_rate/len(all_possible_images):.1f} numbers per possible image")
real_count = 2 ** jnp.ceil(jnp.log2(info_rate)).astype(int)
print(f"Next power of two: {real_count} bits or {real_count // 8} bytes")


# %% Explanation of what the latent space means for AE
# ## Explaining the AE latent space

# %%
# Correlation to known features
reload(inspect)

correlation = inspect.latent_space_correlation(ae_trained_model, all_possible_images, parameters)

inspect.visualize_latent_correlation(correlation, feature_names=["shape", "size", "x", "y"], name="ae-latent-corelation")

inspect.visualize_latent_by_category(ae_trained_model, all_possible_images, parameters[:, 0], name="ae-latent-by-shape")

# %%
# Varying latent dimensions and seeing the effect

reload(inspect)

inspect.zero_out_latent_and_reconstruct(
    ae_trained_model,
    all_possible_images[jnp.floor(jnp.linspace(0, len(all_possible_images)-1, 5)).astype(int)],
    name="ae-zero-out")

inspect.latent_space_traversal_and_reconstruct(
    ae_trained_model,
    all_possible_images[jnp.floor(jnp.linspace(0, len(all_possible_images)-1, 3)).astype(int)],
    traversal_range=(-3, 3),
    steps=7,
    name="ae-traversal")

# %% [markdown]

# # Comparing the two models

# To compare these two models I will simply run the final performane information function and put that in a table. It hives a good understanding of reocnsutrction error and some understanding of generative performaance.

# %%
reload(inspect)

inspect.create_comparison_table(vae_trained_model, ae_trained_model, all_possible_images, key, "ae-vs-vae")
    
