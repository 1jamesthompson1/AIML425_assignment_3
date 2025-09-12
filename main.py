# %% Start up
from flax import nnx
import matplotlib.pyplot as plt

from jax import random
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

database = data.generate_database(100, random.key(44), size_bounds=[10,10])

inspect.vis_grid(database[:25])

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

trained_model = nnx.merge(trained_state.graphdef, trained_state.params, trained_state.counts)

# %% Inspecting the output
reload(inspect)


batch = next(train_batches(key=key, minibatch_size=10))
inspect.visualize_reconstruction(trained_model, batch, rng_key=key, num_images=10)


# %% [markdown]
# Understanding the performance of the genearttion

# %%
reload(inspect)
generated_imgs = inspect.sample_and_generate(trained_model, latent_dim=32, num_samples=9, rng_key=key)

inspect.vis_grid(generated_imgs)

# %%
reload(inspect)
reload(data)

all_possible_images = data.generate_all_possible_images(sizes=[10])

print(f"Number of possible images: {len(all_possible_images)}, with shape {all_possible_images.shape}")


performance = inspect.final_generation_performance_evaluation(
    trained_model, training_data=all_possible_images, latent_dim=32, num_samples=100, rng_key=key
)

performance