# %% Start up
import jax
import jax.numpy as jnp

from jax import random, vmap
from importlib import reload
import utils
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

key = random.key(42)

reload(utils)

# %% [markdown]

# # Generate data

# The data generated will be images of size 28x28 of shapes at different locations. A single shape per image.

# %%
def vis(img):
    plt.imshow(img, cmap='gray')
    plt.grid(True, which='both')
    plt.show()

def generate_square(location, dim=(28,28), size=10):
    if location[0] + size//2 > dim[0] or\
        location[0] < 0 or\
        location[1] + size//4 > dim[1] or\
        location[1] < 0:
        raise ValueError("Triangle out of bounds")
    image = jnp.zeros(dim)
    x, y = location
    image = image.at[y:y+size, x:x+size].set(1)
    return image

def generate_circle(location, dim=(28,28), size=10):
    if location[0] + size//2 > dim[0] or\
        location[0] - size//2 < 0 or\
        location[1] + size//2 > dim[1] or\
        location[1] - size//2 < 0:
        raise ValueError("Circle out of bounds")
    x, y = location
    yy, xx = jnp.meshgrid(jnp.arange(dim[0]), jnp.arange(dim[1]), indexing="ij")
    mask = (xx - x)**2 + (yy - y)**2 <= size/2**2
    return mask.astype(jnp.float32)

def generate_triangle(location, dim=(28,28), size=10):
    if location[0] + size//2 > dim[0] or\
        location[0] - size//2 < 0 or\
        location[1] + size//4 > dim[1] or\
        location[1] - size//4 < 0:
        raise ValueError("Triangle out of bounds")
    
    image = jnp.zeros(dim)
    x, y = location

    for i in range(size // 2):
        image = image.at[i+x-size//4, -i+y:i+y+1].set(1)
    
    return image

def generate_database(num, key):
    shapes = [generate_square, generate_circle, generate_triangle]

    images = []
    
    while len(images) < num:

        to_add = num - len(images)
        # Pre-generate all random numbers
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        shape_idx = random.randint(subkey1, (to_add,), 0, len(shapes))
        locations = random.randint(subkey2, (to_add, 2), 0, 28)
        sizes = random.randint(subkey3, (to_add,), 2, 15)

        def make_image(idx, loc, size):
            try: 
                return shapes[idx](location=loc, size=size)
            except ValueError:
                return None

        for shape, location, size in zip(shape_idx, locations, sizes):
            img = make_image(shape, location, size)
            if img is not None:
                images.append(img)
    return images

# %%

database = generate_database(10000, key)

vis(database[23])

# %% [markdown]

# # Basic Variation Auto Encoder

# %%
