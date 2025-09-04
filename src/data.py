import jax.numpy as jnp
from jax import random
import jax


def generate_square(location, dim=(28, 28), size=10):
    if (
        location[0] + size > dim[0]
        or location[0] < 0
        or location[1] + size > dim[1]
        or location[1] < 0
    ):
        raise ValueError("Triangle out of bounds")
    image = jnp.zeros(dim)
    x, y = location
    image = image.at[y : y + size, x : x + size].set(1)
    return image


def generate_circle(location, dim=(28, 28), size=10):
    if (
        location[0] + size // 2 > dim[0]
        or location[0] - size // 2 < 0
        or location[1] + size // 2 > dim[1]
        or location[1] - size // 2 < 0
    ):
        raise ValueError("Circle out of bounds")
    x, y = location
    yy, xx = jnp.meshgrid(jnp.arange(dim[0]), jnp.arange(dim[1]), indexing="ij")
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= size / 2**2
    return mask.astype(jnp.float32)


def generate_triangle(location, dim=(28, 28), size=10):
    if (
        location[0] + size // 2 > dim[0]
        or location[0] - size // 2 < 0
        or location[1] + size // 4 > dim[1]
        or location[1] - size // 4 < 0
    ):
        raise ValueError("Triangle out of bounds")

    image = jnp.zeros(dim)
    x, y = location

    for i in range(size // 2):
        image = image.at[i + x - size // 4, -i + y : i + y + 1].set(1)

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
    return jnp.array(images)


def create_batches(x_data, minibatch_size, key=None):
    x_data_flatten = x_data.reshape((x_data.shape[0], -1))

    x_data = jax.device_put(x_data_flatten)

    n_samples = x_data.shape[0]
    if key is not None:
        indices = jnp.arange(n_samples)
        shuffled_indices = random.permutation(key, indices)
        x_data = x_data[shuffled_indices]
    if x_data.ndim == 1:
        x_data = x_data[:, None]
    n_batches = n_samples // minibatch_size
    for i in range(n_batches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        batch = {
            "input": x_data[start_idx:end_idx],
        }
        yield batch
    if n_samples % minibatch_size != 0:
        start_idx = n_batches * minibatch_size
        batch = {
            "input": x_data[start_idx:],
        }
        yield batch
