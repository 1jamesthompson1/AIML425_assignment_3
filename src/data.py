import jax.numpy as jnp
from jax import random
import jax
import time
from itertools import product


def generate_square(x, y, xx, yy, size=10):
    """
    Make a square with top-left corner at (x, y) with given size.
    """
    mask = (xx >= x) & (xx < x + size) & (yy >= y) & (yy < y + size)
    
    return mask.astype(jnp.float32)


def generate_circle(x, y, xx, yy, size=10):
    """
    Make a circle centered at (x, y) with given radius (size).
    """
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= size
    return mask.astype(jnp.float32)


def generate_triangle(x, y, xx, yy, size=10):
    """
    Make a upright isosceles trianlge with the base centered at (x, y + size//2) and height size.
    """
    
    in_triangle = (yy >= y) & (yy < y + size//1.2) & (xx >= x - (yy - y)) & (xx <= x + (yy - y))
    
    return in_triangle.astype(jnp.float32)

def valid_bounds(shape, size, dim):
    '''
    Returns the xmin, xmax, ymin, ymax for placing a shape of given size in a canvas of given dimensions.
    '''
    match shape:
        case "square" | 0:
            return (
                0, dim[0] - size,
                0, dim[1] - size
            )
        case "circle" | 1:
            return (
                size // 2, dim[0] - size // 2,
                size // 2, dim[1] - size // 2
            )
        case "triangle" | 2:
            return (
                size // 1.2, dim[0] - size // 1.2,
                size // 1.2, dim[1] - size // 1.2
            )
        case _:
            raise ValueError("Unknown shape")


def generate_database(num, key, size_bounds=(10, 10), dim=(28, 28)):
    start_time = time.time()

    # Pre-generate all random numbers
    key, shapekey, locationkey, sizekey = random.split(key, 4)
    shape_idx = random.randint(shapekey, (num,), 0, 3)
    
    location_bounds_list = jnp.array([valid_bounds(s, size_bounds[1], dim) for s in shape_idx])

    locations = random.randint(
        shape=(num, 2),
        key=locationkey,
        minval=location_bounds_list[:, [0, 2]],
        maxval=location_bounds_list[:, [1, 3]],
    )

    sizes = random.randint(sizekey, (num,), size_bounds[0], size_bounds[1] + 1)

    print(f"Random number generation took {time.time()- start_time:.2f} seconds")

    creation_start_time = time.time()

    # Define a single function that handles all shapes
    def generate_single_image(shape_idx, location, size):
        x, y = location
        
        # Create coordinate grids
        yy, xx = jnp.meshgrid(jnp.arange(dim[0]), jnp.arange(dim[1]), indexing="ij")

        return jax.lax.switch(
            shape_idx, 
            [
                lambda: generate_square(x, y, xx, yy, size),
                lambda: generate_circle(x, y, xx, yy, size),
                lambda: generate_triangle(x, y, xx, yy, size)
            ]
        )    

    # Use vmap to generate all images in parallel
    images = jax.vmap(generate_single_image)(shape_idx, locations, sizes)

    print(f"Image creation took {time.time() - creation_start_time:.2f} seconds")
    return images


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


def generate_all_possible_images(dim=(28,28), sizes=[10]):
    shapes = [generate_square, generate_circle, generate_triangle]
    images = []

    for i, shape in enumerate(shapes):
        for size in sizes:
            x_min, x_max, y_min, y_max = valid_bounds(i, size, dim)
            for x, y in product(range(int(x_min), int(x_max)), range(int(y_min), int(y_max))):
                img = shape(x, y, *jnp.meshgrid(jnp.arange(dim[0]), jnp.arange(dim[1]), indexing="ij"), size=size)
                images.append(img)
        
    return jnp.array(images)
