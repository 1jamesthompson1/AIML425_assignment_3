import jax.numpy as jnp
from jax import random
import jax
import time
import math
from itertools import product


def generate_square(x, y, xx, yy, size=10):
    """
    Make a square with top-left corner at (x, y) with given size.
    """
    mask = (xx >= x) & (xx < x + size) & (yy >= y) & (yy < y + size)
    
    return mask.astype(jnp.float32)


# # No longer using this generate function as it doesn't have anti-aliasing so the edges are very jagged.
# def generate_circle(x, y, xx, yy, size=10):
#     """
#     Make a circle centered at (x, y) with given radius (size).
#     """
#     mask = (xx - x) ** 2 + (yy - y) ** 2 <= (size//2) ** 2
#     return mask.astype(jnp.float32)

def generate_circle(x, y, xx, yy, size=10, subsamples=4):
    """
    Make a circle centered at (x, y) with given diameter (size).
    Uses supersampling to get smoother edges.
    """
    radius = size / 2.0
    
    # Make fractional grid offsets for subpixel sampling
    offsets = jnp.linspace(-0.5, 0.5, subsamples)
    xv, yv = jnp.meshgrid(offsets, offsets, indexing="ij")

    # Expand grid to subsamples
    xx_sub = xx[..., None, None] + xv
    yy_sub = yy[..., None, None] + yv

    # Compute distances from center
    dist = jnp.hypot(xx_sub - x, yy_sub - y)
    mask = (dist <= radius).astype(jnp.float32)

    # print(f"Message to understand what is going on: {mask.shape=}, {mask.mean(axis=(-1, -2)).shape=}\n\nxv={xv.shape}, yv={yv.shape} and {xx_sub.shape=}, {yy_sub.shape=}")
    
    return (mask.mean(axis=(-1, -2)) > 0.5).astype(jnp.float32)

def generate_triangle(x, y, xx, yy, size=10):
    """
    Make a upright isosceles triangle with the base centered at (x, y + height) where height = size * 5 // 6. Made  slightly shorter to better fit the overall size of the image.
    """
    height = size * 5 // 6
    in_triangle = (yy >= y) & (yy < y + height) & (xx >= x - (yy - y)) & (xx <= x + (yy - y))
    
    return in_triangle.astype(jnp.float32)


def create_location_bounds(shape_idx, sizes, dim):
    '''
    vectorized way of creating valid_bounds.
    
    Args:
        shape_idx: Array of shape indices (0 for square, 1 for circle, 2 for triangle).
        sizes: Array of sizes corresponding to each shape.
        dim: Tuple (height, width) representing the dimensions of the canvas.

    Returns:
        A JAX array of shape (N, 4) where N is the number of shapes, and each row contains (xmin, xmax, ymin, ymax) for the corresponding shape.

    '''
    
    is_square = shape_idx == 0
    is_circle = shape_idx == 1
    is_triangle = shape_idx == 2
    
    # For squares
    square_xmin = jnp.zeros_like(sizes)
    square_xmax = dim[0] - sizes + 1
    square_ymin = jnp.zeros_like(sizes)
    square_ymax = dim[1] - sizes + 1
    
    # For circles
    radii = sizes / 2
    circle_xmin = jnp.floor(radii)
    circle_xmax = dim[0] - jnp.ceil(radii)
    circle_ymin = jnp.floor(radii)
    circle_ymax = dim[1] - jnp.ceil(radii)
    
    # For triangles
    heights = sizes * 5 // 6
    triangle_xmin = heights
    triangle_xmax = dim[0] - heights
    triangle_ymin = jnp.zeros_like(sizes)
    triangle_ymax = dim[1] - heights
    
    # Select bounds based on shape_idx
    xmin = jnp.where(is_square, square_xmin, 
                     jnp.where(is_circle, circle_xmin, triangle_xmin))
    xmax = jnp.where(is_square, square_xmax, 
                     jnp.where(is_circle, circle_xmax, triangle_xmax))
    ymin = jnp.where(is_square, square_ymin, 
                     jnp.where(is_circle, circle_ymin, triangle_ymin))
    ymax = jnp.where(is_square, square_ymax, 
                     jnp.where(is_circle, circle_ymax, triangle_ymax))
    
    return jnp.stack([xmin, xmax, ymin, ymax], axis=1)

def generate_database(num, key, size_bounds, dim):
    '''
    Generate a database of images containing random shapes (squares, circles, triangles) at random locations and sizes.
    
    Args:
        num: Number of images to generate.
        key: JAX random key.
        size_bounds: Tuple (min_size, max_size) for the size of the shapes, inclusive.
        dim: Tuple (height, width) for the dimensions of the images.
    Returns:
        A JAX array of shape (num, dim[0], dim[1]) containing the generated images.

    '''
    start_time = time.time()

    # Pre-generate all random numbers
    key, shapekey, locationkey, sizekey = random.split(key, 4)
    shape_idx = random.randint(shapekey, (num,), 0, 3)
    sizes = random.randint(sizekey, (num,), size_bounds[0], size_bounds[1] + 1)
    
    location_bounds_start = time.time()
    location_bounds_list = create_location_bounds(shape_idx, sizes, dim)

    print(f"Location bounds calculation took {time.time() - location_bounds_start:.2f} seconds")
    locations = random.randint(
        shape=(num, 2),
        key=locationkey,
        minval=location_bounds_list[:, [0, 2]],
        maxval=location_bounds_list[:, [1, 3]],
    )


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


def create_batches(x_data, minibatch_size, key):
    '''
    Create mini-batches from the input data. It yields a generator that produces batches of the specified size.
    
    Args:
        x_data: JAX array of shape (num_samples, height, width) or (num_samples, features).
        minibatch_size: Size of each mini-batch.
        key: JAX random key for shuffling the data.
    
    Yields:
        A dictionary with key "input" containing a mini-batch of data.
    '''
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


def generate_all_possible_images(dim, size_bounds):
    '''
    Uses the valid_bounds and shape generation functions to generate all possible images of the given sizes and dimensions.
    
    Args:
        dim: Tuple (height, width) for the dimensions of the images.
        size_bounds: Tuple (min_size, max_size) for the size of the shapes.
    Returns:
        A JAX array of shape (num_images, dim[0], dim[1]) containing all possible generated images.
    '''
    shapes = [generate_square, generate_circle, generate_triangle]

    yy, xx = jnp.meshgrid(jnp.arange(dim[0]), jnp.arange(dim[1]), indexing="ij")

    # all (shape_idx, size) pairs
    possible_values = jnp.array(
        list(product(range(len(shapes)), range(size_bounds[0], size_bounds[1] + 1)))
    )
    shape_idx = possible_values[:, 0]
    sizes = possible_values[:, 1]

    # bounds for each shape/size
    bounds = create_location_bounds(shape_idx, sizes, dim)  # (N, 4)

    # expand each bound into all valid (x,y) locations
    def expand_locations(bound):
        xmin, xmax, ymin, ymax = bound.astype(int)
        xs = jnp.arange(xmin, xmax+1)
        ys = jnp.arange(ymin, ymax+1)
        xv, yv = jnp.meshgrid(xs, ys, indexing="ij")
        return jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # (num_locs, 2)

    locations_per_config = [expand_locations(b) for b in bounds]

    # Flatten into one big (M, 3) array: (shape_idx, size, (x,y))
    shape_repeated = []
    size_repeated = []
    locs_all = []
    for i, locs in enumerate(locations_per_config):
        shape_repeated.append(jnp.full((locs.shape[0],), shape_idx[i]))
        size_repeated.append(jnp.full((locs.shape[0],), sizes[i]))
        locs_all.append(locs)

    shape_repeated = jnp.concatenate(shape_repeated)
    size_repeated = jnp.concatenate(size_repeated)
    locs_all = jnp.concatenate(locs_all, axis=0)

    # Render function
    def render(shape_idx, loc, size):
        x, y = loc
        return jax.lax.switch(
            shape_idx,
            [
                lambda: generate_square(x, y, xx, yy, size),
                lambda: generate_circle(x, y, xx, yy, size),
                lambda: generate_triangle(x, y, xx, yy, size)
            ]
        )

    images = jax.vmap(render)(shape_repeated, locs_all, size_repeated)

    print(f"Generated {images.shape[0]} images of shape {dim}")
    return images