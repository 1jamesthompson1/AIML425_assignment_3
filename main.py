# %% Start up
import jax
import jax.numpy as jnp

import jax.random as random
from importlib import reload
import utils
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

key = random.key(42)

reload(utils)

jax.config.update('jax_platform_name', 'gpu')

# %% [markdown]

