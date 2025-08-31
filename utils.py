import jax.numpy as jnp
import jax
import jax.random as random
from flax import nnx
from flax.training import train_state 
import matplotlib.pyplot as plt
from functools import partial
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

# Plotting function
def plot_data(
    X, Y, 
    x_hist=False,
    plot_names = ["X Data", "Y Data"],
    Y_names = ["target", "generated"],
    ):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    n = X.shape[0]
    colors = plt.cm.get_cmap('viridis')(jnp.linspace(0, 1, n))
    if x_hist:
        axs[0].hist(X.flatten(), bins=30, color='skyblue', alpha=0.7)
        axs[0].set_title(plot_names[0])
        axs[0].set_xlabel('X values (flattened)')
        axs[0].set_ylabel('Frequency')
    else:
        axs[0].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
        axs[0].set_title(plot_names[0])
        axs[0].set_xlabel('X1')
        axs[0].set_ylabel('X2')
    
    # Check if Y is a list of outputs
    if isinstance(Y, list):
        for i, y_data in enumerate(Y):
            axs[1].scatter(y_data[:, 0], y_data[:, 1], label=Y_names[i], alpha=0.4)
        axs[1].legend()
    else:
        axs[1].scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.7)
    
    axs[1].set_title(plot_names[1])
    axs[1].set_xlabel('Y1')
    axs[1].set_ylabel('Y2')
    plt.tight_layout()
    plt.show()

def create_batches(gen, size, minibatch_size, key=None):
    x_data, y_data = gen(key, num_samples=size)
    x_data = jax.device_put(x_data)
    y_data = jax.device_put(y_data)

    n_samples = x_data.shape[0]
    if key is not None:
        indices = jnp.arange(n_samples)
        shuffled_indices = random.permutation(key, indices)
        x_data = x_data[shuffled_indices]
        y_data = y_data[shuffled_indices]
    if x_data.ndim == 1:
        x_data = x_data[:, None]
    n_batches = n_samples // minibatch_size
    for i in range(n_batches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        batch = {
            'input': x_data[start_idx:end_idx],
            'target': y_data[start_idx:end_idx]
        }
        yield batch
    if n_samples % minibatch_size != 0:
        start_idx = n_batches * minibatch_size
        batch = {
            'input': x_data[start_idx:],
            'target': y_data[start_idx:]
        }
        yield batch


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim, hidden_dim, output_dim, layers, rngs: nnx.Rngs,
        activation,
        layernorm, dropout, sin_transform):
        
        self.sin_transform = sin_transform
        self.layernorm = layernorm
        
        self.activation = activation
        self.input = nnx.Linear(input_dim, hidden_dim, rngs=rngs)

        self.hidden_layers = [
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs) for _ in range(layers - 2)
        ]
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self.layernorm = nnx.LayerNorm(hidden_dim, rngs=rngs) if layernorm else None

        self.output = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
    
    def __call__(self, x, rng=None, *, deterministic=None):
        x = self.input(x)
        if self.sin_transform > 0.0:
            x = jnp.sin(self.sin_transform * x)
        for layer in self.hidden_layers:
            x = layer(x)
            if self.dropout is not None:
                x = self.dropout(x, deterministic=deterministic)
            if self.layernorm is not None:
                x = self.layernorm(x)
            x = self.activation(x)
        x = self.output(x)
        return x

class TrainState(train_state.TrainState):
    counts: nnx.State
    graphdef: nnx.GraphDef

class Count(nnx.Variable[nnx.A]):
  pass

class SimpleSGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def init(self, params):
        return {}
    def update(self, grads, opt_state, params=None):
        def update_param(grad):
            return -self.learning_rate * grad
        updates = jax.tree.map(update_param, grads)
        return updates, opt_state

def gaussian_kernel(x, y, sigma):
    x = x[:, None, :]
    y = y[None, :, :]
    pairwise_sq_dists = jnp.sum((x - y) ** 2, axis=-1)
    return jnp.exp(-pairwise_sq_dists / (2 * sigma ** 2))

def compute_mmd(x, y, sigmas):
    assert x.ndim == 2 and y.ndim == 2, "x and y must be 2D arrays"
    assert x.shape[1] == y.shape[1], "x and y must have the same number of features"
    assert x.shape[0] > 1 and y.shape[0] > 1, "x and y must be a batch"

    mm2s = []    
    for sigma in jnp.atleast_1d(sigmas):
        k_xx = gaussian_kernel(x, x, sigma)
        k_yy = gaussian_kernel(y, y, sigma)
        k_xy = gaussian_kernel(x, y, sigma)

        mm2s.append(jnp.mean(k_xx) + jnp.mean(k_yy) - 2 * jnp.mean(k_xy))
    mmd2 = jnp.mean(jnp.array(mm2s))
    return jnp.sqrt(jnp.maximum(mmd2, 0))

def compute_gaussian_penalty(y_pred, weight):
    mean = jnp.mean(y_pred, axis=0)
    cov = jnp.cov(y_pred, rowvar=False)
    return weight * (
        jnp.sum(mean**2) 
        + jnp.sum((cov - jnp.eye(2))**2)
    )

def compute_regularization(params, regularisation):
    l1_reg = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
    l2_reg = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
    return regularisation[0] * l1_reg + regularisation[1] * l2_reg

def loss_fn(params, state, batch, rng, sigma, gaussian_weight, regularisation, deterministic):
    x, y = batch["input"], batch["target"]
    model = nnx.merge(state.graphdef, params, state.counts)
    y_pred = model(x, rng, deterministic=deterministic)
    
    mmd_value = compute_mmd(y, y_pred, sigma)
    
    counts = nnx.state(model, Count)

    gaussian_penalty = compute_gaussian_penalty(y_pred, gaussian_weight)

    reg_penalty = compute_regularization(params, regularisation)

    loss = mmd_value + gaussian_penalty + reg_penalty
    
    return loss, y_pred, counts



# Data generation functions

def plot_progress(metrics_history):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['train_epochs'], metrics_history['train_loss'], label='Training Loss', alpha=0.7)
    plt.plot(metrics_history['val_epochs'], metrics_history['val_loss'], 'o-', label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def train_model(state, train_batches, valid_batches, metrics, num_epochs, eval_every, key, minibatch_size, sigma, regularisation, gaussian_weight, visualize=True, console: Console | None = None):
    metrics_history = {
        'train_epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_epochs': [],
    }

    epoch_key = random.split(key)[1]
    total_eval_time = 0
    start_time = time.time()
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        batch_key, epoch_key = random.split(epoch_key, 2)

        # Train on all training batches
        for batch in train_batches(key=epoch_key, minibatch_size=minibatch_size):
            batch_key = random.split(batch_key)[1]
            state, metrics_dict = train_step(
                state, batch, rng=batch_key, sigma=sigma, regularisation=regularisation, gaussian_weight=gaussian_weight)
            metrics.update(**metrics_dict)
        train_metrics = metrics.compute()
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_epochs'].append(epoch)
        metrics.reset()

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            eval_start_time = time.time()
            epoch_key = random.split(epoch_key, 1)[1]
            batch_key = epoch_key
            for batch in valid_batches(minibatch_size=minibatch_size, key=epoch_key):
                batch_key = random.split(batch_key, 1)[1]
                metrics_dict = eval_step(
                    state, batch, rng=batch_key, sigma=sigma, regularisation=regularisation, gaussian_weight=gaussian_weight)
                metrics.update(**metrics_dict)
            val_metrics = metrics.compute()
            metrics_history['val_loss'].append(val_metrics['loss'])
            metrics_history['val_epochs'].append(epoch)
            metrics.reset()
            eval_time = time.time() - eval_start_time
            total_eval_time += eval_time

            avg_epoch_time = float(jnp.mean(jnp.array(epoch_times))) if len(epoch_times) > 0 else 0.0

            print(
                f"Epoch {epoch} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Eval Time: {eval_time:.2f}s, "
                f"Avg Epoch Time: {avg_epoch_time:.4f}s"
            )

            epoch_times = []
            if epoch > 0 and visualize:
                plot_progress(metrics_history)

        epoch_times.append(time.time() - epoch_start_time)

    total_time = time.time() - start_time

    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Evaluation Time: {total_eval_time:.2f}s")
    print(f"Average Epoch Time: {total_time / num_epochs:.2f}s")
    if len(metrics_history['val_loss']) > 0:
        best_val = float(jnp.min(jnp.array(metrics_history['val_loss'])))
        print(f"Best Val Loss: {best_val:.4f}")

    return state

@jax.jit
def train_step(state, batch, rng, sigma, regularisation, gaussian_weight):
    def local_fn(params, rng):
        loss, _, counts = loss_fn(params, state, batch, rng, sigma, gaussian_weight, regularisation, False)
        return loss, counts
    grads, counts = jax.grad(local_fn, has_aux=True)(state.params, rng)
    state = state.apply_gradients(grads=grads)
    loss = local_fn(state.params, rng)[0]
    return state, {'loss': loss}

@jax.jit
def eval_step(state, batch, rng, sigma, regularisation, gaussian_weight):
    return {'loss': loss_fn(state.params, state, batch, rng, sigma, gaussian_weight, regularisation, True)[0]}

    
def do_complete_experiment(
    key,
    data_gen, distributions, train_size, valid_size,
    learning_rate=0.005, minibatch_size=256, num_epochs=50, eval_every=5, sigma=0.5, hidden_dim=64, layers=3, layernorm=False, dropout=0.0, sin_transform=0.0, activation=nnx.relu,
    l1_regularisation=0.0, l2_regularisation=0.0, gaussian_weight=0.0
    ):

    console = Console(force_jupyter=True)
    
    temp_data = data_gen(key, num_samples=10)
    input_dim = temp_data[0].shape[1] if len(temp_data[0].shape) > 1 else 1
    output_dim = temp_data[1].shape[1] if len(temp_data[1].shape) > 1 else 1

    model = MLP(
        rngs=nnx.Rngs(42),
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=hidden_dim, layers=layers,
        activation=activation,
        layernorm=layernorm, sin_transform=sin_transform, dropout=dropout
    )
    
    graphdef, params, counts = nnx.split(model, nnx.Param, nnx.Variable)
    sgd = SimpleSGD(learning_rate=learning_rate)

    state = TrainState.create(
        apply_fn=None,
        graphdef=graphdef,
        params=params,
        tx=sgd,
        counts=counts,
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    # Create compact rich tables
    model_table = Table(title="Model", show_header=False, box=None)
    model_table.add_row("Input Dim", str(input_dim))
    model_table.add_row("Output Dim", str(output_dim))
    model_table.add_row("Hidden Dim", str(hidden_dim))
    model_table.add_row("Layers", str(layers))
    model_table.add_row("Parameters", f"{sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    model_table.add_row("Activation", activation.__name__)
    model_table.add_row("LayerNorm", str(layernorm))
    model_table.add_row("Dropout", str(dropout))
    model_table.add_row("Sin Transform", str(sin_transform))

    train_table = Table(title="Training", show_header=False, box=None)
    train_table.add_row("Train Size", f"{train_size:,}")
    train_table.add_row("Valid Size", f"{valid_size:,}")
    train_table.add_row("Learning Rate", str(learning_rate))
    train_table.add_row("Batch Size", str(minibatch_size))
    train_table.add_row("Epochs", str(num_epochs))
    train_table.add_row("Eval Every", str(eval_every))

    loss_table = Table(title="Loss", show_header=False, box=None)
    loss_table.add_row("MMD Sigma", str(sigma))
    loss_table.add_row("L1 Reg", str(l1_regularisation))
    loss_table.add_row("L2 Reg", str(l2_regularisation))
    loss_table.add_row("Gaussian Weight", str(gaussian_weight))

    console.print(Panel(Columns([model_table, train_table, loss_table]), title="[bold blue]Experiment Configuration[/bold blue]"))

    experiment_start_time = time.time()

    trained_state = train_model(
        state,
        partial(create_batches, gen=data_gen, size=train_size),
        partial(create_batches, gen=data_gen, size=valid_size),
        metrics,
        num_epochs=num_epochs,
        eval_every=eval_every,
        sigma=sigma,
        minibatch_size=minibatch_size,
        regularisation=(l1_regularisation, l2_regularisation),
        gaussian_weight=gaussian_weight,
        key=key,
        visualize=False,
        console=console,
    )

    experiment_time = time.time() - experiment_start_time
    console.print(f"[green]Experiment completed in {experiment_time:.2f}s[/green]")

    visual_data = next(create_batches(
        gen=data_gen, size=1000, minibatch_size=valid_size, key=random.key(200)
    ))

    loss, pred, _ = loss_fn(trained_state.params, trained_state, visual_data, sigma=0.5, rng=random.key(1012), regularisation=(l1_regularisation, l2_regularisation), gaussian_weight=gaussian_weight, deterministic=True)

    plot_data(
        visual_data['input'], [visual_data['target'], pred],
        x_hist=visual_data['input'].ndim == 1,
        plot_names=distributions,
        Y_names=["Target Distribution", "Generated Distribution"]
    )

    return trained_state


def final_performance(
    key, 
    state,
    sigma,
    data_gen, size=1000, 
):
    test_data = next(create_batches(data_gen, size=size, minibatch_size=size, key=key))

    test_data = jax.device_put(test_data)

    loss, pred, counts = loss_fn(
        state.params, state, test_data, rng=key, sigma=sigma, 
        regularisation=(0.0, 0.0), gaussian_weight=0.0, deterministic=True
    )

    return loss

