import jax.numpy as jnp
import jax
import jax.random as random
from flax import nnx
from flax.training import train_state

from optax import adam, sigmoid_binary_cross_entropy

import matplotlib.pyplot as plt
import time
from functools import partial
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

class TrainState(train_state.TrainState):
    counts: nnx.State
    graphdef: nnx.GraphDef


class Count(nnx.Variable[nnx.A]):
    pass


def reconstruction_error(x, y):
    '''
    Simple mean squared error reconstruction loss

    x: original input (n_batch, n_features)
    y: reconstructed input (n_batch, n_features)

    returns: mean squared error between x and y (scalar)
    '''
    return jnp.mean(jnp.square(x - y))

def kl_divergence(mean, logvar):
    '''
    KL divergence between N(mean, var) and N(0, 1)

    mean: (n_batch, latent_dim)
    logvar: (n_batch, latent_dim)
    
    Computes KL divergence for each element in the batch and sums over latent dimensions. Can use the analytical form of KL.
    
    returns: (n_batch,) KL divergence for each element in the batch
    
    '''
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def vae_loss_fn(params, state, batch, rng, deterministic, kl_beta=1.0):
    x = batch["input"]
    model = nnx.merge(state.graphdef, params, state.counts)
    x_recon, mean, logvar = model(x, rng, deterministic=deterministic)

    counts = nnx.state(model, Count)

    reconstruction_error_val = sigmoid_binary_cross_entropy(x_recon, x).sum()

    kl_divergence_val = kl_divergence(mean, logvar).mean()

    loss = reconstruction_error_val + kl_beta * kl_divergence_val

    return loss, x_recon, counts, (reconstruction_error_val, kl_beta * kl_divergence_val)


def compute_mmd(z, z_prior, sigmas):
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        z: Samples from the model's latent space, shape (n_samples, latent_dim).
        z_prior: Samples from the prior distribution, shape (n_samples, latent_dim).
        sigmas: List or array of bandwidths for the RBF kernel.

    Returns:
        MMD value (scalar).
    """
    def rbf_kernel(x, y, sigma):
        x_norm = jnp.sum(x ** 2, axis=1).reshape(-1, 1)
        y_norm = jnp.sum(y ** 2, axis=1).reshape(1, -1)
        cross_term = jnp.dot(x, y.T)
        dist = x_norm + y_norm - 2 * cross_term
        return jnp.exp(-dist / (2 * sigma ** 2))

    mmd = 0.0
    for sigma in sigmas:
        k_zz = rbf_kernel(z, z, sigma)
        k_pp = rbf_kernel(z_prior, z_prior, sigma)
        k_zp = rbf_kernel(z, z_prior, sigma)

        mmd += jnp.mean(k_zz) + jnp.mean(k_pp) - 2 * jnp.mean(k_zp)

    return mmd

def ae_loss_fn(params, state, batch, rng, deterministic, regularization_weight=1.0, mmd_sigma=(0.5, 1.0, 2.0)):

    x = batch["input"]
    
    model = nnx.merge(state.graphdef, params, state.counts)
    x_recon, z = model(x, rng, deterministic=deterministic)

    reconstruction_error_val = reconstruction_error(x, nnx.sigmoid(x_recon))

    # Not using as just caused a two hump distribution
    if mmd_sigma is None:
        reg_penalty = jnp.mean(jnp.mean(z, axis=1)**2 + (jnp.var(z, axis=1) - 1)**2)
    else:
        reg_penalty = compute_mmd(z, random.normal(rng, z.shape), mmd_sigma)
    reg_penalty = regularization_weight * reg_penalty

    loss = reconstruction_error_val + reg_penalty
    counts = nnx.state(model, Count)
    return loss, x_recon, counts, (reconstruction_error_val, reg_penalty)


def plot_progress(metrics_history):
    plt.figure(figsize=(10, 5))
    plt.plot(
        metrics_history["train_epochs"],
        metrics_history["train_loss"],
        label="Training Loss",
        alpha=0.7,
    )
    plt.plot(
        metrics_history["val_epochs"],
        metrics_history["val_loss"],
        "o-",
        label="Validation Loss",
        alpha=0.7,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def train_model(
    state,
    train_batches,
    valid_batches,
    loss_fn,
    metrics,
    num_epochs,
    minibatch_size,
    eval_every,
    key,
    visualize=True,
    console: Console | None = None,
):
    metrics_history = {
        "train_epochs": [],
        "train_loss": [],
        "train_loss_recon": [],
        "train_loss_reg": [],
        "val_loss": [],
        "val_loss_recon": [],
        "val_loss_reg": [],
        "val_epochs": [],
        "val_loss_parts": [],
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
            state, metrics_dict = train_step(state, loss_fn, batch, rng=batch_key)
            metrics.update(**metrics_dict)
        train_metrics = metrics.compute()
        metrics_history["train_loss"].append(train_metrics["loss"])
        metrics_history["train_loss_recon"].append(train_metrics["reconstruction_loss"])
        metrics_history["train_loss_reg"].append(train_metrics["regularization_loss"])
        metrics_history["train_epochs"].append(epoch)
        metrics.reset()

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            eval_start_time = time.time()
            epoch_key = random.split(epoch_key, 1)[1]
            batch_key = epoch_key
            for batch in valid_batches(minibatch_size=minibatch_size, key=epoch_key):
                batch_key = random.split(batch_key, 1)[1]
                metrics_dict = eval_step(state, loss_fn, batch, rng=batch_key)
                metrics.update(**metrics_dict)
            val_metrics = metrics.compute()
            metrics_history["val_loss"].append(val_metrics["loss"])
            metrics_history["val_loss_recon"].append(val_metrics["reconstruction_loss"])
            metrics_history["val_loss_reg"].append(val_metrics["regularization_loss"])
            metrics_history["val_epochs"].append(epoch)
            metrics.reset()
            eval_time = time.time() - eval_start_time
            total_eval_time += eval_time

            avg_epoch_time = (
                float(jnp.mean(jnp.array(epoch_times))) if len(epoch_times) > 0 else 0.0
            )

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
    if len(metrics_history["val_loss"]) > 0:
        best_val = float(jnp.min(jnp.array(metrics_history["val_loss"])))
        print(f"Best Val Loss: {best_val:.4f}")

    return state, metrics_history


@partial(jax.jit,static_argnames=["loss_fn"])
def train_step(state, loss_fn, batch, rng):
    def local_fn(params, rng):
        loss, _, counts, parts = loss_fn(params, state, batch, rng, False)
        return loss, (counts, parts)

    (grads, (counts, parts)) = jax.grad(local_fn, has_aux=True)(state.params, rng)
    state = state.apply_gradients(grads=grads)
    loss = local_fn(state.params, rng)[0]
    return state, {"loss": loss, "reconstruction_loss": parts[0], "regularization_loss": parts[1]}


@partial(jax.jit,static_argnames=["loss_fn"])
def eval_step(state, loss_fn, batch, rng):
    loss, _, _, parts = loss_fn(state.params, state, batch, rng, True)
    return {"loss": loss, "reconstruction_loss": parts[0], "regularization_loss": parts[1]}


def do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class,
    loss_fn,
    model_kwargs={},
    learning_rate=0.005,
    minibatch_size=256,
    num_epochs=50,
    eval_every=5,
    encoder_arch=[1000, 1000, 1000],
    decoder_arch=[1000, 1000, 1000],
    latent_dim=20,
    layernorm=False,
    dropout=0.0,
    activation=nnx.relu,
):
    console = Console(force_jupyter=True)

    temp = train_batches(key=key, minibatch_size=minibatch_size)
    input_dim = temp.__next__()["input"].shape[-1]

    model = model_class(
        rngs=nnx.Rngs(42),
        input=input_dim,
        encoder_arch=encoder_arch,
        decoder_arch=decoder_arch,
        latent_dim=latent_dim,
        dropout=dropout,
        layernorm=layernorm,
        **model_kwargs,
    )

    graphdef, params, counts = nnx.split(model, nnx.Param, nnx.Variable)

    state = TrainState.create(
        apply_fn=None,
        graphdef=graphdef,
        params=params,
        tx=adam(learning_rate),
        counts=counts,
    )
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        reconstruction_loss=nnx.metrics.Average("reconstruction_loss"),
        regularization_loss=nnx.metrics.Average("regularization_loss"),
    )

    # Create compact rich tables
    model_table = Table(title="Model", show_header=False, box=None)
    model_table.add_row("Input Dim", str(input_dim))
    model_table.add_row("Encoder arch", str(encoder_arch))
    model_table.add_row("Decoder arch", str(decoder_arch))
    model_table.add_row("Latent Dim", str(latent_dim))
    model_table.add_row(
        "Parameters", f"{sum(x.size for x in jax.tree_util.tree_leaves(params)):,}"
    )
    model_table.add_row("Activation", activation.__name__)
    model_table.add_row("LayerNorm", str(layernorm))
    model_table.add_row("Dropout", str(dropout))

    train_table = Table(title="Training", show_header=False, box=None)
    train_table.add_row("Learning Rate", str(learning_rate))
    train_table.add_row("Batch Size", str(minibatch_size))
    train_table.add_row("Epochs", str(num_epochs))
    train_table.add_row("Eval Every", str(eval_every))
    train_table.add_row("Loss function", str(loss_fn))

    console.print(
        Panel(
            Columns([model_table, train_table]),
            title="[bold blue]Experiment Configuration[/bold blue]",
        )
    )

    experiment_start_time = time.time()

    trained_state, history = train_model(
        state,
        train_batches,
        valid_batches,
        loss_fn,
        metrics,
        num_epochs=num_epochs,
        eval_every=eval_every,
        minibatch_size=minibatch_size,
        key=key,
        visualize=False,
        console=console,
    )

    experiment_time = time.time() - experiment_start_time
    console.print(f"[green]Experiment completed in {experiment_time:.2f}s[/green]")

    return nnx.merge(trained_state.graphdef, trained_state.params, trained_state.counts), history
