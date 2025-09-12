import jax.numpy as jnp
import jax
import jax.random as random
from flax import nnx
from flax.training import train_state

from optax import adam

import matplotlib.pyplot as plt
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

from src.model import VAE


class TrainState(train_state.TrainState):
    counts: nnx.State
    graphdef: nnx.GraphDef


class Count(nnx.Variable[nnx.A]):
    pass


def reconstruction_error(x, y):
    return jnp.mean(jnp.square(x - y))

def binary_cross_entropy_with_logits(logits, batch):
  logits = nnx.log_sigmoid(logits)
  return -jnp.sum(
      batch * logits + (1.0 - batch) * jnp.log(-jnp.expm1(logits)+1e-8)
  )

def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def loss_fn(params, state, batch, rng, deterministic):
    x = batch["input"]
    model = nnx.merge(state.graphdef, params, state.counts)
    x_recon, mean, logvar = model(x, rng, deterministic=deterministic)

    counts = nnx.state(model, Count)

    loss = binary_cross_entropy_with_logits(x_recon, x).mean() + kl_divergence(mean, logvar).mean()

    return loss, x_recon, counts


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
        "val_loss": [],
        "val_epochs": [],
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
            state, metrics_dict = train_step(state, batch, rng=batch_key)
            metrics.update(**metrics_dict)
        train_metrics = metrics.compute()
        metrics_history["train_loss"].append(train_metrics["loss"])
        metrics_history["train_epochs"].append(epoch)
        metrics.reset()

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            eval_start_time = time.time()
            epoch_key = random.split(epoch_key, 1)[1]
            batch_key = epoch_key
            for batch in valid_batches(minibatch_size=minibatch_size, key=epoch_key):
                batch_key = random.split(batch_key, 1)[1]
                metrics_dict = eval_step(state, batch, rng=batch_key)
                metrics.update(**metrics_dict)
            val_metrics = metrics.compute()
            metrics_history["val_loss"].append(val_metrics["loss"])
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

    return state


@jax.jit
def train_step(state, batch, rng):
    def local_fn(params, rng):
        loss, _, counts = loss_fn(params, state, batch, rng, False)
        return loss, counts

    grads, counts = jax.grad(local_fn, has_aux=True)(state.params, rng)
    state = state.apply_gradients(grads=grads)
    loss = local_fn(state.params, rng)[0]
    return state, {"loss": loss}


@jax.jit
def eval_step(state, batch, rng):
    return {"loss": loss_fn(state.params, state, batch, rng, True)[0]}


def do_complete_experiment(
    key,
    train_batches,
    valid_batches,
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

    model = VAE(
        rngs=nnx.Rngs(42),
        input=input_dim,
        encoder_arch=encoder_arch,
        decoder_arch=decoder_arch,
        latent_dim=latent_dim,
        dropout=dropout,
        layernorm=layernorm,
    )

    graphdef, params, counts = nnx.split(model, nnx.Param, nnx.Variable)

    state = TrainState.create(
        apply_fn=None,
        graphdef=graphdef,
        params=params,
        tx=adam(learning_rate),
        counts=counts,
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

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

    console.print(
        Panel(
            Columns([model_table, train_table]),
            title="[bold blue]Experiment Configuration[/bold blue]",
        )
    )

    experiment_start_time = time.time()

    trained_state = train_model(
        state,
        train_batches,
        valid_batches,
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

    return trained_state
