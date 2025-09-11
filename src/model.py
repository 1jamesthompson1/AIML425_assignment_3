from flax import nnx
import jax.numpy as jnp
import jax.random as random


class Encoder(nnx.Module):
    def __init__(self, dim, latent_dim, arch, rngs):
        self.latent_dim = latent_dim
        
        self.hidden_layers = []
        for i, h_dim in enumerate(arch):
            layer = nnx.Linear(dim if i == 0 else arch[i-1], h_dim, rngs=rngs)
            self.hidden_layers.append(layer)

        self.mean = nnx.Linear(arch[-1], latent_dim, rngs=rngs)
        self.logvar = nnx.Linear(arch[-1], latent_dim, rngs=rngs)

    def __call__(self, x):
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


class Decoder(nnx.Module):
    def __init__(self, dout, latent_dim, arch, rngs):
        self.latent_dim = latent_dim

        self.hidden_layers = []
        for i, h_dim in enumerate(arch):
            layer = nnx.Linear(latent_dim if i == 0 else arch[i-1], h_dim, rngs=rngs)
            self.hidden_layers.append(layer)

        self.out = nnx.Linear(arch[-1], dout, rngs=rngs)

    def __call__(self, z):
        for layer in self.hidden_layers:
            z = nnx.relu(layer(z))
        z = self.out(z)
        return z


class VAE(nnx.Module):
    def __init__(self, input,
                 latent_dim, encoder_arch, decoder_arch, rngs):
        self.latent_dim = latent_dim
        self.encoder = Encoder(dim=input, latent_dim=latent_dim, arch=encoder_arch, rngs=rngs)
        self.decoder = Decoder(dout=input, latent_dim=latent_dim, arch=decoder_arch, rngs=rngs)

    def __call__(self, x, z_rng, deterministic=False):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nnx.sigmoid(self.decoder(z))
    

def reparameterize(rng, mean, logvar):
    std = jnp.exp(logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std
