from flax import nnx
import jax.numpy as jnp
import jax.random as random


class Encoder(nnx.Module):
    def __init__(self, dim, latent_dim, arch, rngs, dropout=0.0, layernorm=False, modelvar=True):
        self.latent_dim = latent_dim

        self.modelvar = modelvar
        self.dropout = dropout
        self.layernorm = layernorm

        self.hidden_layers = []
        self.layer_norms = []
        self.dropouts = []
        for i, h_dim in enumerate(arch):
            layer = nnx.Linear(dim if i == 0 else arch[i - 1], h_dim, rngs=rngs)
            self.hidden_layers.append(layer)

            if self.layernorm:
                ln = nnx.LayerNorm(num_features=h_dim, rngs=rngs)
                self.layer_norms.append(ln)
            else:
                self.layer_norms.append(None)

            if self.dropout > 0.0:
                drop = nnx.Dropout(rate=self.dropout, rngs=rngs)
                self.dropouts.append(drop)
            else:
                self.dropouts.append(None)

        self.mean = nnx.Linear(arch[-1], latent_dim, rngs=rngs)
        if modelvar:
            self.logvar = nnx.Linear(arch[-1], latent_dim, rngs=rngs)

    def __call__(self, x, rngs, deterministic=False):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
            x = nnx.relu(x)
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x, deterministic=deterministic)
        mean = self.mean(x)
        if self.modelvar:
            logvar = self.logvar(x)
            return mean, logvar
        else:
            return mean


class Decoder(nnx.Module):
    def __init__(self, dout, latent_dim, arch, rngs, dropout, layernorm):
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.layernorm = layernorm

        self.hidden_layers = []
        self.layer_norms = []
        self.dropouts = []
        for i, h_dim in enumerate(arch):
            layer = nnx.Linear(latent_dim if i == 0 else arch[i - 1], h_dim, rngs=rngs)
            self.hidden_layers.append(layer)

            if self.layernorm:
                ln = nnx.LayerNorm(num_features=h_dim, rngs=rngs)
                self.layer_norms.append(ln)
            else:
                self.layer_norms.append(None)

            if self.dropout > 0.0:
                drop = nnx.Dropout(rate=self.dropout, rngs=rngs)
                self.dropouts.append(drop)
            else:
                self.dropouts.append(None)
        self.out = nnx.Linear(arch[-1], dout, rngs=rngs)

    def __call__(self, z, rngs, deterministic=False):
        for i, layer in enumerate(self.hidden_layers):
            z = layer(z)
            if self.layer_norms[i] is not None:
                z = self.layer_norms[i](z)
            z = nnx.relu(z)
            if self.dropouts[i] is not None:
                z = self.dropouts[i](z, deterministic=deterministic)

        z = self.out(z)
        return z


class VAE(nnx.Module):
    def __init__(
        self,
        input,
        latent_dim,
        encoder_arch,
        decoder_arch,
        rngs,
        dropout=0.0,
        layernorm=False,
    ):
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            dim=input,
            latent_dim=latent_dim,
            arch=encoder_arch,
            rngs=rngs,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.decoder = Decoder(
            dout=input,
            latent_dim=latent_dim,
            arch=decoder_arch,
            rngs=rngs,
            dropout=dropout,
            layernorm=layernorm,
        )

    def __call__(self, x, z_rng, deterministic=False):
        mean, logvar = self.encoder(x, rngs=z_rng, deterministic=deterministic)
        if deterministic:
            z = mean
        else:
            z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z, rngs=z_rng, deterministic=deterministic)
        return recon_x, mean, logvar

    def generate(self, z):
        return nnx.sigmoid(self.decoder(z, rngs=None, deterministic=True))


def reparameterize(rng, mean, logvar):
    std = jnp.exp(logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class AutoEncoder(nnx.Module):
    def __init__(
        self,
        input,
        latent_dim,
        encoder_arch,
        decoder_arch,
        rngs,
        dropout=0.0,
        layernorm=False,
        latent_noise_scale=0.0
    ):
        self.latent_dim = latent_dim
        self.latent_noise_scale = latent_noise_scale
        self.encoder = Encoder(
            dim=input,
            latent_dim=latent_dim,
            arch=encoder_arch,
            rngs=rngs,
            dropout=dropout,
            layernorm=layernorm,
            modelvar=False,
        )
        self.decoder = Decoder(
            dout=input,
            latent_dim=latent_dim,
            arch=decoder_arch,
            rngs=rngs,
            dropout=dropout,
            layernorm=layernorm,
        )

    def __call__(self, x, z_rng, deterministic=False):
        mean = self.encoder(x, rngs=z_rng, deterministic=deterministic)
        if not deterministic and self.latent_noise_scale > 0.0:
            noise = random.normal(z_rng, mean.shape) * self.latent_noise_scale
            z = mean + noise
        else:
            z = mean
        recon_x = self.decoder(z, rngs=z_rng, deterministic=deterministic)
        return recon_x, mean

    def generate(self, z):
        return nnx.sigmoid(self.decoder(z, rngs=None, deterministic=True))