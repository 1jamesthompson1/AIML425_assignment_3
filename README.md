# Instructions

I have to choose a single assignment to work on.

## Assignments I
1. Autoencoder, generative modeling, and information rate:
▶ Create 3D data that are uniformly distributed over the surface of a cube.
▶ Create an autoencoder (not VAE) with an MSE objective function, with an
adjustable number of neurons in the latent layer (bottleneck).
▶ Provide a probabilistic interpretation of the MSE objective function.
▶ Include a method to control the distribution of the latent (bottleneck) variable.
▶ Add iid Gaussian noise to the latent (bottleneck) variable with an adjustable
(not learned) SNR (ratio of variance of variable=signal over variance of noise).
▶ Estimate the information in bits passing through the latent layer for your
settings (see equation (2) ).
▶ Discuss the attributes of the reconstruction that you obtain at various SNRs
and dimensionalities of the latent vector (select a range of interesting settings).
▶ Create a generative system from your autoencoder, define a suitable quality
measure, and quantify the generative performance of your autoencoder.
▶ Extend your autoencoder study with an interesting direction of your choice.
15


## Assignments II

2. Autoencoder, generation of shapes, and information rate.
▶ Create a database of sufficient size of 28 × 28 images with circles, triangles,
and rectangles at random locations (one shape per image).
▶ Create a suitable variational autoencoder (VAE) for the data in the database.
Motivate the network configuration (architecture, objective functions,
hyperparameters) you select.
▶ Use your VAE to generate new data and select and provide values for a suitable
quantitative measure of its performance.
▶ Modify your VAE into a system that does not use the ELBO:
▶ Make it into a basic autoencoder (MSE) that ensures that the distribution
of the latent layer is iid Gaussian (MMD, or, easier, variance control).
▶ Now add Gaussian noise in the latent layer with a fixed variance (you set
the strength).
▶ For the modified VAE estimate the information passing through the latent
layer, in bits (see equation (2) ).
▶ Attempt to explain what this information represents for your reconstruction.
▶ Compare the modified VAE and the standard VAE, based on suitable measures
and settings

## Syncing of library

My workflow involves working on my laptop for most tasks but access vscode on a remote machine (on VUW campus) which is more powerful to do my coding and report writing. Therefore I need to sync my Zotero .bib export from my laptop to this University machine. Therefore to do this I run this command:
```bash
rsync -avz /home/james/code/VUWMAI/bibliographies/AIML425.bib vuw-lab:/home/thompsjame1/code/AIML425_assignment_3/references.bib
```