# VAE

This is how the original data looks like:

![Real MNIST](/data/examples/real_mnist.jpg)

Here is the reconstruction of VAE for the same data points:

![VAE Reconstruction](/data/examples/vae_reconstruction.jpg)

In these images generated by a VAE, some edges are quite blurry. This is because the VAE picks the safe path and makes some parts blurry by choosing a "safe" pixel value, which minimizes the loss, but does not provide good images.

Here we do the 2D projection of all the points from the test set into the latent space and their class. First, we display the 2D latent space onto the graph. We then map out the classes of these generated examples and color them accordingly, as per the legend on the right. Here we can see that the classes tend to be neatly grouped together, which tells us that this is a good representation.

![2D Projection](/data/examples/latent_space.jpg)

We map out the values of a subset of the latent space on a grid and pass each of those latent space values through the generator to produce this plot. This gives us a sense of how much the resulting picture changes as we vary z.

![Latent Space Morph](/data/examples/latent_morph.jpg)

# Original GAN

## Examples

GAN was trained on data from MNIST dataset. Here is how the digits from the dataset look like:

![MNIST Grid](/data/examples/mnist_grid.jpg)

You can see how the network is slowly learning to capture the data distribution during training:

![MNIST Training](/data/examples/gan_progress.gif)

After the generator is trained we can use it to generate all 10 digits:

![GAN generated](/data/examples/generated_by_gan.jpg)

Similarly to how it was done in VAE, we can take 2 generated numbers, save their latent vectors, and subsequently linearly or spherically
interpolate between them to generate new images and understand how the latent space is structured:

![GAN Interpolated](/data/examples/gan_interpolate.jpg)

We can see how the number 4 is slowly morphing into 9 and then into the number 3.
