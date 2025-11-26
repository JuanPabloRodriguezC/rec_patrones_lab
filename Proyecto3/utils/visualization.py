import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

def show_image_grid(images, n=8, title="Grid"):
    grid = images[:n].detach().cpu()
    fig, axs = plt.subplots(1, n, figsize=(n*2, 2))
    for i in range(n):
        axs[i].imshow((grid[i].permute(1,2,0)+1)/2)
        axs[i].axis("off")
    plt.suptitle(title)
    plt.show()

def plot_reconstructions(input_imgs, recon_imgs, n=8, title="Reconstrucciones"):
    input_imgs = input_imgs[:n]
    recon_imgs = recon_imgs[:n]

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

    for i in range(n):
        axes[0, i].imshow(input_imgs[i].permute(1, 2, 0).detach().cpu())
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_imgs[i].permute(1, 2, 0).detach().cpu())
        axes[1, i].axis("off")

    plt.suptitle(title)
    plt.show()

def plot_latent_tsne(latents, labels, title="t-SNE Espacio Latente"):
    tsne = TSNE(n_components=2)
    z = tsne.fit_transform(latents)
    plt.scatter(z[:,0], z[:,1], c=labels, cmap="tab10", s=10)
    plt.title(title)
    plt.show()
