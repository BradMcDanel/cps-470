import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())

images, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)))


def show_grid(imgs, titles=None, cols=8):
    """Display a grid of images. imgs: (N, 3, H, W) tensor."""
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flat if rows > 1 else axes
    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation='nearest')
            if titles:
                ax.set_title(titles[i], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def hflip(img):
    # img: (C, H, W)
    pass


def pixel_jitter(img, sigma=0.1):
    # img: (C, H, W)
    pass


if __name__ == "__main__":
    # Checkpoint A: display a grid of CIFAR-10 images with class labels

    # Checkpoint B: implement hflip and pixel_jitter above, then display the results
    flipped  = hflip(images)
    jittered = pixel_jitter(images, sigma=0.1)
    # show_grid(...)

    # Checkpoint C: copy train_transform from the slides and display a batch
    train_transform = transforms.Compose([
        # ...
    ])
    aug_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    aug_images, _ = next(iter(torch.utils.data.DataLoader(aug_dataset, batch_size=16, shuffle=True)))
    # show_grid(...)
