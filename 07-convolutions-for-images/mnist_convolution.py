import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def load_mnist_sample(index=0):
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    image, label = mnist[index]  # image shape: (1, 28, 28)
    image = image.squeeze(0)  # shape: (28, 28)
    return image, label


def show_image(img, title=""):
    plt.figure(figsize=(3, 3))
    plt.imshow(img.detach().cpu(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def my_conv2d(image, kernel):
    # image: (H, W), kernel: (K, K)
    # output: (H - K + 1, W - K + 1)
    H, W = image.shape
    K = kernel.shape[0]
    out_h = H - K + 1
    out_w = W - K + 1
    out = torch.zeros((out_h, out_w), dtype=image.dtype)

    # TODO: implement nested-loop convolution (no F.conv2d)
    # For simplicity, we assume no padding or striding for now.

    return out


if __name__ == "__main__":
    # =========================================================================
    # Checkpoint A: Load MNIST and visualize a digit
    # =========================================================================
    # TODO: change the index or add a loop to inspect a few different digits
    image, label = load_mnist_sample(index=0)

    print("Checkpoint A:")
    print(f"  Label: {label}")
    print(f"  Image shape: {tuple(image.shape)}")
    show_image(image, title=f"MNIST digit: {label}")
    print()

    # =========================================================================
    # Checkpoint B: Implement my_conv2d(image, kernel)
    # =========================================================================
    # TODO: define a 3x3 kernel tensor
    kernel = None

    my_out = my_conv2d(image, kernel)
    expected_shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1)
    torch_out = F.conv2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
    ).squeeze(0).squeeze(0)

    print("Checkpoint B:")
    print(f"  expected shape (no stride, no padding): {expected_shape}")
    print(f"  my_out shape: {tuple(my_out.shape)}")
    print(f"  torch_out shape: {tuple(torch_out.shape)}")
    print(f"  Max absolute difference: {(my_out - torch_out).abs().max().item():.6f}")
    print()

    # =========================================================================
    # Checkpoint C: Experiment with different kernels
    # =========================================================================
    # TODO: define several kernels (e.g., vertical edge, horizontal edge, blur, sharpen)
    # TODO: apply each kernel to the MNIST image using my_conv2d
    # TODO: visualize each output feature map with show_image
    # TODO: inspect which kernels highlight edges, smooth, or sharpen
