import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load Fashion-MNIST (grayscale, 28x28, 10 clothing categories)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
full_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(full_dataset, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=128)
test_loader  = DataLoader(test_dataset,  batch_size=128)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_and_report(model, optimizer, epochs=5):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"  Test accuracy: {test_acc:.4f}")
    return test_acc


# =============================================================================
# Example: VGGBlock as an nn.Module
# =============================================================================
# So far we've built models with nn.Sequential, which works great for straight
# chains of layers. But sometimes we need more control --- for example, a
# reusable block we can stack, or a skip connection that adds two tensors.
#
# nn.Module lets us define any computation in forward().
# Here's how VGG's block looks as a module:

class VGGBlock(nn.Module):
    def __init__(self, n_convs, in_channels, out_channels):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                    out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# VGG-16 uses 5 blocks with increasing channels:
vgg16 = nn.Sequential(
    VGGBlock(2, 1,   64),   # 28x28 -> 14x14
    VGGBlock(2, 64,  128),  # 14x14 -> 7x7
    VGGBlock(3, 128, 256),  # 7x7   -> 3x3
    nn.Flatten(),
    nn.Linear(256 * 3 * 3, 10),
)

if __name__ == "__main__":
    # Showing how this works in practice..
    x = torch.zeros(1, 1, 28, 28)
    print("VGGBlock output shape:", vgg16(x).shape)  # should be (1, 10)

    # =========================================================================
    # Checkpoint A: Plain CNN baseline
    # =========================================================================
    # Build a simple CNN for Fashion-MNIST (1x28x28 input, 10 classes).
    # Use 3 conv blocks, each with: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    # Channels: 1 -> 32 -> 64 -> 128, kernel_size=3, padding=1
    # Then Flatten -> Linear(128 * 3 * 3, 10)
    #
    # plain_cnn = nn.Sequential(...)

    # optimizer = torch.optim.Adam(plain_cnn.parameters(), lr=1e-3)
    # print("Plain CNN:")
    # train_and_report(plain_cnn, optimizer)

    # =========================================================================
    # Checkpoint B: Implement a residual block
    # =========================================================================
    # Create a class ResidualBlock(nn.Module) with:
    #   - Two Conv2d(channels, channels, 3, padding=1) layers
    #   - BatchNorm2d after each conv
    #   - ReLU after the first batchnorm
    #   - Skip connection: add input x before the final ReLU
    #
    # Hint: forward(self, x) should return relu(F(x) + x)
    #
    # class ResidualBlock(nn.Module):
    #     ...
    #
    # Then build a residual CNN: same as checkpoint A but replace
    # the middle conv block with a ResidualBlock(64).
    #
    # res_cnn = nn.Sequential(...)
    # optimizer = torch.optim.Adam(res_cnn.parameters(), lr=1e-3)
    # print("\nResidual CNN:")
    # train_and_report(res_cnn, optimizer)

    # =========================================================================
    # Checkpoint C: Compare optimizers
    # =========================================================================
    # Train the residual CNN with two optimizers and compare val accuracy:
    #   1. torch.optim.SGD(lr=0.01, momentum=0.9)
    #   2. torch.optim.Adam(lr=1e-3)

    pass
