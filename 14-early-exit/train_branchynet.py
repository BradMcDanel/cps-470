import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BranchyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Three backbone blocks
        self.block0 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )

        # Exit branches
        self.exit0 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
        self.exit2 = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 10))

    def forward(self, x):
        h0 = self.block0(x)
        h1 = self.block1(h0)
        h2 = self.block2(h1)

        logits0 = self.exit0(h0)
        logits1 = self.exit1(h1)
        logits2 = self.exit2(h2)
        return [logits0, logits1, logits2]


# Checkpoint A: Train with multi-exit loss
# Compute cross-entropy at each exit and combine them:
#   total loss = 0.3 * loss0 + 0.3 * loss1 + 0.4 * loss2
# Track accuracy using the final exit (exit 2).
#
def train_branchy_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        exits = model(images)  # list of 3 logit tensors
        # TODO: compute weighted loss across all exits
        # loss = ...
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (exits[-1].argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate_branchy(model, loader, criterion):
    """Evaluate using the final exit (standard evaluation)."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            exits = model(images)
            loss = criterion(exits[-1], labels)
            total_loss += loss.item() * images.size(0)
            correct += (exits[-1].argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_and_report_branchy(model, optimizer, epochs=5):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss, train_acc = train_branchy_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_branchy(model, val_loader, criterion)
        print(f"  Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    test_loss, test_acc = evaluate_branchy(model, test_loader, criterion)
    print(f"  Test accuracy: {test_acc:.4f}")
    return test_acc


if __name__ == "__main__":
    print("Training BranchyCNN")
    model = BranchyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_and_report_branchy(model, optimizer)
    torch.save(model.state_dict(), "branchycnn.pth")
    print("Saved to branchycnn.pth")
