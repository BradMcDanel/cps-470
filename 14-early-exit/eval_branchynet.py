import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_branchynet import BranchyCNN

# Load Fashion-MNIST test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=128)


# Checkpoint B: Early exit inference
# Run through blocks one at a time. After each block, check the exit
# branch's confidence: if max(softmax(logits)) > threshold, take that
# prediction and skip the remaining blocks. Fall back to the final exit.
# We process one image at a time here for simplicity.
#
def evaluate_early_exit(model, loader, threshold=0.8):
    model.eval()
    correct = 0
    total = 0
    exit_counts = [0, 0, 0]  # how many samples exit at each branch

    with torch.no_grad():
        for images, labels in loader:
            for i in range(images.size(0)):
                x = images[i:i+1]  # single image, keep batch dim
                label = labels[i].item()

                # Run through each block and check the exit
                # Hint: use model.block0, model.block1, model.block2
                #       and model.exit0, model.exit1, model.exit2
                # Hint: confidence = logits.softmax(dim=1).max().item()

                # TODO: implement early exit logic

                total += 1

    accuracy = correct / total
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Exit distribution: exit0={exit_counts[0]}, exit1={exit_counts[1]}, exit2={exit_counts[2]}")
    avg_blocks = (exit_counts[0] * 1 + exit_counts[1] * 2 + exit_counts[2] * 3) / total
    print(f"  Avg blocks used: {avg_blocks:.2f} / 3")
    return accuracy, exit_counts


if __name__ == "__main__":
    model = BranchyCNN()
    model.load_state_dict(torch.load("branchycnn.pth", weights_only=True))
    print("Loaded branchycnn.pth")

    # Checkpoint B: Early exit inference
    # evaluate_early_exit(model, test_loader, threshold=0.8)

    # Checkpoint C: Sweep thresholds
    # What happens at threshold=0.5? At threshold=0.99? At threshold=1.0?
    #
    # for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
    #     evaluate_early_exit(model, test_loader, threshold=t)
    #     print()
