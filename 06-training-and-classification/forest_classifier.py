import torch
import torch.nn as nn
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(test_fraction=0.25, n_samples=5000):
    """Load forest cover type dataset, normalize, and split into train/test."""
    covtype = fetch_covtype()

    # Use a subset for manageable training time
    np.random.seed(42)
    indices = np.random.choice(len(covtype.data), n_samples, replace=False)
    X = covtype.data[indices]
    y = covtype.target[indices] - 1  # Convert to 0-indexed (originally 1-7)

    # Shuffle and split
    perm = np.random.RandomState(42).permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = int(len(X) * test_fraction)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    # Normalize (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"Dataset: Forest Cover Type")
    print(f"Features: 54 (cartographic variables)")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Classes: 7 forest cover types")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print()

    # =========================================================================
    # Checkpoint A: Build and train a classifier
    # =========================================================================
    # Build a one hidden layer MLP that outputs 7 class scores.
    # Input: 54 features, Hidden: 128 neurons, Output: 7 classes
    # Use nn.CrossEntropyLoss and torch.optim.Adam (lr=0.001).
    # Train for 200 epochs. Print accuracy every 50 epochs.
    #
    # To compute accuracy:
    #   preds = logits.argmax(dim=1)
    #   acc = (preds == labels).float().mean()

    # TODO: define model, loss function, and optimizer

    # TODO: training loop

    print("Checkpoint A:")
    with torch.no_grad():
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Test accuracy:  {test_acc:.2%}")
    print()

    # =========================================================================
    # Checkpoint B: Overfit on purpose
    # =========================================================================
    # Build a much larger model (3 hidden layers, 256 neurons each).
    # Architecture: 54 -> 256 -> 256 -> 256 -> 7
    # Train for 300 epochs. Track train and test loss each epoch.
    # Print final train vs. test accuracy. You should see overfitting.

    # TODO: define oversized model, loss function, and optimizer

    # TODO: training loop (store train_losses and test_losses lists)

    print("Checkpoint B:")
    with torch.no_grad():
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Test accuracy:  {test_acc:.2%}")
    print()

    # =========================================================================
    # Checkpoint C: Add dropout
    # =========================================================================
    # Rebuild the oversized model with nn.Dropout(0.3) after each ReLU.
    # Train the same way (300 epochs). Compare test accuracy to Checkpoint B.
    # Dropout should reduce overfitting and improve generalization.
    #
    # Remember: model.train() before training, model.eval() before evaluation.

    # TODO: define model with dropout, loss function, and optimizer

    # TODO: training loop

    print("Checkpoint C:")
    with torch.no_grad():
        model.eval()
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
        print(f"  Train accuracy: {train_acc:.2%}")
        print(f"  Test accuracy:  {test_acc:.2%}")
