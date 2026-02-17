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
    model = nn.Sequential(
        nn.Linear(54, 128),
        nn.ReLU(),
        nn.Linear(128, 7)
    )

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # TODO: training loop
    for epoch in range(200):
        preds = model(X_train)
        optim.zero_grad()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optim.step()

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
    model = nn.Sequential(
        nn.Linear(54, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 7)
    )

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        preds = model(X_train)
        optim.zero_grad()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optim.step()

        if epoch % 50 == 0:
            print("Epoch:", epoch)
            with torch.no_grad():
                train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
                test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
                print(f"  Train accuracy: {train_acc:.2%}")
                print(f"  Test accuracy:  {test_acc:.2%}")
            print()

    model = nn.Sequential(
        nn.Linear(54, 256),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(256, 7)
    )

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        preds = model(X_train)
        optim.zero_grad()
        loss = loss_fn(preds, y_train)
        loss.backward()
        optim.step()

        if epoch % 50 == 0:
            print("Epoch:", epoch)
            with torch.no_grad():
                train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean()
                test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
                print(f"  Train accuracy: {train_acc:.2%}")
                print(f"  Test accuracy:  {test_acc:.2%}")
            print()
