import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing


def load_data():
    """Load California Housing dataset as float32 tensors."""
    data = fetch_california_housing()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)
    print(f"Features: {data.feature_names}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Target: median house value (in $100k)")
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print()

    # =========================================================================
    # Checkpoint A: Normalize the features
    # =========================================================================
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    print("Checkpoint A:")
    print(f"  Feature means:  {X.mean(dim=0).tolist()}")
    print(f"  Feature stds:   {X.std(dim=0).tolist()}")
    print()

    # =========================================================================
    # Checkpoint B: Build and train a single hidden layer MLP
    # =========================================================================
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    for epoch in range(100):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.4f}")

    print("Checkpoint B:")
    with torch.no_grad():
        y_hat = model(X)
        mse = loss_fn(y_hat, y)
        print(f"  Training MSE: {mse.item():.4f}")
    print()

    # =========================================================================
    # Checkpoint C: Experiment with depth and width
    # =========================================================================
    configs = [
        ("2 hidden layers, 128 wide", [
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        ]),
        ("3 hidden layers, 256 wide", [
            nn.Linear(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ]),
        ("4 hidden layers, 256 wide", [
            nn.Linear(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ]),
    ]

    print("Checkpoint C:")
    for name, layers in configs:
        torch.manual_seed(42)
        model = nn.Sequential(*layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

        for epoch in range(100):
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_hat = model(X)
            mse = loss_fn(y_hat, y)
        print(f"  {name}: Training MSE = {mse.item():.4f}")
