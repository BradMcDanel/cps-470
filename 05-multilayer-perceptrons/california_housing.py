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
    # Standardize each feature to mean 0, standard deviation 1.
    # X = (X - mean) / std
    #
    # Hint: X.mean(dim=0) and X.std(dim=0) give per-feature statistics

    # TODO: normalize X

    print("Checkpoint A:")
    print(f"  Feature means:  {X.mean(dim=0).tolist()}")  # should be ~0
    print(f"  Feature stds:   {X.std(dim=0).tolist()}")   # should be ~1
    print()

    # =========================================================================
    # Checkpoint B: Build and train a single hidden layer MLP
    # =========================================================================
    # Use nn.Sequential to build: Linear(8, 64) -> ReLU -> Linear(64, 1)
    # Loss: nn.MSELoss()
    # Optimizer: torch.optim.SGD(model.parameters(), lr=0.02)
    # Train for 100 epochs (full-batch, no DataLoader needed)

    # TODO: define model
    model = None

    # TODO: define loss function and optimizer
    loss_fn = None
    optimizer = None

    # TODO: training loop
    # for epoch in range(100):
    #     1. forward pass: y_hat = model(X)
    #     2. compute loss: loss = loss_fn(y_hat, y)
    #     3. backward: optimizer.zero_grad(), loss.backward(), optimizer.step()
    #     4. print loss every 20 epochs

    print("Checkpoint B:")
    with torch.no_grad():
        y_hat = model(X)
        mse = loss_fn(y_hat, y)
        print(f"  Training MSE: {mse.item():.4f}")
    print()

    # =========================================================================
    # Checkpoint C: Experiment with depth and width
    # =========================================================================
    # Try different architectures and document what you find.
    # Suggested experiments:
    #   - 2 hidden layers, 128 neurons each
    #   - 3 hidden layers, 256 neurons each
    #   - 4 hidden layers, 256 neurons each
    #
    # For each: rebuild the model, reset the optimizer,
    # train for 100 epochs, and record the final training MSE.
    #
    # Questions to answer:
    #   - Does adding layers help?
    #   - Does adding width help?
    #   - What is the best architecture you can find?

    # TODO: your experiments here
