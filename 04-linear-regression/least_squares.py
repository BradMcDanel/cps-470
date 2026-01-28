import torch


def load_house_data():
    """Load house prices and return X (features) and y (prices)."""
    # sqft (in 1000s), bedrooms, price (in $100k)
    data = [
        (1.0, 2, 2.0),
        (1.2, 2, 2.3),
        (1.5, 3, 2.85),
        (1.8, 3, 3.4),
        (2.0, 4, 4.0),
        (2.2, 4, 4.5),
        (2.5, 5, 5.2),
        (0.9, 1, 1.7),
    ]
    sqft = torch.tensor([d[0] for d in data])
    bedrooms = torch.tensor([float(d[1]) for d in data])
    price = torch.tensor([d[2] for d in data])
    return sqft, bedrooms, price


if __name__ == "__main__":
    sqft, bedrooms, price = load_house_data()
    n = len(price)

    # =========================================================================
    # Checkpoint A: Build the design matrix
    # =========================================================================
    # Create X with shape (n, 3): columns are [sqft, bedrooms, 1]
    # The column of 1s handles the bias term

    # TODO: build X by stacking sqft, bedrooms, and a column of ones
    # Hint: use torch.stack(..., dim=1) or torch.column_stack(...)
    X = None

    # TODO: reshape price to be a column vector y with shape (n, 1)
    y = None

    print("Checkpoint A:")
    print(f"  X shape: {X.shape}")  # should be (8, 3)
    print(f"  y shape: {y.shape}")  # should be (8, 1)
    print()

    # =========================================================================
    # Checkpoint B: Compute X^T X and X^T y
    # =========================================================================
    # Recall the formula: w* = (X^T X)^{-1} X^T y
    # We'll compute X^T X and X^T y, then solve the system

    # TODO: compute X^T X (should be 3x3)
    # Hint: X.T gives the transpose, @ is matrix multiply
    XtX = None

    # TODO: compute X^T y (should be 3x1)
    Xty = None

    print("Checkpoint B:")
    print(f"  X^T X shape: {XtX.shape}")  # should be (3, 3)
    print(f"  X^T y shape: {Xty.shape}")  # should be (3, 1)
    print()

    # =========================================================================
    # Checkpoint C: Solve for w
    # =========================================================================
    # We need to solve: X^T X w = X^T y
    # Use torch.linalg.solve(A, b) which solves Ax = b for x

    # TODO: solve for w using torch.linalg.solve
    w = None

    print("Checkpoint C:")
    print(f"  w_sqft     = {w[0].item():.3f}")
    print(f"  w_bedrooms = {w[1].item():.3f}")
    print(f"  bias       = {w[2].item():.3f}")
    print()

    # TODO: compute predictions y_hat = X @ w
    y_hat = None

    # TODO: compute MSE loss = mean((y_hat - y)^2)
    loss = None

    print(f"  MSE loss: {loss.item():.4f}")
