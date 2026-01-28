import torch


def make_sine_data(n_samples=50):
    """Generate noisy sine wave data."""
    torch.manual_seed(42)
    x = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.sin(x) + 0.5 * torch.sin(2 * x) + torch.randn(n_samples, 1) * 0.1
    return x, y


if __name__ == "__main__":
    X, y = make_sine_data()
    n, d = X.shape
    hidden_size = 10

    print(f"Data: {n} samples, {d} features")
    print(f"Hidden size: {hidden_size}")
    print()

    # =========================================================================
    # Checkpoint A: Define the parameters
    # =========================================================================
    # Layer 1: takes d inputs, produces hidden_size outputs
    # Layer 2: takes hidden_size inputs, produces 1 output
    #
    # Use (torch.randn(...) * 0.1).requires_grad_(True) for weights
    # Use torch.zeros(...).requires_grad_(True) for biases

    # TODO: w1 should have shape (d, hidden_size)
    w1 = None

    # TODO: b1 should have shape (hidden_size,)
    b1 = None

    # TODO: w2 should have shape (hidden_size, 1)
    w2 = None

    # TODO: b2 should have shape (1,)
    b2 = None

    print("Checkpoint A:")
    print(f"  w1 shape: {w1.shape}")  # should be (1, 10)
    print(f"  b1 shape: {b1.shape}")  # should be (10,)
    print(f"  w2 shape: {w2.shape}")  # should be (10, 1)
    print(f"  b2 shape: {b2.shape}")  # should be (1,)
    print()

    # =========================================================================
    # Checkpoint B: Define the forward pass
    # =========================================================================
    # Layer 1: h = sigmoid(X*w1 + b1)
    # Layer 2: y_hat = h*w2 + b2  (no sigmoid on output for regression!)
    # 
    # Use torch.sigmoid() for the hidden layer activation
    # Instead of * above you need to use matmul

    def forward(X):
        # TODO: compute h (hidden layer output)
        h = None

        # TODO: compute y_hat (final prediction - no activation!)
        y_hat = None

        return y_hat

    y_hat = forward(X)
    print("Checkpoint B:")
    print(f"  y_hat shape: {y_hat.shape}")  # should be (50, 1)
    print()

    # =========================================================================
    # Training loop (provided); no need to modify.
    # We will go over optimizers soon..!
    # =========================================================================
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=0.1)

    for step in range(1000):
        y_hat = forward(X)
        loss = ((y_hat - y) ** 2).mean()  # MSE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    print(f"\nFinal MSE: {loss.item():.4f}")
