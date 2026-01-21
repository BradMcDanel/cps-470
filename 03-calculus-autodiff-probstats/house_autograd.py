import csv
import os

import torch


def load_house_data(*, dtype=torch.float32):
    csv_path = os.path.join(os.path.dirname(__file__), "house_prices_small.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    x1 = torch.tensor([float(r["sqft"]) / 1000.0 for r in rows], dtype=dtype)
    x2 = torch.tensor([float(r["bedrooms"]) for r in rows], dtype=dtype)
    y = torch.tensor([float(r["price"]) / 100000.0 for r in rows], dtype=dtype)
    return x1, x2, y


if __name__ == "__main__":
    # Checkpoint A: match the slides
    x = torch.tensor([1.0, 2.0])  # (x1, x2)
    y = torch.tensor(2.0)
    w = torch.tensor([2.0, 0.5], requires_grad=True)  # (w1, w2)
    eta = 0.1

    # TODO: compute loss
    loss = None
    loss.backward()
    print("checkpoint A:", "loss", loss.item(), "grad", w.grad.tolist())

    # SGD update step
    with torch.no_grad():
        w -= eta * w.grad
    w.grad.zero_()

    print("checkpoint A:", "w", w.tolist())

    # Checkpoint B: one SGD step on the dataset
    x1, x2, y_data = load_house_data()
    w = torch.tensor([2.0, 0.5], requires_grad=True)

    # TODO: compute a scalar loss on the dataset
    loss = None
    loss_before = loss.item()
    loss.backward()

    with torch.no_grad():
        w -= eta * w.grad
    w.grad.zero_()

    # TODO: recompute the scalar loss after the update
    loss = None
    loss_after = loss.item()

    print("checkpoint B:", "loss", (loss_before, loss_after), "w", w.tolist())

    # Checkpoint C: training loop
    w = torch.tensor([2.0, 0.5], requires_grad=True)
    steps = 100
    for step in range(steps):
        # TODO: do Checkpoint B inside this loop (loss -> backward -> SGD step -> zero grads)

        # This prints the loss (should decrease over steps)
        if step % 10 == 0:
            print("loss:", loss.item())
