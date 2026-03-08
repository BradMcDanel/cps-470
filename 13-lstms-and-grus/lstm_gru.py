import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

DATA_FILE = "data/names.txt"


def download_names():
    """Download a list of names if not already present."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DATA_FILE):
        return
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    urllib.request.urlretrieve(url, DATA_FILE)


def load_names(path=DATA_FILE):
    """Load names and build a character vocabulary."""
    with open(path, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    chars = sorted(set("".join(names)))
    char_to_idx = {c: i + 2 for i, c in enumerate(chars)}
    char_to_idx["<pad>"] = 0
    char_to_idx["<eos>"] = 1
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return names, char_to_idx, idx_to_char


def encode_name(name, char_to_idx):
    """Encode a name as a list of character indices, ending with <eos>."""
    return [char_to_idx[c] for c in name] + [char_to_idx["<eos>"]]


def make_batches(names, char_to_idx, batch_size=64):
    """Create batches of padded sequences. Returns (input, target) pairs."""
    encoded = [encode_name(n, char_to_idx) for n in names]
    encoded.sort(key=len)
    batches = []
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i : i + batch_size]
        max_len = max(len(s) for s in batch)
        inputs = [s[:-1] + [0] * (max_len - len(s)) for s in batch]
        targets = [s[1:] + [0] * (max_len - len(s)) for s in batch]
        batches.append(
            (torch.tensor(inputs, dtype=torch.long),
             torch.tensor(targets, dtype=torch.long))
        )
    return batches


# =============================================================================
# Checkpoint A: Build a CharLSTM
# =============================================================================
# Last time we used nn.RNN. Now swap it for nn.LSTM.
# Two things change:
#   1. Replace nn.RNN with nn.LSTM
#   2. LSTM returns (output, (h_n, c_n)) instead of (output, h_n)
#      You need to handle the tuple hidden state.

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # TODO: change nn.RNN to nn.LSTM
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        h, _ = self.rnn(e)
        logits = self.out(h)
        return logits


def train(model, batches, epochs=10, lr=0.003):
    """Train the model. Returns final average loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        total_chars = 0
        for inputs, targets in batches:
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask = targets != 0
            total_loss += loss.item() * mask.sum().item()
            total_chars += mask.sum().item()
        avg_loss = total_loss / total_chars
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch + 1:2d}  loss = {avg_loss:.4f}  perplexity = {ppl:.2f}")
    return avg_loss


if __name__ == "__main__":
    download_names()
    names, char_to_idx, idx_to_char = load_names()
    batches = make_batches(names, char_to_idx)
    vocab_size = len(char_to_idx)

    # --- Checkpoint A: Train the LSTM ---
    print("=" * 50)
    print("Training LSTM")
    print("=" * 50)
    # TODO: uncomment once you've implemented CharLSTM
    # model = CharLSTM(vocab_size)
    # loss = train(model, batches, epochs=10)
    #
    # torch.save({
    #     "model_state": model.state_dict(),
    #     "char_to_idx": char_to_idx,
    #     "idx_to_char": idx_to_char,
    # }, "data/char_lstm.pt")
    # print("Model saved to data/char_lstm.pt")
