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
    """Load names and build a character vocabulary.

    Returns:
        names:       list of name strings, e.g. ["emma", "olivia", ...]
        char_to_idx: dict mapping each character (and special tokens) to an
                     integer index, e.g. {'<pad>': 0, '<eos>': 1, 'a': 2, ...}
        idx_to_char: reverse dict mapping indices back to characters
    """
    with open(path, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    # Build vocab: all unique characters + special tokens
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
    # Sort by length for efficient batching
    encoded.sort(key=len)
    batches = []
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i : i + batch_size]
        max_len = max(len(s) for s in batch)
        # Input: all chars except last; Target: all chars except first
        inputs = [s[:-1] + [0] * (max_len - len(s)) for s in batch]
        targets = [s[1:] + [0] * (max_len - len(s)) for s in batch]
        batches.append(
            (torch.tensor(inputs, dtype=torch.long),
             torch.tensor(targets, dtype=torch.long))
        )
    return batches


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len) of character indices
        Returns: (batch, seq_len, vocab_size) logits
        """
        e = self.embed(x)          # (batch, seq_len, embed_dim)
        h, _ = self.rnn(e)         # (batch, seq_len, hidden_dim)
        logits = self.out(h)       # (batch, seq_len, vocab_size)
        return logits


def train(model, batches, epochs=10, lr=0.003):
    """Train the model and print loss each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        total_chars = 0
        for inputs, targets in batches:
            logits = model(inputs)  # (B, T, vocab_size)
            # Reshape for cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # ignore padding
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask = targets != 0
            total_loss += loss.item() * mask.sum().item()
            total_chars += mask.sum().item()
        print(f"Epoch {epoch + 1:2d}  loss = {total_loss / total_chars:.4f}")


if __name__ == "__main__":
    download_names()
    # names:       list[str]      — all names, e.g. ["emma", "olivia", ...]
    # char_to_idx: dict[str, int] — maps characters to indices, e.g. {'a': 2, 'b': 3, ...}
    # idx_to_char: dict[int, str] — maps indices back to characters (reverse of char_to_idx)
    names, char_to_idx, idx_to_char = load_names()

    # Checkpoint A: Explore the data
    # Explore the names dataset and answer:
    #   - How many names are there?
    #   - How many unique characters?
    #   - What is the longest name? The shortest?
    #   - Print a few example names and their encoded forms


    # Checkpoint B: Train the model
    # Uncomment the lines below to train the model and save weights.
    # Once training is done, move on to generate.py for Checkpoint C.

    # batches = make_batches(names, char_to_idx)
    # print(f"{len(batches)} batches")
    #
    # model = CharRNN(len(char_to_idx))
    # train(model, batches, epochs=10)
    #
    # torch.save({
    #     "model_state": model.state_dict(),
    #     "char_to_idx": char_to_idx,
    #     "idx_to_char": idx_to_char,
    # }, "data/char_rnn.pt")
    # print("Model saved to data/char_rnn.pt")
