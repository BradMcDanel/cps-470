import torch
import torch.nn.functional as F
import urllib.request
import os


# =============================================================================
# Load pretrained GloVe embeddings (50-dimensional, ~163 MB)
# =============================================================================

GLOVE_FILE = "data/glove.6B.50d.txt"


def download_glove():
    """Download GloVe 50d embeddings (~163 MB) if not already present."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(GLOVE_FILE):
        return
    print("Downloading glove.6B.50d.txt (~163 MB)...")
    url = "https://huggingface.co/kcz358/glove/resolve/main/glove.6B/glove.6B.50d.txt"
    urllib.request.urlretrieve(url, GLOVE_FILE)
    print("Done.")


def load_glove(path=GLOVE_FILE):
    """Load GloVe vectors into a word->index dict and a weight matrix."""
    words = []
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    word_to_idx = {w: i for i, w in enumerate(words)}
    weights = torch.tensor(vectors, dtype=torch.float32)
    return word_to_idx, words, weights


if __name__ == "__main__":
    download_glove()
    word_to_idx, words, weights = load_glove()
    print(f"Loaded {len(words)} words, embedding dim = {weights.shape[1]}")

    # =========================================================================
    # Checkpoint A: Cosine similarity between word pairs
    # =========================================================================
    # Look up embedding vectors for words using:
    #   weights[word_to_idx["cat"]]   -> tensor of shape (50,)
    #
    # Compute cosine similarity using F.cosine_similarity().
    #   Hint: it expects 2D inputs, so unsqueeze first.
    #
    # Pairs to try: (cat, dog), (cat, car), (king, queen), (paris, france)
    # Which pairs are most/least similar? Does this match your intuition?

    # =========================================================================
    # Checkpoint B: Word analogy
    # =========================================================================
    # Implement: a is to b as c is to ?
    #
    # Steps:
    #   1. Compute: result = e_b - e_a + e_c
    #   2. Find the word whose embedding is most similar to result
    #      (excluding a, b, c from the search)
    #
    # def analogy(a, b, c):
    #     ...
    #
    # Test: man:woman :: king:?       (expect queen)
    #       paris:france :: tokyo:?   (expect japan)
    #       slow:slower :: big:?      (expect bigger)

    # =========================================================================
    # Checkpoint C: Nearest neighbors
    # =========================================================================
    # Write a function that finds the k nearest words to a given word.
    #
    # Hint: F.cosine_similarity can compare one vector against all rows
    #       of the weight matrix at once if you unsqueeze correctly.
    #
    # Try it on a few words. Do the neighbors make sense?
    # Try to find an analogy that fails. Why might it fail?

    pass
