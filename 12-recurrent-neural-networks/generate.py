import torch
from char_rnn import CharRNN

# Checkpoint C: Generate new names
# Load the trained model and implement sampling to generate new names.
# Sample one character at a time: start with a random first character,
# feed it through the model, sample from the output distribution,
# feed that back in, repeat until <eos> or max length.


def generate(model, char_to_idx, idx_to_char, max_len=20, temperature=1.0):
    """Generate a single name by sampling."""
    # TODO: implement generation
    #   1. Start with a random character (or a specific one)
    #   2. Feed it through the model
    #   3. Sample from softmax(logits / temperature)
    #   4. Repeat until <eos> or max_len
    #   5. Return the decoded string
    #
    # Hint: you'll need to manage the hidden state manually.
    #   Use a loop and feed one character at a time.
    #   torch.multinomial is useful for sampling.
    pass


if __name__ == "__main__":
    # Load the saved model and vocabulary
    checkpoint = torch.load("data/char_rnn.pt", weights_only=False)
    char_to_idx = checkpoint["char_to_idx"]  # dict[str, int]
    idx_to_char = checkpoint["idx_to_char"]  # dict[int, str]

    model = CharRNN(len(char_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Generated names:")
    for _ in range(10):
        name = generate(model, char_to_idx, idx_to_char)
        if name:
            print(f"  {name}")

    # Try different temperatures:
    #   temperature < 1.0 -> more conservative (common names)
    #   temperature > 1.0 -> more creative (weird names)
    # What do you notice?
