import math
import torch
import torch.nn.functional as F
from lstm_gru import CharLSTM, encode_name

# Checkpoint B: Perplexity explorer
# Load your trained LSTM and see what surprises it.

checkpoint = torch.load("data/char_lstm.pt", weights_only=False)
char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]

model = CharLSTM(len(char_to_idx))
model.load_state_dict(checkpoint["model_state"])
model.eval()


# --- Generate some names ---

def generate(model, char_to_idx, idx_to_char, max_len=20, temperature=1.0):
    """Generate a single name by sampling one character at a time."""
    vocab_size = len(char_to_idx)

    idx = torch.randint(2, vocab_size, (1,)).item()
    result = [idx_to_char[idx]]

    x = torch.tensor([[idx]])
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            e = model.embed(x)
            out, hidden = model.rnn(e, hidden)
            logits = model.out(out)

            probs = F.softmax(logits[0, 0] / temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()

            if idx == char_to_idx["<eos>"]:
                break

            result.append(idx_to_char[idx])
            x = torch.tensor([[idx]])

    return "".join(result)


print("Generated names:")
for _ in range(10):
    print(f"  {generate(model, char_to_idx, idx_to_char)}")


# --- Perplexity explorer ---
# TODO: implement compute_perplexity and try it on the strings below.
#
# def compute_perplexity(model, string, char_to_idx):
#     """Compute perplexity of a single string under the model.
#
#     Steps:
#       1. Encode the string (use encode_name)
#       2. Create input (all chars except last) and target (all chars except first)
#       3. Run model forward, compute cross-entropy loss
#       4. Return math.exp(loss)
#     """
#     pass
#
#
# Test strings to try:
#   - A common name:    "james"
#   - An uncommon name: "zephyr"
#   - Not a name:       "xxxxx"
#   - A real word:      "table"
#   - Your own name!
#
# Questions to think about:
#   - Which strings have the lowest perplexity? Why?
#   - Does "james" or "zephyr" surprise the model more?
#   - What perplexity does "aaa" get? Why?
