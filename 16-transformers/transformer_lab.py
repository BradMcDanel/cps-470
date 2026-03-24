import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


MODEL_NAME = "distilgpt2"


def load_model():
    """Load distilgpt2 and its tokenizer."""
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    )
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {MODEL_NAME}: {params/1e6:.0f}M parameters")
    print(f"  {model.config.n_layer} layers, {model.config.n_head} heads, "
          f"d_model={model.config.n_embd}")
    return model, tokenizer


# Checkpoint A: Attention Patterns
#
# Extract attention weights from distilgpt2 and visualize as heatmaps.
# The model returns attention when called with output_attentions=True.
# outputs.attentions is a tuple of tensors, one per layer,
# each with shape (batch, num_heads, seq_len, seq_len).

def get_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = [tokenizer.decode(t) for t in inputs["input_ids"][0]]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    return tokens, outputs.attentions


def plot_attention(tokens, attentions, layer, head):
    # attentions[layer] has shape (1, num_heads, seq_len, seq_len)
    attn = attentions[layer][0, head].numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(attn, cmap="Blues")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Key (attends to)")
    ax.set_ylabel("Query (token)")
    ax.set_title(f"Layer {layer}, Head {head}")
    plt.tight_layout()
    plt.savefig(f"attention_L{layer}_H{head}.png", dpi=150)
    plt.show()
    print(f"Saved to attention_L{layer}_H{head}.png")


# Checkpoint B: Greedy Generation
#
# Implement greedy autoregressive generation from scratch.
# At each step: run the model, take the last position's logits,
# argmax to pick the next token, append, repeat.

def greedy_generate(model, tokenizer, prompt, max_new_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)

        # TODO: get logits for the last position
        # outputs.logits shape: (1, seq_len, vocab_size)
        next_logits = None

        # TODO: pick the most likely next token
        next_token = None

        # TODO: append next_token to input_ids using torch.cat
        # next_token needs shape (1, 1) via .unsqueeze(0).unsqueeze(0)
        input_ids = None

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])


# Checkpoint C: Generation with KV Cache
#
# Same as B, but reuse cached keys/values instead of recomputing.
# The model accepts past_key_values and returns updated ones.
# Only feed the NEW token each step (not the full sequence).

def greedy_generate_with_cache(model, tokenizer, prompt, max_new_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    past = None
    all_token_ids = input_ids

    for i in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past, use_cache=True)

        # TODO: get logits for the last position
        next_logits = None

        # TODO: pick the most likely next token
        next_token = None

        # TODO: update the KV cache from outputs.past_key_values
        past = None

        # Only feed the new token next iteration
        next_token_2d = next_token.unsqueeze(0).unsqueeze(0)
        input_ids = next_token_2d
        all_token_ids = torch.cat([all_token_ids, next_token_2d], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(all_token_ids[0])


if __name__ == "__main__":
    model, tokenizer = load_model()

    # --- Checkpoint A ---
    print("\nCheckpoint A: Attention Patterns")
    prompt = input("Enter a sentence for attention analysis: ")
    tokens, attentions = get_attention(model, tokenizer, prompt)
    print(f"Tokens: {tokens}")

    plot_attention(tokens, attentions, layer=0, head=0)

    # TODO: try other layer/head combinations.
    # distilgpt2 has 6 layers (0-5) and 12 heads (0-11).
    #   - Which heads attend to the previous token?
    #   - Which heads attend to the first token?
    #   - Do deeper layers look different from early ones?

    # --- Checkpoint B ---
    print("\nCheckpoint B: Greedy Generation")
    prompt = input("Enter a prompt to complete: ")
    # TODO: uncomment once you've implemented greedy_generate
    # t0 = time.time()
    # text = greedy_generate(model, tokenizer, prompt, max_new_tokens=50)
    # t1 = time.time()
    # print(f"Generated in {t1-t0:.2f}s:")
    # print(text)

    # --- Checkpoint C ---
    print("\nCheckpoint C: KV Cache Generation")
    # TODO: uncomment once you've implemented greedy_generate_with_cache
    # t0 = time.time()
    # text_cached = greedy_generate_with_cache(model, tokenizer, prompt, max_new_tokens=50)
    # t1 = time.time()
    # print(f"Generated in {t1-t0:.2f}s:")
    # print(text_cached)
    # print(f"Outputs match: {text == text_cached}")
