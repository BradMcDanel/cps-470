import torch


# Checkpoint A: Implement Uniform Quantization
#
# Implement the quantize and dequantize functions we saw in lecture.
#
# Quantize:   q = round(x / S + Z),  clamped to [0, 2^bits - 1]
# Dequantize: x_hat = (q - Z) * S
#
# For symmetric quantization:
#   alpha = max(|x|)
#   S = alpha / (2^(bits-1) - 1)
#   Z = 2^(bits-1)

def quantize(x, bits):
    """Quantize a float tensor to the given bit-width (symmetric).

    Returns: (q, S, Z) where q is an integer tensor, S is the scale, Z is the zero-point.
    """
    # 1. compute alpha (max absolute value)
    # 2. compute scale S
    # 3. compute zero-point Z (center of the integer range)
    # 4. compute q = round(x / S + Z), clamped to [0, 2^bits - 1]
    alpha = None
    S = None
    Z = None
    q = None

    return q, S, Z


def dequantize(q, S, Z):
    """Dequantize an integer tensor back to float.

    Returns: x_hat (float tensor).
    """
    # x_hat = (q - Z) * S
    x_hat = None

    return x_hat


if __name__ == "__main__":
    torch.manual_seed(42)

    # Create a sample weight tensor (like a small layer)
    weights = torch.randn(4, 4) * 0.5
    print("Original weights:")
    print(weights)
    print()

    # Checkpoint A: Quantize to 8-bit and dequantize
    # 1. call quantize() with bits=8
    # 2. call dequantize() to get x_hat
    # 3. print the quantized integers, dequantized values, and max absolute error

    print("Checkpoint A: 8-bit quantization")
    # q, S, Z = ...
    # x_hat = ...
    # print(f"  Scale S = {S:.6f}, Zero-point Z = {Z}")
    # print(f"  Quantized integers:\n{q}")
    # print(f"  Dequantized:\n{x_hat}")
    # print(f"  Max error: {(weights - x_hat).abs().max():.6f}")
    print()

    # Checkpoint B: Quantization Error vs. Bit-Width
    #
    # Loop over bit-widths from 8 down to 1. For each:
    #   1. Quantize and dequantize the weights
    #   2. Compute the mean squared error (MSE) vs. the original
    #
    # Print or plot the results.

    print("Checkpoint B: Error vs. bit-width")
    bit_widths = [8, 7, 6, 5, 4, 3, 2, 1]

    # Use a larger random tensor for smoother results
    weights_large = torch.randn(256, 256) * 0.5

    for bits in bit_widths:
        # 1. quantize and dequantize weights_large
        # 2. compute MSE between original and dequantized
        # 3. print or plot the MSE
        pass
