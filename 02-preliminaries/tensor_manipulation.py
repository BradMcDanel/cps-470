"""
Tensor Manipulation Exercises
"""

import torch

# SECTION 1: Creating Tensors
print("=" * 40)
print("SECTION 1: Creating Tensors")
print("=" * 40)

# Create a 1D tensor of 5 ones
ones_1d = None

# Create a 2D tensor of ones with shape (3, 4)
ones_2d = None

# Create a 3D tensor of ones with shape (2, 3, 4)
ones_3d = None

print("ones_1d:", ones_1d)
print("ones_2d:")
print(ones_2d)
print("ones_3d:")
print(ones_3d)

# Create a tensor with values 0 through 11 using arange
seq = None
print("\nsequence:", seq)

# Create a 2D tensor of random values with shape (3, 4)
rand_2d = None
print("random 2D:")
print(rand_2d)


# SECTION 2: Reshaping
print("\n" + "=" * 40)
print("SECTION 2: Reshaping")
print("=" * 40)

# Reshape seq (from above) into a 3x4 matrix and print the shape
reshaped = None

# Reshape seq into a 2x2x3 tensor and print the shape
reshaped_3d = None


# SECTION 3: Indexing
print("\n" + "=" * 40)
print("SECTION 3: Indexing")
print("=" * 40)

matrix = torch.arange(12).reshape(3, 4)
print("matrix:")
print(matrix)

# Get the first row
first_row = None
print("\nfirst row:", first_row)

# Get the element at row 1, column 2
elem = None
print("element at [1,2]:", elem)

cube = torch.arange(24).reshape(2, 3, 4)
print("\ncube:")
print(cube)

# Get the first 3x4 slice
first_slice = None
print("\nfirst slice:", first_slice)

# Get element at position [1, 0, 2]
elem_3d = None
print("element at [1,0,2]:", elem_3d)


# SECTION 4: Slicing and Assignment
print("\n" + "=" * 40)
print("SECTION 4: Slicing and Assignment")
print("=" * 40)

data = torch.zeros(4, 4)
print("starting tensor:")
print(data)

# Set the entire second row to 5

# Set the entire third column to 7

# Set the bottom-right 2x2 region to 9
