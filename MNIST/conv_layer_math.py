import torch
import torch.nn.functional as F
import numpy as np

#%%
# This script demonstrates how to manually compute a 2D convolution operation
# using PyTorch's conv2d function and compare the results.
# It also shows how to include a bias term in the convolution operation.
#%%
# Without bias
# ------------------------------
# Create an input tensor of shape [batch_size, channels, height, width]
# In this example, we use a single 5x5 image with one channel.
input_tensor = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
print("Input Tensor:")
print(input_tensor)

# Define a 3x3 convolution kernel.
# Here, we're using a simple edge-detecting kernel.
kernel = torch.tensor(
    [[[[ 1,  0, -1],
       [ 1,  0, -1],
       [ 1,  0, -1]]]], dtype=torch.float32)
print("\nKernel:")
print(kernel)

# Compute convolution using PyTorch's F.conv2d.
# With stride=1 and padding=0, output spatial dimensions will be (5-3+1)=3 (i.e., 3x3 output).
output_pytorch = F.conv2d(input_tensor, kernel, bias=None, stride=1, padding=0)
print("\nOutput using PyTorch conv2d:")
print(output_pytorch)

# Now, manually compute the convolution.
# Convert input and kernel to numpy arrays.
input_np = input_tensor.numpy().squeeze()  # shape: (5, 5)
kernel_np = kernel.numpy().squeeze()         # shape: (3, 3)

# Determine the output spatial dimensions.
in_height, in_width = input_np.shape
k_height, k_width = kernel_np.shape
out_height = in_height - k_height + 1  # (5-3+1) = 3
out_width  = in_width - k_width + 1    # (5-3+1) = 3

# Initialize an array to hold the manual output.
output_manual = np.zeros((out_height, out_width), dtype=np.float32)

# Slide the kernel over the input image.
for i in range(out_height):
    for j in range(out_width):
        # Extract the current patch from the input.
        patch = input_np[i:i + k_height, j:j + k_width]
        # Compute element-wise multiplication and sum the result.
        output_manual[i, j] = np.sum(patch * kernel_np)

print("\nOutput computed manually:")
print(output_manual)

# Verify that both outputs match (within numerical tolerance).
assert np.allclose(output_pytorch.numpy().squeeze(), output_manual), "Outputs do not match!"
print("\nThe manually computed output matches the PyTorch conv2d output.")

#%% 
# When adding a bias term to the convolution, 
# we need to ensure that the bias is added after the convolution operation.
# ------------------------------
# 1. Create an input tensor
# ------------------------------
# Create an input tensor of shape [batch_size, channels, height, width]
# In this example, we use a single 5x5 image with one channel.
input_tensor = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
print("Input Tensor:")
print(input_tensor)

# ------------------------------
# 2. Define a 3x3 convolution kernel
# ------------------------------
# Here, we define a simple edge-detecting kernel.
kernel = torch.tensor(
    [[[[ 1,  0, -1],
       [ 1,  0, -1],
       [ 1,  0, -1]]]], dtype=torch.float32)
print("\nKernel:")
print(kernel)

# ------------------------------
# 3. Define the bias for the convolution
# ------------------------------
# We have one bias term for the filter, so the bias tensor has shape [1].
bias = torch.tensor([1.0], dtype=torch.float32)
print("\nBias:")
print(bias)

# ------------------------------
# 4. Compute convolution using PyTorch (including bias)
# ------------------------------
# With stride=1 and padding=0, the output spatial dimensions will be (5 - 3 + 1) = 3 (i.e., a 3x3 output).
output_pytorch = F.conv2d(input_tensor, kernel, bias=bias, stride=1, padding=0)
print("\nOutput using PyTorch conv2d (including bias):")
print(output_pytorch)

# ------------------------------
# 5. Manually compute the convolution (including bias)
# ------------------------------
# Convert input and kernel to numpy arrays.
input_np = input_tensor.numpy().squeeze()  # Shape becomes (5, 5)
kernel_np = kernel.numpy().squeeze()         # Shape becomes (3, 3)
bias_val = bias.item()                         # Convert bias tensor to a scalar

# Determine the output spatial dimensions.
in_height, in_width = input_np.shape
k_height, k_width = kernel_np.shape
out_height = in_height - k_height + 1  # 5 - 3 + 1 = 3
out_width  = in_width - k_width + 1    # 5 - 3 + 1 = 3

# Initialize an array to hold the manual output.
output_manual = np.zeros((out_height, out_width), dtype=np.float32)

# Slide the kernel over the input image.
for i in range(out_height):
    for j in range(out_width):
        # Extract the current 3x3 patch from the input.
        patch = input_np[i:i + k_height, j:j + k_width]
        # Perform element-wise multiplication, sum the result, then add the bias.
        conv_value = np.sum(patch * kernel_np) + bias_val
        output_manual[i, j] = conv_value

print("\nOutput computed manually (including bias):")
print(output_manual)

# ------------------------------
# 6. Verify that both outputs match
# ------------------------------
if np.allclose(output_pytorch.numpy().squeeze(), output_manual):
    print("\nThe manually computed output with bias matches the PyTorch conv2d output.")
else:
    print("\nThe outputs do not match!")

#%%
#
#  Now, let's compute the convolution with padding to keep the output size the same as the input size.
#
# ------------------------------
# 3. Compute convolution using PyTorch (with padding to keep same size)
# ------------------------------
# Use padding=1 for a 3x3 kernel so that the output size equals the input size (5x5)
output_pytorch = F.conv2d(input_tensor, kernel, bias=bias, stride=1, padding=1)
print("\nOutput using PyTorch conv2d (with padding=1):")
print(output_pytorch)

# ------------------------------
# 4. Manually compute the convolution (with padding)
# ------------------------------
# Convert input tensor to a numpy array and pad it.
input_np = input_tensor.numpy().squeeze()  # Shape: (5, 5)
# Pad the input: pad one pixel on each side, using constant value 0.
padded_input = np.pad(input_np, pad_width=1, mode='constant', constant_values=0)
print("\nPadded Input (numpy):")
print(padded_input)

in_height, in_width = input_np.shape
out_height, out_width = in_height, in_width

# Initialize an array to hold the manually computed output.
output_manual = np.zeros((out_height, out_width), dtype=np.float32)

# Compute the convolution manually by sliding the 3x3 kernel over the padded input.
for i in range(out_height):
    for j in range(out_width):
        # For each output position, extract the 3x3 patch from the padded input.
        patch = padded_input[i:i+3, j:j+3]
        # Perform element-wise multiplication, sum the results, then add the bias.
        conv_value = np.sum(patch * kernel_np) + bias_val
        output_manual[i, j] = conv_value

print("\nOutput computed manually (with padding):")
print(output_manual)

# ------------------------------
# 5. Verify that both outputs match
# ------------------------------
if np.allclose(output_pytorch.numpy().squeeze(), output_manual):
    print("\nThe manually computed output with bias and padding matches the PyTorch conv2d output.")
else:
    print("\nThe outputs do not match!")