import torch
import torch.nn.functional as F
import numpy as np
#%%
# This script demonstrates how to compute the Cross Entropy Loss using PyTorch's built-in function
# and manually.
# Cross Entropy Loss is commonly used in classification tasks, especially with softmax outputs.
#
# ------------------------------
# 1. Define example logits and true labels.
# ------------------------------
# logits: raw outputs from the model for 2 samples and 3 classes.
logits = torch.tensor([[2.0, 1.0, 0.1], # sample 0 [0] should be 
                       [1.5, 0.5, 2.5]]) # sample 1
# True labels (indices) for each sample.
targets = torch.tensor([0, 2])

# ------------------------------
# 2. Compute Cross Entropy Loss using PyTorch's built-in function.
# ------------------------------
loss_builtin = F.cross_entropy(logits, targets)
print("Built-in Cross Entropy Loss:", loss_builtin.item())

# ------------------------------
# 3. Manual Computation of Cross Entropy Loss for the first sample.
# ------------------------------
# Let's take the first sample (index 0) as an example:
logits_sample = logits[0]  # tensor: [2.0, 1.0, 0.1]
target = targets[0]        # true class index: 0

# Step 1: Compute the softmax probabilities.
# The softmax function is given by:
# softmax(z_i) = exp(z_i) / sum_j(exp(z_j))
exp_logits = torch.exp(logits_sample)
softmax_sample = exp_logits / exp_logits.sum()
print("\nSoftmax probabilities for sample 0:", softmax_sample.numpy())

# Step 2: Compute the cross entropy loss manually.
# For the true class (target index 0):
loss_manual = -torch.log(softmax_sample[target])
print("Manually Computed Loss for Sample 0:", loss_manual.item())

# ------------------------------
# 4. (Optional) Manual Computation over the entire batch.
# ------------------------------
def manual_cross_entropy(logits, targets):
    losses = []
    for i in range(logits.shape[0]):
        logit = logits[i]
        target = targets[i]
        exp_logits = torch.exp(logit)
        softmax = exp_logits / exp_logits.sum()
        loss = -torch.log(softmax[target])
        losses.append(loss)
    return torch.stack(losses).mean()  # mean over the batch

loss_manual_batch = manual_cross_entropy(logits, targets)
print("\nManually Computed Loss for the Batch:", loss_manual_batch.item())

#%% pseudocode for one iteration to compute the gradients based on the cross entropy loss
""" 
# Example logits for two samples
logits = torch.tensor([[2.0, 1.0, 0.1],   # Sample 0, true label is 0
                        [1.5, 0.5, 2.5]])  # Sample 1, true label is 2

# True labels for the two samples
targets = torch.tensor([0, 2])

# Compute the cross entropy loss using PyTorch
loss = F.cross_entropy(logits, targets)
loss.backward()  # Backpropagation computes gradients for the logits

# The computed gradients will indicate that:
# - For sample 0: Gradient pushes logits[0,0] (correct) up.
# - For sample 1: Gradient pushes logits[1,2] (correct) up and logits[1,0] down.
"""

