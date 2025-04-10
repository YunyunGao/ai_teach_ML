import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple transform to convert images to tensors.
transform = transforms.ToTensor()

# Load the MNIST training dataset.
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader that loads the entire training set.
train_loader = DataLoader(mnist_train, batch_size=len(mnist_train), shuffle=False)

# Retrieve all images in one batch.
data_iter = iter(train_loader)
images, _ = next(data_iter)

# Compute mean and std across all channels (only one channel for grayscale).
mean = images.mean().item()
std = images.std().item()

print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")