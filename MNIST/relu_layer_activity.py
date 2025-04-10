import torch
import torch.nn as nn

#%%
# This script demonstrates how to analyze the activation of neurons in a simple convolutional neural network (CNN) using PyTorch.
# It focuses on the ReLU activation function, which is commonly used in CNNs.
# The example uses a small CNN with one convolutional layer followed by a ReLU activation.
# The goal is to visualize the activation of neurons after applying the ReLU function.

#%%
# Define a small network component: a conv layer followed by a ReLU.
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # conv1: converts a 1-channel input to 32 channels using a 3x3 kernel and padding=1.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # relu1: the ReLU activation function.
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        # Compute convolution.
        x = self.conv1(x)
        # Apply ReLU activation.
        x = self.relu1(x)
        return x

# Instantiate the model.
model = SimpleConvNet()

# Create an example input: a single 28x28 grayscale image.
input_tensor = torch.randn(1, 1, 28, 28)  # Using random input for demonstration

# Run the input through the conv1 and relu1 layers.
with torch.no_grad():  # No gradient calculations needed for inference/analysis.
    conv_output = model.conv1(input_tensor)
    activation = model.relu1(conv_output)

# Detect neuron activity: count all activation values > 0.
active_neurons = (activation > 0).sum().item()
total_neurons = activation.numel()

print("Activation tensor shape:", activation.shape)
print(f"Active neurons: {active_neurons} out of {total_neurons}")

# Optionally, you can print a summary of the activations.
# For instance, compute the percentage of active neurons.
active_percentage = 100.0 * active_neurons / total_neurons
print(f"Percentage of active neurons: {active_percentage:.2f}%")
#%%
# This script visualizes the activation of neurons in a simple CNN using PyTorch.
# It uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits.
# The script defines a simple CNN model, loads the MNIST dataset, trains the model, and records the activations of the neurons after applying the ReLU activation function.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Define a Simple CNN Model
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # conv1: converts 1-channel input to 32 channels using a 3x3 kernel with padding=1.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # relu1: ReLU activation
        self.relu1 = nn.ReLU()
        # Additional layer: a fully connected layer for classification.
        # For simplicity, we flatten the output from conv1 (which has shape [batch, 32, 28, 28])
        # so the fully connected layer input size is 32*28*28.
        self.fc = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        # Save the activation from relu1 for analysis later if needed.
        x = self.relu1(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# ------------------------------
# 2. Load MNIST Dataset and Prepare DataLoader
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------------------------
# 3. Set Up the Model, Loss, and Optimizer
# ------------------------------
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ------------------------------
# 4. Train the Model and Record Activations
# ------------------------------
num_epochs = 10
# Initialize an array to store the average activation of each of the 32 neurons (filters)
# for every epoch.
activations_record = np.zeros((num_epochs, 32))

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Record activations on the first batch of each epoch only.
        if batch_idx == 5:
            # Switch to evaluation mode to turn off dropout, etc.
            model.eval()
            with torch.no_grad():
                # Forward pass through conv1 and relu1 only.
                conv_output = model.conv1(data)
                relu_output = model.relu1(conv_output)
                # Compute the average activation per neuron (i.e. per output channel).
                # The activation tensor shape is [batch, 32, 28, 28].
                # Average over batch and spatial dimensions (dim 0, 2, and 3).
                avg_activation = relu_output.mean(dim=(0, 2, 3))
                # Record these average activations (result is a tensor of shape [32]).
                activations_record[epoch, :] = avg_activation.cpu().numpy()
            # Only record for one batch per epoch.
            break
    print(f"Epoch {epoch+1}/{num_epochs} processed.")

# ------------------------------
# 5. Visualize the Activation Evolution Across Epochs
# ------------------------------
plt.figure(figsize=(10, 6))
# Use imshow to display the 2D heatmap: rows correspond to epochs, columns to neurons.
plt.imshow(activations_record, aspect='auto', cmap='viridis')
plt.colorbar(label='Average Activation')
plt.xlabel('Neuron (Filter) Index')
plt.ylabel('Epoch')
plt.title('Evolution of conv1 ReLU Activations Across Epochs')
plt.show()