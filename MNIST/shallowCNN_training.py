import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import os

# Set device for training: use GPU if available, else default to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Define a transform to convert the images to tensors and normalize them.
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors (shape [C, H, W] scaled between 0 and 1)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST dataset mean and standard deviation
])

# Download and prepare the MNIST training and test datasets.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a Convolutional Neural Network model for MNIST classification.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolution block: input channels=1, output channels=32, kernel_size=3x3.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution block: input channels=32, output channels=64.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers:
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for digits 0-9
        
    def forward(self, x):
        # Pass input through the first conv block.
        x = self.conv1(x)    # [batch, 32, 28, 28]
        x = self.relu1(x)
        x = self.pool1(x)    # [batch, 32, 14, 14]
        
        # Pass through the second conv block.
        x = self.conv2(x)    # [batch, 64, 14, 14]
        x = self.relu2(x)
        x = self.pool2(x)    # [batch, 64, 7, 7]
        
        # Flatten feature map.
        x = x.view(-1, 64 * 7 * 7)
        
        # Pass through fully connected layers.
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Initialize the model and move it to the appropriate device.
model = CNN().to(device)
print(model)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function: trains the network for one epoch.
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Set model to training mode.
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device.
        images, labels = images.to(device), labels.to(device)
        
        # Clear the gradients.
        optimizer.zero_grad()
        
        # Forward pass.
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize.
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] training complete. Average Loss: {avg_loss:.4f}")

# Evaluation function: computes the accuracy on the test dataset.
def evaluate(model, device, test_loader):
    model.eval()  # Set model to evaluation mode.
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Get predictions from the maximum value.
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (100 * correct / total)
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

def main():
    # Create directory to save the model if it doesn't exist.
    os.makedirs("saved_models", exist_ok=True)
    
    # Train the network.
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, test_loader)
    
    # Save the trained model.
    model_path = "saved_models/mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    main()