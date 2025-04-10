#%%
import torch
from torchvision import transforms
from PIL import Image

# Define the same CNN class you used during training.
# For example, if you defined it as follows:
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))   # [batch, 32, 14, 14]
        x = self.pool2(self.relu2(self.conv2(x)))     # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and load the state dictionary.
model = CNN().to(device)
model.load_state_dict(torch.load('saved_models/mnist_cnn.pth', map_location=device))
# Ensure the model is in evaluation mode.
model.eval()  # Set the model to evaluation mode

# Define the same transforms that were applied during training.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is single channel (for MNIST)
    transforms.Resize((28, 28)),                   # Resize to 28x28 pixels if necessary
    transforms.ToTensor(),                         # Convert the image to a PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))      # Normalize using the MNIST training data stats
])

# Load your image. 
image_path = 'n5_001.jpg'
image = Image.open(image_path)

# Preprocess the image.
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

# Move input to the same device as the model and make prediction
input_tensor = input_tensor.to(device)
with torch.no_grad():  # Disable gradient computations for inference
    output = model(input_tensor)
    # For classification, the predicted class is the one with the highest score.
    predicted_class = output.argmax(dim=1).item()

print("Predicted digit:", predicted_class)
