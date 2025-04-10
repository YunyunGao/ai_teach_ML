import torch
from captum.attr import IntegratedGradients
import torchvision.transforms as transforms
from PIL import Image
#%% load the model
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
model.eval()
#%%

# Prepare an example input image (for MNIST, a grayscale 28x28 image)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
image = Image.open('path_to_mnist_sample.png')
input_tensor = transform(image).unsqueeze(0)  # shape: [1, 1, 28, 28]

# Create an IntegratedGradients object
ig = IntegratedGradients(model)

# Compute attributions for the predicted class
attributions, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)

# Visualize the attributions (you could convert to numpy and use matplotlib)
print("Attributions computed using Integrated Gradients:")
print(attributions)