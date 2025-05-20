import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust to ResNet input
    transforms.ToTensor()
])

# Load the CIFAR-10 dataset 
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


from src.Models.AutoEncoder import Autoencoder 

from src.Utils.Training import Trainer

# Initialize the model
model = Autoencoder().to(device)

# Initialize the trainer
trainer = Trainer(model, train_loader, device=device, lr=1e-4)

# Train the model
trainer.train(num_epochs=10)

# Evaluate the reconstruction
trainer.evaluate_reconstruction(num_images=6)

# Save the model
torch.save(model.state_dict(), 'autoencoder.pth')

# Load the model
model.load_state_dict(torch.load('autoencoder.pth'))

# Evaluate the reconstruction again
trainer.evaluate_reconstruction(num_images=6)


