

# Training Loop

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, device="cuda", lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, _ in self.train_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, images)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}")

    def evaluate_reconstruction(self, num_images=6):
        self.model.eval()
        with torch.no_grad():
            images, _ = next(iter(self.train_loader))
            images = images.to(self.device)
            outputs = self.model(images)
            images = images.cpu()
            outputs = outputs.cpu()

            fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
            for i in range(num_images):
                axes[0, i].imshow(images[i].permute(1, 2, 0))
                axes[1, i].imshow(outputs[i].permute(1, 2, 0))
                axes[0, i].axis('off')
                axes[1, i].axis('off')
            plt.show()
