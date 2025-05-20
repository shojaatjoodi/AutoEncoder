
import torchvision.models as models
import torch.nn as nn

# 1. Encoder: Use Pretrained ResNet (e.g., ResNet18)
# This encoder will output feature maps instead of a single vector
class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512): 
        super(ResNetEncoder, self).__init__() 
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # Keep layers before avgpool
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)  # Output shape: [B, C, H, W]


# 2. Decoder: Upsampling the ResNet Features
# This decoder will take the feature maps and reconstruct the original image
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Assuming encoder output is [B, 512, 4, 4] for 128x128 input
            # Transpose Conv 1: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Transpose Conv 2: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Transpose Conv 3: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Transpose Conv 4: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Transpose Conv 5: 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output values are between 0 and 1
        )

    def forward(self, x):
        return self.decoder(x) # Output shape: [B, 3, 128, 128]
    

# 3. Autoencoder: Combine Encoder and Decoder 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded