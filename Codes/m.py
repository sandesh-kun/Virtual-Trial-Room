import glob
import numpy as np
from PIL import Image
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import squeezenet1_0
from torchvision.transforms import Resize
from torch.nn import functional as F

# Load images
images_path = glob.glob('~/Downloads/clothing-co-parsing-master/photos/*')[:1004]
images = [np.array(Image.open(img_path)) for img_path in images_path]

# Load masks
masks_path = glob.glob('~/Downloads/clothing-co-parsing-master/annotations/pixel-level/*')
masks = [io.imread(mask_path) for mask_path in masks_path]

# Set dimensions
dims = (104, 157)

# Resize images and masks
resized_images = [resize(img, dims, mode='reflect', anti_aliasing=True, preserve_range=True) for img in images]
resized_masks = [resize(mask, dims, mode='reflect', anti_aliasing=True, preserve_range=True) for mask in masks]

# Convert images and masks to integer type
resized_images = [img.astype(np.uint8) for img in resized_images]
resized_masks = [mask.astype(np.uint8) for mask in resized_masks]

# Prepare data for neural network
data = list(zip(resized_images, resized_masks))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # Load SqueezeNet model
        squeezenet = squeezenet1_0(pretrained=True)
        self.encoder = squeezenet.features

        # Center
        self.center = ConvBlock(512, 1024)

        # Decoder
        self.decoder = nn.ModuleList([
            ConvBlock(1536, 512),
            ConvBlock(768, 256),
            ConvBlock(384, 128),
            ConvBlock(192, 64)
        ])

        # Final output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        activations = []
        for i, m in enumerate(self.encoder):
            x = m(x)
            if isinstance(m, nn.ReLU):
                activations.append(x)

        x = self.center(x)

        for i, decoder in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, activations[-i-2]], dim=1)
            x = decoder(x)

        x = self.final(x)
        x = F.softmax(x, dim=1)

        return x

# Initialize the model
num_classes = 59
model = UNet(num_classes).to(device)

# Define the optimizer and the loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):  # 5 epochs for example
    for inputs, targets in data:  # assuming data is your dataset with the training data
        inputs, targets = torch.tensor(inputs).unsqueeze(0).to(device), torch.tensor(targets).unsqueeze(0).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'clothing_segmentation_model.pth')
