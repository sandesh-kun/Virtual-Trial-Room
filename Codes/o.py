import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_0
import matplotlib.pyplot as plt

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
        squeezenet = squeezenet1_0(weights='imagenet')

        self.encoder = squeezenet.features
        self.center = ConvBlock(512, 1024)
        self.decoder = nn.ModuleList([
            ConvBlock(1536, 512),
            ConvBlock(768, 256),
            ConvBlock(384, 128),
            ConvBlock(192, 64)
        ])
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define num_classes and dims based on your training script
num_classes = 59
dims = (104, 157)

# Load the trained model
model = UNet(num_classes).to(device)

# Load pretrained SqueezeNet weights
squeezenet = models.squeezenet1_0(pretrained=True)
model.encoder = squeezenet.features

model.load_state_dict(torch.load('clothing_segmentation_model.pth', map_location=device))
model.eval()

# Load a test image
test_image_path = 'C:/Project/san.jpg'  # ensure the correct path to your test image
test_image = Image.open(test_image_path)
test_image = np.array(test_image.resize(dims))

# Convert test_image to have the right format
test_image = test_image.transpose((2, 0, 1))  # move the channel dimension to the first place
test_image = test_image.astype(np.float32) / 255.  # normalize to [0, 1] range

# Prepare the image for the model
test_image = torch.tensor(test_image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = model(test_image)
    predicted_mask = output.argmax(dim=1).squeeze().cpu().numpy()

# Visualize the predicted_mask as an RGB image and overlay it on the original image
plt.imshow(test_image.squeeze().cpu().numpy().transpose(1, 2, 0))  # adjust here to correctly display the image
plt.imshow(predicted_mask, alpha=0.7, cmap='jet')
plt.show()
