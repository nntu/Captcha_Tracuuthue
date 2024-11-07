import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os

# Define image dimensions and transforms
img_height = 50
img_width = 130
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

# Load and transform image
img_path = "captcha/capcha_ok/2a2n5.png"
img = Image.open(img_path)
img_tensor = transform(img)

# Display original and transformed images side by side
plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Transformed image
plt.subplot(1, 2, 2)
plt.imshow(img_tensor.squeeze(), cmap='gray')  # squeeze removes the channel dimension
plt.title(f'Transformed Image ({img_height}x{img_width})')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print tensor information
print(f"Tensor shape: {img_tensor.shape}")
print(f"Value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")