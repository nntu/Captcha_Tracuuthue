import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
from trainv2 import CAPTCHADataset, GridCAPTCHANet
def predict_captcha(model, image_path, device):
    """
    Predict CAPTCHA value from an image using the trained model.
    
    Args:
        model: Trained GridCAPTCHANet model
        image_path: Path to the CAPTCHA image
        device: torch.device for computation
    
    Returns:
        str: Predicted CAPTCHA text
    """
    model.eval()
    
    # Character mapping
    idx_to_char = {}
    # Add digits
    for i in range(10):
        idx_to_char[i] = str(i)
    # Add uppercase letters
    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
        idx_to_char[i + 10] = char
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold and clean
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Convert to PIL and apply transforms
    image = Image.fromarray(cleaned)
    image = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image)
        predictions = [output.max(1)[1].item() for output in outputs]
        predicted_chars = [idx_to_char[idx] for idx in predictions]
    
    return ''.join(predicted_chars)
 




def load_and_predict():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = GridCAPTCHANet(num_chars=5, num_classes=36).to(device)
    model.load_state_dict(torch.load('grid_captcha_model.pth', map_location=device))
    model.eval()
    
    # Get all PNG files in current directory
    image_files = list(Path('.').glob('*.jpg'))
    
    # Predict each image
    for img_path in image_files:
        try:
            prediction = predict_captcha(model, str(img_path), device)
            print(f"Image: {img_path.name}, Predicted CAPTCHA: {prediction}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    load_and_predict()