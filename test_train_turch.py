import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Optional
import matplotlib.pyplot as plt

class OCRModel(nn.Module):
    def __init__(self, num_chars, img_width=130, img_height=50):
        super(OCRModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate features after CNN
        self.features_h = img_height // 4
        self.features_w = img_width // 4
        
        # Dense layer after CNN
        self.dense1 = nn.Linear(self.features_h * 64, 64)
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        
        # Output layer
        self.dense2 = nn.Linear(128, num_chars + 1)  # +1 for CTC blank
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Reshape for dense layer
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x = x.reshape(batch_size, -1, self.features_h * 64)
        
        # Dense layer
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.dense2(x)
        x = F.log_softmax(x, dim=2)
        
        return x

def predict_captcha(image_path: str, 
                   model: nn.Module, 
                   char_mappings: dict,
                   device: Optional[str] = None) -> str:
    """
    Predict text from a CAPTCHA image.
    
    Args:
        image_path: Path to the CAPTCHA image
        model: Trained OCR model
        char_mappings: Dictionary containing character mappings
        device: Device to run prediction on ('cuda' or 'cpu')
    
    Returns:
        Predicted text string
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Prepare image transform
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 130)),
        transforms.ToTensor(),
    ])
    
    # Load and transform image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Decode prediction
    predictions = decode_predictions(outputs, char_mappings['num_to_char'])
    return predictions[0]  # Return first prediction

def decode_predictions(outputs: torch.Tensor, 
                      num_to_char: dict,
                      max_length: Optional[int] = None) -> List[str]:
    """Decode model outputs to text."""
    predictions = []
    outputs = outputs.detach().cpu().numpy()
    
    for output in outputs:
        # Get argmax for each timestep
        pred = np.argmax(output, axis=1)
        
        # Remove repeated characters and blanks
        decoded = []
        prev_char = -1
        for p in pred:
            if p != 0 and p != prev_char:  # 0 is CTC blank
                decoded.append(p)
            prev_char = p
        
        # Truncate to max_length if specified
        if max_length is not None:
            decoded = decoded[:max_length]
        
        # Convert to text
        text = ''.join([num_to_char[d] for d in decoded])
        predictions.append(text)
    
    return predictions

def visualize_prediction(image_path: str, prediction: str):
    """Visualize image and its prediction."""
    image = Image.open(image_path)
    plt.figure(figsize=(10, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load the model
    print("Loading model and character mappings...")
    
    # Load character mappings
    char_mappings = torch.load('char_mappings.pth')
    num_chars = len(char_mappings['char_to_num'])
    
    # Initialize and load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OCRModel(num_chars=num_chars)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model = model.to(device)
    
    # Example: predict multiple images
    test_images = [
        'captcha/capcha_error/2aby[UNK]_20241104_200402.png',
        'captcha/capcha_ok/2a8ax.png',
        'captcha/capcha_ok/2b44g.png'
    ]
    
    print("\nMaking predictions...")
    for image_path in test_images:
        # Skip if file doesn't exist
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} - file not found")
            continue
            
        # Make prediction
        prediction = predict_captcha(
            image_path=image_path,
            model=model,
            char_mappings=char_mappings,
            device=device
        )
        
        print(f"\nImage: {image_path}")
        print(f"Predicted text: {prediction}")
        
        # # Visualize result
        # visualize_prediction(image_path, prediction)