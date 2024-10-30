import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os

# Custom Dataset for CAPTCHA images
class CAPTCHADataset(Dataset):
    def __init__(self, image_dir, labels=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def preprocess_image(self, image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_name)
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(processed_image)
        
        if self.transform:
            image = self.transform(image)
        
        # If labels are provided
        if self.labels:
            label = torch.tensor(self.labels[idx])
            return image, label
        return image

# CNN Model for CAPTCHA recognition
class CAPTCHANet(nn.Module):
    def __init__(self, num_chars=4, num_classes=10):
        super(CAPTCHANet, self).__init__()
        
        # CNN layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Calculate the size after CNN layers
        self.feature_size = self._get_feature_size()
        
        # Fully connected layers for each digit
        self.digit_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            ) for _ in range(num_chars)
        ])
    
    def _get_feature_size(self):
        # Helper function to calculate feature size
        x = torch.randn(1, 1, 50, 200)  # Assuming input size of 50x200
        x = self.features(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Get predictions for each digit
        digits = [classifier(x) for classifier in self.digit_classifiers]
        return digits

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss for each digit
            loss = sum(criterion(output, labels[:, i]) for i, output in enumerate(outputs))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predictions = [output.max(1)[1] for output in outputs]
            for i, pred in enumerate(predictions):
                correct += (pred == labels[:, i]).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / (total * 4)  # 4 digits
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# Prediction function
def predict_captcha(model, image_path, device):
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    dataset = CAPTCHADataset(".", transform=transform)
    processed_image = dataset.preprocess_image(image)
    image = Image.fromarray(processed_image)
    image = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predictions = [output.max(1)[1].item() for output in outputs]
    
    return ''.join(map(str, predictions))

# Example usage
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = CAPTCHANet().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    # You'll need to provide your own dataset path and labels
    dataset = CAPTCHADataset(
        image_dir="path_to_your_images",
        labels=None,  # Add your labels here
        transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save the model
    torch.save(model.state_dict(), 'captcha_model.pth')
    
    # Example prediction
    result = predict_captcha(model, "example_captcha.png", device)
    print(f"Predicted CAPTCHA: {result}")

if __name__ == "__main__":
    main()