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

class CAPTCHADataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = list(self.image_dir.glob("*.jpg"))
        self.char_to_idx = {str(i): i for i in range(10)}
        # Add uppercase letters A-Z
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char_to_idx[char] = i + 10
        print(self.char_to_idx)

        
    def __len__(self):
        return len(self.images)
    
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return cleaned

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        image = cv2.imread(img_path)
        
        # Convert filename characters to indices
        label = self.images[idx].stem
        label_indices = [self.char_to_idx[char] for char in label]
        label = torch.tensor(label_indices, dtype=torch.long)
        
        processed_image = self.preprocess_image(image)
        image = Image.fromarray(processed_image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class GridCAPTCHANet(nn.Module):
    def __init__(self, num_chars=5, num_classes=36):  # 10 digits + 26 letters
        super(GridCAPTCHANet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        self.digit_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            ) for _ in range(num_chars)
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return [classifier(x) for classifier in self.digit_classifiers]

def predict(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    dataset = CAPTCHADataset("dummy")
    image = cv2.imread(image_path)
    processed_image = dataset.preprocess_image(image)
    image = Image.fromarray(processed_image)
    image = transform(image).unsqueeze(0).to(device)
    
    idx_to_char = {v: k for k, v in dataset.char_to_idx.items()}
    
    with torch.no_grad():
        outputs = model(image)
        predictions = [output.max(1)[1].item() for output in outputs]
        chars = [idx_to_char[idx] for idx in predictions]
    
    return ''.join(chars)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = sum(criterion(output, labels[:, i]) for i, output in enumerate(outputs))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predictions = [output.max(1)[1] for output in outputs]
            for i, pred in enumerate(predictions):
                correct += (pred == labels[:, i]).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / (total * 5)
        
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = GridCAPTCHANet(num_chars=5, num_classes=36).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    dataset = CAPTCHADataset(
        image_dir="jpg",
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    train_model(model, train_loader, criterion, optimizer, device)
    torch.save(model.state_dict(), 'grid_captcha_model.pth')

if __name__ == "__main__":
    main()