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

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, img_width=130, img_height=50, char_to_num=None):
        self.image_paths = image_paths
        self.labels = labels
        self.img_width = img_width
        self.img_height = img_height
        self.char_to_num = char_to_num
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        img = Image.open(img_path)
        img = self.transform(img)
        
        # Convert label to number sequence
        label_nums = torch.tensor([self.char_to_num[c] for c in label], dtype=torch.long)
        
        return img, label_nums

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

def train_model(model, train_loader, val_loader, device, num_epochs=1000, patience=100):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Prepare CTC loss inputs
            input_lengths = torch.full(size=(outputs.size(0),), 
                                    fill_value=outputs.size(1), 
                                    dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(label) for label in labels], 
                                        dtype=torch.long).to(device)
            
            loss = ctc_loss(outputs.transpose(0, 1), labels,
                          input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                input_lengths = torch.full(size=(outputs.size(0),), 
                                        fill_value=outputs.size(1), 
                                        dtype=torch.long).to(device)
                target_lengths = torch.tensor([len(label) for label in labels], 
                                            dtype=torch.long).to(device)
                
                loss = ctc_loss(outputs.transpose(0, 1), labels,
                              input_lengths, target_lengths)
                val_loss += loss.item()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def decode_predictions(outputs, num_to_char):
    """Decode CTC outputs to text."""
    predictions = []
    outputs = outputs.detach().cpu().numpy()
    
    for output in outputs:
        # Get argmax for each timestep
        pred = np.argmax(output, axis=1)
        
        # Remove repeated characters and blanks
        prev_char = -1
        decoded = []
        for p in pred:
            if p != 0 and p != prev_char:  # 0 is CTC blank
                decoded.append(p)
            prev_char = p
        
        # Convert to text
        text = ''.join([num_to_char[d] for d in decoded])
        predictions.append(text)
    
    return predictions

def main():
    # Data preparation
    data_dir = Path("./captcha_train/")
    images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
    labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
    
    # Character mapping
    characters = sorted(list(set(char for label in labels for char in label)))
    char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}  # 0 reserved for CTC blank
    num_to_char = {idx + 1: char for idx, char in enumerate(characters)}
    print(characters)
    print(char_to_num)
    print(num_to_char)
    
    # Split data
    train_size = int(0.9 * len(images))
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    val_images = images[train_size:]
    val_labels = labels[train_size:]
    
    # Create datasets
    train_dataset = CaptchaDataset(train_images, train_labels, char_to_num=char_to_num)
    val_dataset = CaptchaDataset(val_images, val_labels, char_to_num=char_to_num)
    
    # Create dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = OCRModel(num_chars=len(characters))
    
    # Train model
    model = train_model(model, train_loader, val_loader, device)
    
    # Save character mappings
    torch.save({
        'char_to_num': char_to_num,
        'num_to_char': num_to_char
    }, 'char_mappings.pth')

if __name__ == "__main__":
    main()