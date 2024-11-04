import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CAPTCHALabelProcessor:
    def __init__(self, num_digits=4):
        self.num_digits = num_digits
        self.char_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                           '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
    def encode_label(self, label_str):
        """
        Convert a string label (e.g., '2063') to tensor format
        Returns: torch tensor of shape (num_digits,)
        """
        if len(label_str) != self.num_digits:
            raise ValueError(f"Label must be {self.num_digits} digits long")
            
        label = torch.zeros(self.num_digits, dtype=torch.long)
        for i, char in enumerate(label_str):
            label[i] = self.char_to_idx[char]
        return label
    
    def decode_label(self, label_tensor):
        """
        Convert a tensor label back to string format
        """
        return ''.join(self.idx_to_char[idx.item()] for idx in label_tensor)

class CAPTCHADatasetWithLabels(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Dataset class that extracts labels from filenames
        Assumes filenames are in format: 'captcha_2063.png'
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_processor = CAPTCHALabelProcessor()
        
    def __len__(self):
        return len(self.images)
    
    def extract_label_from_filename(self, filename):
        """
        Extract label from filename (e.g., 'captcha_2063.png' -> '2063')
        Modify this based on your filename format
        """
        # Assuming filename format is 'captcha_2063.png'
        return filename.split('.')[0]
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        # Extract and encode label
        label_str = self.extract_label_from_filename(img_name)
        label = self.label_processor.encode_label(label_str)
        
        return image, label

def create_label_file(image_dir, output_file='labels.txt'):
    """
    Create a label file from image filenames
    """
    with open(output_file, 'w') as f:
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                label = filename.split('.')[0]
                f.write(f"{filename}\t{label}\n")

def verify_labels(image_dir, label_file='labels.txt'):
    """
    Verify that all images have corresponding labels and vice versa
    """
    errors = []
    
    # Read label file
    labels_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split('\t')
            labels_dict[filename] = label
    
    # Check all images have labels
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            if filename not in labels_dict:
                errors.append(f"Missing label for {filename}")
            else:
                # Verify label format
                label = labels_dict[filename]
                if not label.isdigit() or len(label) != 5:
                    errors.append(f"Invalid label format for {filename}: {label}")
    
    # Check all labels have images
    for filename in labels_dict:
        if not os.path.exists(os.path.join(image_dir, filename)):
            errors.append(f"Missing image for label: {filename}")
    
    return errors

# Example usage
def main():
    # Example of creating and using the dataset
    image_dir = "captcha_images"
    
    # Create label file
    create_label_file(image_dir)
    
    # Verify labels
    errors = verify_labels(image_dir)
    if errors:
        print("Found errors:")
        for error in errors:
            print(error)
    else:
        print("All labels verified successfully!")
    
    # Create dataset
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
    ])
    
    dataset = CAPTCHADatasetWithLabels(
        image_dir=image_dir,
        transform=transform
    )
    
    # Test the dataset
    image, label = dataset[0]
    label_processor = CAPTCHALabelProcessor()
    print(f"Image shape: {image.shape}")
    print(f"Label tensor: {label}")
    print(f"Decoded label: {label_processor.decode_label(label)}")
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example of iterating through the data
    for batch_images, batch_labels in train_loader:
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break

if __name__ == "__main__":
    main()