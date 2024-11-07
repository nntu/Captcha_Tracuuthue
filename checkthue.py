from PIL import Image
import requests
import ssl
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt
from check_re import CaptchaPredictor
import numpy as np

def download_and_process_captcha(url):
    """Download CAPTCHA from URL and process it."""
    # Set up SSL context
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    
    # Download image
    with urllib.request.urlopen(url, context=ctx) as response:
        image_content = response.read()
        
    # Process image
    im = Image.open(BytesIO(image_content))
    new_image = Image.new("RGBA", im.size, "WHITE")  # Create white background
    new_image.paste(im, (0, 0), im)  # Paste image on background
    new_image = new_image.convert('RGB')
    
    # Save processed image
    new_image.save('test.jpg', "JPEG")
    return new_image

def show_prediction(image, prediction):
    """Display image and its prediction."""
    plt.figure(figsize=(10, 4))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show processed image
    plt.subplot(1, 2, 2)
    processed_img = Image.open('test.jpg')
    plt.imshow(processed_img)
    plt.title(f'Processed Image\nSize: {processed_img.size}\nPrediction: {prediction}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# URL of the CAPTCHA
url = 'https://tracuunnt.gdt.gov.vn/tcnnt/captcha.png'

# Initialize predictor
predictor = CaptchaPredictor()

# Download and process image
original_image = download_and_process_captcha(url)

# Make prediction
prediction = predictor.predict('test.jpg')
print(f"Predicted text: {prediction}")

# Show images and prediction
show_prediction(original_image, prediction)

# Print image information
print("\nImage Information:")
print(f"Original Image Size: {original_image.size}")
print(f"Original Image Mode: {original_image.mode}")
print(f"Processed Image Size: {Image.open('test.jpg').size}")