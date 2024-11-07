from PIL import Image
import requests
import ssl
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt
from check_re import CaptchaPredictor
import numpy as np
import os
import time
from datetime import datetime
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        filename=f'captcha_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_save_directory():
    """Create directory for saving captchas if it doesn't exist"""
    save_dir = "downloaded_captchas"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def download_and_process_captcha(url, save_path):
    """Download CAPTCHA from URL and process it."""
    try:
        # Set up SSL context
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        
        # Download image
        with urllib.request.urlopen(url, context=ctx) as response:
            image_content = response.read()
            
        # Process image
        im = Image.open(BytesIO(image_content))
        new_image = Image.new("RGBA", im.size, "WHITE")
        new_image.paste(im, (0, 0), im)
        new_image = new_image.convert('RGB')
        
        # Save processed image
        new_image.save(save_path, "JPEG")
        return new_image, True
        
    except Exception as e:
        logging.error(f"Error downloading/processing CAPTCHA: {e}")
        return None, False

def batch_download_captchas(url, num_captchas=1000, delay=1):
    """
    Download multiple CAPTCHAs and save them with their predicted text as filename.
    
    Args:
        url: URL to download CAPTCHAs from
        num_captchas: Number of CAPTCHAs to download
        delay: Delay between downloads in seconds
    """
    # Setup
    setup_logging()
    save_dir = create_save_directory()
    predictor = CaptchaPredictor()
    
    # Statistics
    successful_downloads = 0
    failed_downloads = 0
    unique_predictions = set()
    
    print(f"Starting download of {num_captchas} CAPTCHAs...")
    start_time = time.time()
    
    for i in range(num_captchas):
        try:
            # Create temporary file path
            temp_path = os.path.join(save_dir, "temp.jpg")
            
            # Download and process image
            image, success = download_and_process_captcha(url, temp_path)
            
            if not success:
                failed_downloads += 1
                continue
            
            # Predict text
            prediction = predictor.predict(temp_path)
            unique_predictions.add(prediction)
            
            # Create final filename with prediction and timestamp
             
            final_filename = f"{prediction}.jpg"
            final_path = os.path.join(save_dir, final_filename)
            
            # Rename temp file to final filename
            os.rename(temp_path, final_path)
            
            # Log success
            successful_downloads += 1
            
            # Print progress
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_captcha = elapsed_time / (i + 1)
                remaining_time = avg_time_per_captcha * (num_captchas - i - 1)
                
                print(f"\nProgress: {i+1}/{num_captchas}")
                print(f"Success rate: {(successful_downloads/(i+1))*100:.2f}%")
                print(f"Unique predictions: {len(unique_predictions)}")
                print(f"Estimated time remaining: {remaining_time/60:.1f} minutes")
            else:
                print(".", end="", flush=True)
            
            # Log details
            logging.info(f"Successfully processed CAPTCHA {i+1}: {final_filename}")
            
            # Delay between downloads
            time.sleep(delay)
            
        except Exception as e:
            failed_downloads += 1
            logging.error(f"Error processing CAPTCHA {i+1}: {e}")
            print(f"\nError processing CAPTCHA {i+1}: {e}")
            continue
    
    # Print final statistics
    total_time = time.time() - start_time
    print("\n\nDownload Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Success rate: {(successful_downloads/num_captchas)*100:.2f}%")
    print(f"Number of unique predictions: {len(unique_predictions)}")
    print(f"Average time per CAPTCHA: {total_time/num_captchas:.2f} seconds")
    print(f"Images saved in: {save_dir}")
    
    # Log final statistics
    logging.info("\n=== Final Statistics ===")
    logging.info(f"Total time: {total_time/60:.1f} minutes")
    logging.info(f"Successful downloads: {successful_downloads}")
    logging.info(f"Failed downloads: {failed_downloads}")
    logging.info(f"Success rate: {(successful_downloads/num_captchas)*100:.2f}%")
    logging.info(f"Number of unique predictions: {len(unique_predictions)}")
    logging.info(f"Average time per CAPTCHA: {total_time/num_captchas:.2f} seconds")
    
    return successful_downloads, failed_downloads, unique_predictions

# URL of the CAPTCHA
url = 'https://tracuunnt.gdt.gov.vn/tcnnt/captcha.png'

# Start the batch download
successful, failed, unique_predictions = batch_download_captchas(url, num_captchas=1000, delay=1)