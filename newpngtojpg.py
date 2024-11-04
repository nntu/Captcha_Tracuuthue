from PIL import Image
import os

def convert_png_to_jpg(input_path, output_path=None, bg_color=(255, 255, 255)):
    """
    Convert PNG image with transparency to JPG with proper handling of transparency mask.
    
    Args:
        input_path (str): Path to input PNG file
        output_path (str, optional): Path for output JPG file. If None, creates in same directory
        bg_color (tuple): RGB background color to replace transparency (default: white)
    
    Returns:
        str: Path to the saved JPG file
    """
    # Open the PNG image
    img = Image.open(input_path)
    
    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create a new image with white background
    background = Image.new('RGB', img.size, bg_color)
    
    # Paste the image on the background using alpha channel as mask
    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    
    # Generate output path if not provided
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.jpg'
    
    # Save as JPG
    background.save(output_path, 'JPEG', quality=95)
    return output_path

def batch_convert_directory(input_dir, output_dir=None, bg_color=(255, 255, 255)):
    """
    Convert all PNG files in a directory to JPG.
    
    Args:
        input_dir (str): Input directory containing PNG files
        output_dir (str, optional): Output directory for JPG files
        bg_color (tuple): RGB background color to replace transparency
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, 
                                     os.path.splitext(filename)[0] + '.jpg')
            try:
                convert_png_to_jpg(input_path, output_path, bg_color)
                print(f"Converted: {filename}")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
   
    
    # Or convert all PNGs in a directory
    batch_convert_directory("captcha\capcha_ok", "captcha\jpg")