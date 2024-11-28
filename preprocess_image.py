from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def convert_transparent_to_white(image_path, save_path=None):
    """
    Convert transparent background in PNG to white
    
    Parameters:
    image_path (str): Path to input PNG image
    save_path (str): Optional path to save the processed image
    
    Returns:
    PIL.Image: Image with white background
    """
    # Open image and convert to RGBA if it isn't already
    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Convert to numpy array for processing
    image_array = np.array(image)
    
    # Create a white background
    white_background = np.ones_like(image_array) * 255
    
    # Extract alpha channel
    alpha = image_array[:, :, 3]
    
    # Calculate alpha factor for blending
    alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
    
    # Blend original image with white background based on alpha
    result_array = image_array[:, :, :3] * alpha_factor + white_background[:, :, :3] * (1 - alpha_factor)
    
    # Convert back to uint8
    result_array = result_array.astype(np.uint8)
    
    # Create new image from array
    result_image = Image.fromarray(result_array, 'RGB')
    
    if save_path:
        result_image.save(save_path)
    
    return result_image

def show_comparison(original_path, processed_image):
    """
    Display the original and processed images side by side
    """
    # Open original image
    original = Image.open(original_path)
    
    # Create checkered background to show transparency
    def create_checkerboard(size, square_size=10):
        rows, cols = size[1] // square_size + 1, size[0] // square_size + 1
        board = np.indices((rows, cols)).sum(axis=0) % 2
        board = np.kron(board, np.ones((square_size, square_size)))
        board = board[:size[1], :size[0]]
        return (board * 40 + 215).astype(np.uint8)
    
    # Create figure with subplots
    plt.figure(figsize=(12, 6))
    
    # Show original with checkered background to highlight transparency
    plt.subplot(1, 2, 1)
    checkerboard = create_checkerboard(original.size)
    plt.imshow(checkerboard, cmap='gray')
    if original.mode == 'RGBA':
        plt.imshow(original, alpha=original.getchannel('A'))
    else:
        plt.imshow(original)
    plt.title('Original Image\n(checkered background shows transparency)')
    plt.axis('off')
    
    # Show processed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image)
    plt.title('White Background')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_image(image_path, save_path=None):
    """
    Process the image and show comparison
    
    Parameters:
    image_path (str): Path to input PNG image
    save_path (str): Optional path to save processed image
    
    Returns:
    PIL.Image: Processed image
    """
    # Convert transparent to white
    result = convert_transparent_to_white(image_path, save_path)
    
    # Show comparison
    show_comparison(image_path, result)
    
    return result

# Function to handle batch processing
def batch_process_images(input_paths, output_dir):
    """
    Process multiple PNG images
    
    Parameters:
    input_paths (list): List of input image paths
    output_dir (str): Directory to save processed images
    """
    import os
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for input_path in input_paths:
        # Generate output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f'white_bg_{filename}')
        
        # Process image
        try:
            process_image(input_path, output_path)
            print(f'Successfully processed: {filename}')
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')

process_image("2a3ad.jpg")