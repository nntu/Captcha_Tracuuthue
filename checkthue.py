
from PIL import Image
import requests
import ssl
import urllib.request
from io import BytesIO

from check_re import CaptchaPredictor
url = 'https://tracuunnt.gdt.gov.vn/tcnnt/captcha.png'


predictor = CaptchaPredictor()

def predict( image_path):
     
    
    solved_captcha = predictor.predict(image_path)
    print(f"Predicted text: {solved_captcha}")
    return solved_captcha


# Set up SSL context to allow legacy TLS versions
ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT

# Use urllib to open the URL and read the content
with urllib.request.urlopen(url, context=ctx) as response:    
    image_content = response.read()
    print(image_content)
        # Store the raw content for saving later
         
    im = Image.open(BytesIO(image_content))
    new_image = Image.new("RGBA", im.size, "WHITE") # Create a white rgba background
    new_image.paste(im, (0, 0), im)              # Paste the image on the background. Go to the links given below for details.
    new_image.convert('RGB').save('test.jpg', "JPEG")  # Save as JPEG
    
    
 