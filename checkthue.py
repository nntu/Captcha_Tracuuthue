
from PIL import Image
import requests
import ssl
import urllib.request
from io import BytesIO
url = 'https://tracuunnt.gdt.gov.vn/tcnnt/captcha.png'

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
  
    
# im = Image.open("Ba_b_do8mag_c6_big.png")
# rgb_im = im.convert('RGB')
# rgb_im.save('colors.jpg')

