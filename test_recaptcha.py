
import os
from pathlib import Path
import tensorflow as tf
from check_re import CaptchaPredictor
predictor = CaptchaPredictor()

def predict( image_path):
     
    
    solved_captcha = predictor.predict(image_path)
    print(f"Predicted text: {solved_captcha}")
    return solved_captcha

# # Path to the data directory
data_dir = Path("E:/python3/Captcha_Tracuuthue/captcha/capcha_error/")

for img in data_dir.glob("*.png"):
    filenname = img.name.split(".png")[0]
   
    soveld_captcha = predict(os.path.abspath(img))
    if soveld_captcha != filenname:
       print(f"Error: {filenname} != {soveld_captcha}")
    os.rename(img, f"E:/python3/Captcha_Tracuuthue/captcha/capcha_error/{soveld_captcha}.png")   
 