
from pathlib import Path


# Path to the data directory
data_dir = Path("./captcha_images/")


for path in data_dir.glob("*.png"): 
    filename = path.name.replace(".png","")
    if len(filename) != 5:
        path.rename(data_dir  / 'captcha_error' / path.name )
        print(path.name)
    else:
        path.rename(path.name.lower())    