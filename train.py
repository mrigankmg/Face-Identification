import os
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "imgs")

train_x = []
labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # train_x.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
