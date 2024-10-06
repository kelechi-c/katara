import glob
import os
import time
from PIL import Image as pillow
from matplotlib import pyplot as plt
from functools import wraps
import cv2
import numpy as np


def read_image(img, img_size):
    img = np.array(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


def latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} latency => {end_time - start_time:.4f}s")
        return result

    return wrapper


def image_loader(img):
    if isinstance(img, np.ndarray):
        return pillow.fromarray(img)

    elif isinstance(img, str):
        return pillow.open(img)

    elif isinstance(img, pillow):
        return img


def image_grid(images):
    # check if image  count matches grid arrangement
    try:
        image_len = len(images["image"])
        assert image_len % 2 == 0, "Choose an even number to enable grid-show"

        f, ax = plt.subplots(2, 2)
        for index in range(image_len):
            k, v = index // 2, index % 2
            # ax[k, v].set_title(images["image"][index].filename)
            ax[k, v].imshow(images["image"][index])
            ax[k, v].axis("off")

        plt.show()

    except Exception as e:
        print(f"Error in grid display ==> {e}")


def get_all_images(root_dir, extensions=("*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp")):
    image_files = []
    for ext in extensions:
        for directory, _, _ in os.walk(root_dir):
            image_files.extend(glob.glob(os.path.join(directory, ext)))
    print(f"found {len(image_files)} images in {root_dir}")
    return image_files
