from PIL import Image
from utils import IMG_WIDTH, IMG_HEIGHT


def preprocess_image(file_path, bounding_box):
    original = Image.open(fp=file_path)
    cropped = original.crop(bounding_box)
    resized = cropped.resize(size=(IMG_WIDTH, IMG_HEIGHT))
    return resized
