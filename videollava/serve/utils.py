from io import BytesIO

import requests
from PIL import Image


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

video_ext = ['.mp4', '.mov', '.mkv', '.avi']
image_ext = ['.jpg', '.png', '.bmp', '.jpeg']
