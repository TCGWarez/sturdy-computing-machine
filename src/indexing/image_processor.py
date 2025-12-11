"""
src/indexing/image_processor.py: Image preprocessing for indexing
Normalizes Scryfall images to canonical size
"""

from PIL import Image
import numpy as np
import cv2
from pathlib import Path

from src.config import CANONICAL_IMAGE_SIZE


def normalize_image(image_path: Path, target_size: int = CANONICAL_IMAGE_SIZE) -> Image.Image:
    """Normalize image to canonical size while preserving aspect ratio."""
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    x_offset = (target_size - img.width) // 2
    y_offset = (target_size - img.height) // 2
    new_img.paste(img, (x_offset, y_offset))

    return new_img


def preprocess_for_embedding(image: Image.Image, input_size: int = 224) -> np.ndarray:
    """Preprocess image for CNN embedding (normalize, resize to CHW format)."""
    img_resized = image.resize((input_size, input_size), Image.Resampling.LANCZOS)

    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_array = img_array.astype(np.float32)
    img_array = np.transpose(img_array, (2, 0, 1))

    return img_array


def normalize_image_unified(image_path: Path, target_size: int = CANONICAL_IMAGE_SIZE) -> Image.Image:
    """Unified image normalization with CLAHE contrast enhancement."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    img = cv2.merge([l_clahe, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    pil_img = Image.fromarray(img)

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    pil_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    x_offset = (target_size - pil_img.width) // 2
    y_offset = (target_size - pil_img.height) // 2
    new_img.paste(pil_img, (x_offset, y_offset))

    return new_img

