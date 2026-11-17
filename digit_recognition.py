from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def decode_uploaded_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        return image.copy()


def preprocess_digit_image(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    gray = image.convert("L")
    pixels = np.asarray(gray, dtype=np.uint8)
    if pixels.size == 0 or min(gray.size) == 0:
        raise ValueError("The uploaded image is empty.")
    if int(pixels.max()) - int(pixels.min()) < 12:
        raise ValueError("No visible digit was detected. Upload an image with a clear digit.")

    border_width = max(1, round(min(gray.size) * 0.05))
    border = np.concatenate((
        pixels[:border_width, :].ravel(),
        pixels[-border_width:, :].ravel(),
        pixels[:, :border_width].ravel(),
        pixels[:, -border_width:].ravel(),
    ))
    background_is_light = float(np.median(border)) >= 127.5
    ink = 255 - pixels if background_is_light else pixels
    threshold = max(12.0, float(ink.max()) * 0.12)
    foreground = ink > threshold
    if not np.any(foreground):
        raise ValueError("No visible digit was detected. Upload an image with a clear digit.")

    rows, columns = np.nonzero(foreground)
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    cropped = np.where(foreground[top:bottom, left:right], ink[top:bottom, left:right], 0)
    cropped_image = Image.fromarray(cropped.astype(np.uint8), mode="L")

    target_size = 22
    scale = min(target_size / cropped_image.width, target_size / cropped_image.height)
    resized_size = (
        max(1, round(cropped_image.width * scale)),
        max(1, round(cropped_image.height * scale)),
    )
    resized = cropped_image.resize(resized_size, Image.Resampling.LANCZOS)
    processed = Image.new("L", (28, 28), color=0)
    resized_pixels = np.asarray(resized, dtype=np.float32)
    mass = float(resized_pixels.sum())
    row_indices, column_indices = np.indices(resized_pixels.shape)
    center_x = float((column_indices * resized_pixels).sum() / mass)
    center_y = float((row_indices * resized_pixels).sum() / mass)
    left = max(0, min(28 - resized.width, round(13.5 - center_x)))
    top = max(0, min(28 - resized.height, round(13.5 - center_y)))
    processed.paste(resized, (left, top))

    array = np.asarray(processed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy((array - MNIST_MEAN) / MNIST_STD).unsqueeze(0)
    return processed, tensor


def load_classifier_checkpoint(checkpoint_path: str | Path) -> tuple[DigitCNN, dict]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Digit classifier checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The digit classifier checkpoint has an unsupported format.")

    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metrics = {
        key: checkpoint[key]
        for key in ("epochs", "training_loss", "test_loss", "test_accuracy")
        if key in checkpoint
    }
    return model, metrics
