import unittest
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
import torch

from digit_recognition import DigitCNN, decode_uploaded_image, preprocess_digit_image


def make_digit(background, stroke):
    image = Image.new("RGB", (72, 72), color=background)
    draw = ImageDraw.Draw(image)
    draw.line((28, 12, 28, 58), fill=stroke, width=8)
    return image


class DigitPreprocessingTests(unittest.TestCase):
    def test_png_jpg_and_jpeg_uploads_decode(self):
        image = make_digit("white", "black")
        for image_format in ("PNG", "JPEG"):
            content = BytesIO()
            image.save(content, format=image_format)
            decoded = decode_uploaded_image(content.getvalue())
            processed, tensor = preprocess_digit_image(decoded)
            self.assertEqual(processed.size, (28, 28))
            self.assertEqual(tuple(tensor.shape), (1, 28, 28))
            self.assertEqual(decoded.size, image.size)

    def test_invalid_upload_is_rejected(self):
        with self.assertRaises(UnidentifiedImageError):
            decode_uploaded_image(b"not an image")

    def test_white_and_dark_backgrounds_produce_mnist_polarity(self):
        processed_images = []
        for background, stroke in (("white", "black"), ("black", "white")):
            processed, tensor = preprocess_digit_image(make_digit(background, stroke))
            self.assertEqual(processed.size, (28, 28))
            self.assertEqual(tuple(tensor.shape), (1, 28, 28))
            self.assertGreater(float(processed.getpixel((14, 14))), 0)
            self.assertEqual(processed.getpixel((0, 0)), 0)
            processed_images.append(np.asarray(processed, dtype=np.int16))
        np.testing.assert_array_equal(processed_images[0], processed_images[1])

    def test_blank_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No visible digit"):
            preprocess_digit_image(Image.new("RGB", (32, 32), color="white"))

    def test_classifier_returns_ten_logits(self):
        model = DigitCNN().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 1, 28, 28))
        self.assertEqual(tuple(logits.shape), (1, 10))


if __name__ == "__main__":
    unittest.main()
