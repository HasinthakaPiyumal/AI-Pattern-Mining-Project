import io
import os
import shutil
from typing import Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

class ImageEnhancer:
    def __init__(self):
        self.color_shift = (0, 0, 0)

    def _get_average_color(self, image: Image.Image) -> Tuple[int, int, int]:
        image = image.convert("RGB")
        pixels = list(image.getdata())
        if not pixels:
            return (0, 0, 0)

        r_sum = sum(p[0] for p in pixels)
        g_sum = sum(p[1] for p in pixels)
        b_sum = sum(p[2] for p in pixels)

        num_pixels = len(pixels)
        avg_r = r_sum // num_pixels
        avg_g = g_sum // num_pixels
        avg_b = b_sum // num_pixels

        return (avg_r, avg_g, avg_b)

    def learn_transformation(self, before_example_image_path: str, after_example_image_path: str):
        try:
            before_image = Image.open(before_example_image_path)
            after_image = Image.open(after_example_image_path)

            avg_before = self._get_average_color(before_image)
            avg_after = self._get_average_color(after_image)

            self.color_shift = (
                avg_after[0] - avg_before[0],
                avg_after[1] - avg_before[1],
                avg_after[2] - avg_before[2],
            )
        except Exception as e:
            raise ValueError(f"Error learning transformation: {e}")

    def apply_transformation(self, new_image_path: str) -> Image.Image:
        try:
            new_image = Image.open(new_image_path).convert("RGB")

            def transform_pixel(pixel_value, shift):
                return max(0, min(255, pixel_value + shift))

            transformed_channels = []
            for i in range(3):
                transformed_channel = new_image.getchannel(i).point(lambda x: transform_pixel(x, self.color_shift[i]))
                transformed_channels.append(transformed_channel)

            enhanced_image = Image.merge("RGB", transformed_channels)
            return enhanced_image
        except Exception as e:
            raise ValueError(f"Error applying transformation: {e}")


app = FastAPI()
enhancer = ImageEnhancer()

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/enhance")
async def enhance_image(
    before_example_image: UploadFile = File(...),
    after_example_image: UploadFile = File(...),
    new_product_image: UploadFile = File(...),
):
    before_path = os.path.join(TEMP_UPLOAD_DIR, before_example_image.filename)
    after_path = os.path.join(TEMP_UPLOAD_DIR, after_example_image.filename)
    new_path = os.path.join(TEMP_UPLOAD_DIR, new_product_image.filename)

    try:
        with open(before_path, "wb") as buffer:
            shutil.copyfileobj(before_example_image.file, buffer)
        with open(after_path, "wb") as buffer:
            shutil.copyfileobj(after_example_image.file, buffer)
        with open(new_path, "wb") as buffer:
            shutil.copyfileobj(new_product_image.file, buffer)

        enhancer.learn_transformation(before_path, after_path)
        enhanced_image = enhancer.apply_transformation(new_path)

        img_byte_arr = io.BytesIO()
        enhanced_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")
    finally:
        for path in [before_path, after_path, new_path]:
            if os.path.exists(path):
                os.remove(path)
