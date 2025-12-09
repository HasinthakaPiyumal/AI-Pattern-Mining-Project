import gradio as gr
from PIL import Image
import io
import base64
import requests
import threading
import time
from fastapi import FastAPI, UploadFile, File, Form
from uvicorn import Server, Config

PRODUCT_DATA = [
    {"id": "shirt_001", "name": "Blue T-Shirt", "image_path": "product_shirt.png"},
    {"id": "hat_001", "name": "Red Baseball Cap", "image_path": "product_hat.png"},
]

def create_dummy_product_images():
    try:
        Image.new("RGB", (200, 200), color = "blue").save("product_shirt.png")
        Image.new("RGB", (150, 100), color = "red").save("product_hat.png")
    except Exception as e:
        pass

def simulate_ai_try_on(user_image: Image.Image, product_image: Image.Image) -> Image.Image:
    user_width, user_height = user_image.size
    product_image_rgba = product_image.convert("RGBA") if product_image.mode != "RGBA" else product_image

    target_product_width = int(user_width * 0.4)
    if product_image_rgba.width > 0:
        scale_factor = target_product_width / product_image_rgba.width
        product_image_rgba = product_image_rgba.resize((target_product_width, int(product_image_rgba.height * scale_factor)), Image.Resampling.LANCZOS)
    
    final_image = user_image.copy()
    
    paste_x = (user_width - product_image_rgba.width) // 2
    paste_y = int(user_height * 0.25)
    
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)

    mask = product_image_rgba.split()[3] if "A" in product_image_rgba.getbands() else None

    if mask:
        final_image.paste(product_image_rgba, (paste_x, paste_y), mask)
    else:
        final_image.paste(product_image_rgba, (paste_x, paste_y))

    return final_image

def get_product_image_by_id(product_id: str) -> Image.Image:
    for product in PRODUCT_DATA:
        if product["id"] == product_id:
            try:
                return Image.open(product["image_path"]).convert("RGB")
            except FileNotFoundError:
                return Image.new("RGB", (200, 200), color = "gray")
    raise ValueError(f"Product with ID {product_id} not found.")

app = FastAPI()

@app.post("/try_on")
async def try_on_api(user_image_file: UploadFile = File(...), product_id: str = Form(...)):
    user_image_bytes = await user_image_file.read()
    user_image = Image.open(io.BytesIO(user_image_bytes)).convert("RGB")
    
    user_image = user_image.resize((512, 512))
    
    try:
        product_image = get_product_image_by_id(product_id)
    except ValueError as e:
        return {"error": str(e)}

    try_on_result = simulate_ai_try_on(user_image, product_image)

    buffered = io.BytesIO()
    try_on_result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"result_image": img_str}

FASTAPI_SERVER_URL = "http://127.0.0.1:8000"

def virtual_try_on_frontend(user_image: Image.Image, selected_product_id: str) -> Image.Image:
    if user_image is None or selected_product_id is None:
        return Image.new("RGB", (500, 500), color = "white")
    
    buffered = io.BytesIO()
    user_image.save(buffered, format="PNG")
    
    files = {"user_image_file": ("user_image.png", buffered.getvalue(), "image/png")}
    data = {"product_id": selected_product_id}
    
    try:
        response = requests.post(f"{FASTAPI_SERVER_URL}/try_on", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            return Image.new("RGB", (500, 500), color = "red")
        
        img_str = result["result_image"]
        decoded_image = base64.b64decode(img_str)
        return Image.open(io.BytesIO(decoded_image)).convert("RGB")
    except requests.exceptions.ConnectionError:
        return Image.new("RGB", (500, 500), color = "yellow")
    except Exception as e:
        return Image.new("RGB", (500, 500), color = "orange")

product_choices = [(p["name"], p["id"]) for p in PRODUCT_DATA]

iface = gr.Interface(
    fn=virtual_try_on_frontend,
    inputs=[
        gr.Image(type="pil", label="Upload Your Photo"),
        gr.Dropdown(choices=product_choices, label="Select Product", value=product_choices[0][1] if product_choices else None)
    ],
    outputs=gr.Image(type="pil", label="Virtual Try-On Result"),
    title="E-commerce Virtual Try-On (PairedImage Prompting Demo)",
    description="Upload your photo and select a product to see how it looks on you virtually. This demo simulates the 'PairedImage Prompting' concept by overlaying the product."
)

class UvicornServer(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=8000):
        super().__init__()
        self.config = Config(app=app, host=host, port=port, log_level="info")
        self.server = Server(config=self.config)

    def run(self):
        self.server.run()

if __name__ == "__main__":
    create_dummy_product_images()
    
    fastapi_thread = UvicornServer(app)
    fastapi_thread.start()
    
    time.sleep(2) 
    
    iface.launch(inbrowser=True)