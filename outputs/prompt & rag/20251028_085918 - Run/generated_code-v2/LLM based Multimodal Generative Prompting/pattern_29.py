import base64
from io import BytesIO
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import gradio as gr
import requests

# --- FastAPI Backend ---

app = FastAPI()

class GenerateDesignRequest(BaseModel):
    text_prompt: Optional[str] = None
    image_prompt_base64: Optional[str] = None
    bbox_prompts: Optional[List[str]] = None

class GenerateDesignResponse(BaseModel):
    status: str
    message: str
    generated_3d_asset_path: Optional[str] = None
    rendered_image_base64: Optional[str] = None

def simulate_3d_generation(text_prompt: Optional[str], image_prompt_base64: Optional[str], bbox_prompts: Optional[List[str]]):
    """Simulates the 3D generation process and returns a dummy image and asset path."""
    print(f"Simulating 3D generation with:\n  Text: {text_prompt}\n  Image (present): {image_prompt_base64 is not None}\n  BBoxes: {bbox_prompts}")

    # Create a dummy image for rendering
    try:
        # If an image prompt is provided, try to use it as a base or just generate a simple one
        if image_prompt_base64:
            img_data = base64.b64decode(image_prompt_base64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            # For simplicity, let's just resize it if it's too big, or generate a placeholder
            if max(img.size) > 512:
                img.thumbnail((512, 512))
            rendered_img = img
        else:
            # Generate a simple placeholder image
            rendered_img = Image.new('RGB', (512, 384), color = 'lightgray')
            d = ImageDraw.Draw(rendered_img)
            d.text((10,10), f"Generated 3D Scene based on: {text_prompt or 'Prompts'}", fill=(0,0,0))
            if bbox_prompts:
                d.text((10,30), f"BBoxes: {', '.join(bbox_prompts)}", fill=(0,0,0))

    except Exception as e:
        print(f"Error generating dummy image: {e}")
        rendered_img = Image.new('RGB', (512, 384), color='red')
        d = ImageDraw.Draw(rendered_img)
        d.text((10, 10), "Error in image generation", fill=(0,0,0))

    buffered = BytesIO()
    rendered_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    dummy_asset_path = "/path/to/generated_3d_model.obj"
    return dummy_asset_path, img_str

@app.post("/generate_design", response_model=GenerateDesignResponse)
async def generate_design(request: GenerateDesignRequest):
    try:
        generated_3d_asset_path, rendered_image_base64 = simulate_3d_generation(
            request.text_prompt,
            request.image_prompt_base64,
            request.bbox_prompts
        )
        return GenerateDesignResponse(
            status="success",
            message="3D interior design generated successfully!",
            generated_3d_asset_path=generated_3d_asset_path,
            rendered_image_base64=rendered_image_base64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Gradio Frontend ---

FASTAPI_URL = "http://127.0.0.1:8000"

def gradio_interface(text_prompt: str, image_prompt: Image.Image, bbox_prompts_str: str):
    image_prompt_base64 = None
    if image_prompt:
        buffered = BytesIO()
        image_prompt.save(buffered, format="PNG")
        image_prompt_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    bbox_prompts = [p.strip() for p in bbox_prompts_str.split(',') if p.strip()] if bbox_prompts_str else None

    payload = {
        "text_prompt": text_prompt if text_prompt else None,
        "image_prompt_base64": image_prompt_base64,
        "bbox_prompts": bbox_prompts
    }

    try:
        response = requests.post(f"{FASTAPI_URL}/generate_design", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if data["status"] == "success":
            rendered_img_base64 = data.get("rendered_image_base64")
            if rendered_img_base64:
                img_data = base64.b64decode(rendered_img_base64)
                img = Image.open(BytesIO(img_data))
                return img, data["message"]
            else:
                return None, "No rendered image returned, but design generated: " + data["message"]
        else:
            return None, f"Error: {data['message']}"
    except requests.exceptions.ConnectionError:
        return None, f"Connection Error: Could not connect to the backend server at {FASTAPI_URL}. Please ensure the backend is running."
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Virtual Interior Designer (3D Prompting Demo)")
    gr.Markdown("Enter your design prompts below to generate a 3D interior layout.")

    with gr.Row():
        text_input = gr.Textbox(label="Text Prompt", placeholder="e.g., A modern living room with a green sofa and a wooden coffee table")
        image_input = gr.Image(type="pil", label="Image Prompt (e.g., style reference)")

    bbox_input = gr.Textbox(label="Bounding Box Prompts (comma-separated)", placeholder="e.g., sofa: 10,20,0,100,50,80, table: 120,30,0,150,60,50")

    generate_button = gr.Button("Generate 3D Design")

    with gr.Column():
        output_image = gr.Image(label="Generated 3D Scene Render")
        output_message = gr.Textbox(label="Status/Message", interactive=False)

    generate_button.click(
        gradio_interface,
        inputs=[text_input, image_input, bbox_input],
        outputs=[output_image, output_message]
    )

# To run both, you would typically run the FastAPI app separately and then the Gradio app.
# For a single file, we'll provide instructions on how to run them.
# You can uncomment the lines below for a simple local execution if you manually manage processes.

# if __name__ == "__main__":
#     import threading
#     import time
#
#     # Function to run FastAPI
#     def run_fastapi():
#         uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
#
#     # Start FastAPI in a separate thread
#     fastapi_thread = threading.Thread(target=run_fastapi)
#     fastapi_thread.daemon = True # Allow main program to exit even if thread is still running
#     fastapi_thread.start()
#
#     # Give FastAPI a moment to start up
#     time.sleep(2)
#
#     # Launch Gradio interface
#     demo.launch(server_name="0.0.0.0", server_port=7860)
#
#     # Keep the main thread alive if FastAPI is not daemonized
#     # If you remove daemon=True from the thread, you might need something like:
#     # fastapi_thread.join() # This would block, so not ideal for concurrent Gradio


# --- Instructions to Run ---
# To run this application:
# 1. Save the code as `virtual_interior_designer.py`.
# 2. Install necessary libraries:
#    `pip install fastapi uvicorn pydantic pillow gradio requests`
# 3. Open two terminal windows.
# 4. In the first terminal, run the FastAPI backend:
#    `uvicorn virtual_interior_designer:app --host 0.0.0.0 --port 8000`
# 5. In the second terminal, run the Gradio frontend:
#    `python -c "from virtual_interior_designer import demo; demo.launch(server_name='0.0.0.0', server_port=7860)"`
#    (Alternatively, if you wrap the gradio.launch in `if __name__ == "__main__":` block and run `python virtual_interior_designer.py`)
# 6. Access the Gradio interface in your browser at `http://0.0.0.0:7860` (or the address shown in your terminal).
