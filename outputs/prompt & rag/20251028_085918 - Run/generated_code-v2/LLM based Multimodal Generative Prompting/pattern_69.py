import gradio as gr
import os
import time

class ModelStorage:
    def __init__(self, storage_dir="3d_models"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_model(self, filename, model_data):
        if not filename.endswith((".obj", ".fbx", ".gltf")):
            filename += ".obj" # Default extension
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "w") as f:
            f.write(model_data)
        return f"Model saved as {filename}"

    def load_model(self, filename):
        filepath = os.path.join(self.storage_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                model_data = f.read()
            return model_data
        return "Error: Model not found."

    def list_models(self):
        return [f for f in os.listdir(self.storage_dir) if os.path.isfile(os.path.join(self.storage_dir, f))]

class InputProcessor:
    def process_text_prompt(self, prompt):
        return {"type": "text", "content": prompt}

    def process_image_prompt(self, image_path):
        if image_path is None:
            return {"type": "image", "content": None}
        return {"type": "image", "content": f"Image data from {os.path.basename(image_path)}"}

    def process_sketch_input(self, sketch_data):
        if sketch_data is None:
            return {"type": "sketch", "content": None}
        return {"type": "sketch", "content": "Processed sketch data"}

    def process_bounding_box(self, model_id, coords, prompt):
        if not coords or not prompt:
            return None
        return {"type": "bbox", "model_id": model_id, "coords": coords, "prompt": prompt}

class _3DGenerationEngine:
    def generate_from_text(self, processed_input):
        if processed_input["content"]:
            time.sleep(1) # Simulate generation time
            return f"3D Model: {{processed_input['content']}} generated (simplified representation)"
        return "No text prompt provided for generation."

    def generate_from_image(self, processed_input):
        if processed_input["content"]:
            time.sleep(2)
            return f"3D Model: generated from {{processed_input['content']}} (simplified representation)"
        return "No image prompt provided for generation."

    def generate_from_sketch(self, processed_input):
        if processed_input["content"]:
            time.sleep(2)
            return f"3D Model: generated from {{processed_input['content']}} (simplified representation)"
        return "No sketch provided for generation."

    def modify_model(self, current_model, processed_modification_prompt):
        if current_model and processed_modification_prompt:
            time.sleep(1)
            return f"Modified 3D Model: {current_model} with {processed_modification_prompt['content']} (simplified representation)"
        return current_model or "No model or modification prompt."

class TextureManagementModule:
    def apply_texture(self, current_model, processed_texture_prompt):
        if current_model and processed_texture_prompt and processed_texture_prompt["content"]:
            time.sleep(1)
            return f"Textured 3D Model: {current_model} with {{processed_texture_prompt['content']}} (simplified representation)"
        return current_model or "No model or texture prompt."

    def modify_region_texture(self, current_model, processed_bbox_data):
        if current_model and processed_bbox_data:
            time.sleep(1)
            return f"Region-textured 3D Model: {current_model} - region {processed_bbox_data['coords']} now has {{processed_bbox_data['prompt']}} texture (simplified representation)"
        return current_model or "No model or bounding box for texture modification."

class GameAssetGenerator:
    def __init__(self):
        self.input_processor = InputProcessor()
        self.gen_engine = _3DGenerationEngine()
        self.texture_module = TextureManagementModule()
        self.model_storage = ModelStorage()
        self.current_model = None

    def generate_asset(self, text_prompt, image_input, sketch_input):
        generated_model_text = ""
        if text_prompt:
            processed_text = self.input_processor.process_text_prompt(text_prompt)
            generated_model_text = self.gen_engine.generate_from_text(processed_text)
            self.current_model = generated_model_text
        elif image_input:
            processed_image = self.input_processor.process_image_prompt(image_input.name)
            generated_model_text = self.gen_engine.generate_from_image(processed_image)
            self.current_model = generated_model_text
        elif sketch_input:
            processed_sketch = self.input_processor.process_sketch_input(sketch_input.name)
            generated_model_text = self.gen_engine.generate_from_sketch(processed_sketch)
            self.current_model = generated_model_text
        else:
            return "Please provide a text, image, or sketch prompt to generate a model."
        return generated_model_text, self.model_storage.list_models()

    def modify_asset(self, modification_text_prompt, current_model_display):
        if not current_model_display:
            return "No model to modify. Generate or load one first.", self.model_storage.list_models()
        if not modification_text_prompt:
            return current_model_display, self.model_storage.list_models()

        processed_mod = self.input_processor.process_text_prompt(modification_text_prompt)
        modified_model = self.gen_engine.modify_model(current_model_display, processed_mod)
        self.current_model = modified_model
        return modified_model, self.model_storage.list_models()

    def apply_texture(self, texture_text_prompt, current_model_display):
        if not current_model_display:
            return "No model to texture. Generate or load one first.", self.model_storage.list_models()
        if not texture_text_prompt:
            return current_model_display, self.model_storage.list_models()

        processed_texture = self.input_processor.process_text_prompt(texture_text_prompt)
        textured_model = self.texture_module.apply_texture(current_model_display, processed_texture)
        self.current_model = textured_model
        return textured_model, self.model_storage.list_models()

    def modify_region_texture(self, region_coords, region_texture_prompt, current_model_display):
        if not current_model_display:
            return "No model to modify texture. Generate or load one first.", self.model_storage.list_models()
        if not region_coords or not region_texture_prompt:
            return current_model_display, self.model_storage.list_models()

        # In a real app, model_id would be crucial here
        processed_bbox = self.input_processor.process_bounding_box(
            model_id="current_model_placeholder", coords=region_coords, prompt=region_texture_prompt
        )
        modified_model = self.texture_module.modify_region_texture(current_model_display, processed_bbox)
        self.current_model = modified_model
        return modified_model, self.model_storage.list_models()

    def save_current_model(self, filename, current_model_display):
        if not current_model_display:
            return "No model to save.", self.model_storage.list_models()
        if not filename:
            filename = f"generated_model_{int(time.time())}"
        message = self.model_storage.save_model(filename, current_model_display)
        return message, self.model_storage.list_models()

    def load_selected_model(self, selected_filename):
        if not selected_filename:
            return "Please select a model to load.", self.model_storage.list_models()
        model_data = self.model_storage.load_model(selected_filename)
        if not model_data.startswith("Error"):
            self.current_model = model_data
        return model_data, self.model_storage.list_models()

# Initialize the application
app = GameAssetGenerator()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 3D Prompting Game Asset Generator")

    with gr.Tab("Generate 3D Asset"):
        with gr.Row():
            text_prompt_gen = gr.Textbox(label="Text Prompt (e.g., 'medieval knight with glowing blue armor')")
            image_input_gen = gr.File(label="Upload Image (Concept Art)")
            sketch_input_gen = gr.File(label="Upload Sketch")
        generate_btn = gr.Button("Generate Model")
        generated_model_output = gr.Textbox(label="Generated 3D Model (Simplified Representation)", interactive=False)

    with gr.Tab("Modify 3D Asset"):
        gr.Markdown("Current Model for Modification will be displayed below after generation/loading.")
        modification_prompt = gr.Textbox(label="Modification Prompt (e.g., 'make the armor more ornate')")
        # In a real application, bounding box selection would be on a 3D viewer
        modify_btn = gr.Button("Modify Model")
        modified_model_output = gr.Textbox(label="Modified 3D Model (Simplified Representation)", interactive=False)

    with gr.Tab("Apply/Modify Texture"):
        gr.Markdown("Current Model for Texturing will be displayed below after generation/loading.")
        texture_prompt = gr.Textbox(label="Overall Texture Prompt (e.g., 'generate rusty metal texture')")
        apply_texture_btn = gr.Button("Apply Overall Texture")
        region_coords_input = gr.Textbox(label="Region Bounding Box Coords (e.g., 'x1,y1,z1,x2,y2,z2')")
        region_texture_prompt = gr.Textbox(label="Region-specific Texture Prompt (e.g., 'gold trim')")
        modify_region_texture_btn = gr.Button("Modify Region Texture")
        textured_model_output = gr.Textbox(label="Textured 3D Model (Simplified Representation)", interactive=False)

    with gr.Tab("Save/Load Asset"):
        filename_input = gr.Textbox(label="Filename (e.g., knight_v1.obj)", value="")
        save_btn = gr.Button("Save Current Model")
        model_save_status = gr.Textbox(label="Save Status", interactive=False)

        available_models_dropdown = gr.Dropdown(label="Load Existing Model", choices=app.model_storage.list_models())
        load_btn = gr.Button("Load Selected Model")
        loaded_model_output = gr.Textbox(label="Loaded 3D Model (Simplified Representation)", interactive=False)

    gr.Markdown("## Current Working Model and Stored Models")
    current_working_model = gr.Textbox(label="Current Active Model", interactive=False)
    stored_models_list = gr.Dataframe(label="Stored Models", headers=["Filename"], datatype=["str"], interactive=False)

    # Event Handlers
    def update_outputs(model_str, model_list):
        return model_str, [[f] for f in model_list], gr.Dropdown(choices=model_list)

    generate_btn.click(
        app.generate_asset,
        inputs=[text_prompt_gen, image_input_gen, sketch_input_gen],
        outputs=[generated_model_output, stored_models_list, available_models_dropdown]
    ).then(
        update_outputs,
        inputs=[generated_model_output, stored_models_list],
        outputs=[current_working_model, stored_models_list, available_models_dropdown]
    )

    modify_btn.click(
        app.modify_asset,
        inputs=[modification_prompt, current_working_model],
        outputs=[modified_model_output, stored_models_list, available_models_dropdown]
    ).then(
        update_outputs,
        inputs=[modified_model_output, stored_models_list],
        outputs=[current_working_model, stored_models_list, available_models_dropdown]
    )

    apply_texture_btn.click(
        app.apply_texture,
        inputs=[texture_prompt, current_working_model],
        outputs=[textured_model_output, stored_models_list, available_models_dropdown]
    ).then(
        update_outputs,
        inputs=[textured_model_output, stored_models_list],
        outputs=[current_working_model, stored_models_list, available_models_dropdown]
    )

    modify_region_texture_btn.click(
        app.modify_region_texture,
        inputs=[region_coords_input, region_texture_prompt, current_working_model],
        outputs=[textured_model_output, stored_models_list, available_models_dropdown]
    ).then(
        update_outputs,
        inputs=[textured_model_output, stored_models_list],
        outputs=[current_working_model, stored_models_list, available_models_dropdown]
    )

    save_btn.click(
        app.save_current_model,
        inputs=[filename_input, current_working_model],
        outputs=[model_save_status, stored_models_list, available_models_dropdown]
    ).then(
        update_outputs,
        inputs=[current_working_model, stored_models_list],
        outputs=[current_working_model, stored_models_list, available_models_dropdown]
    )

    load_btn.click(
        app.load_selected_model,
        inputs=[available_models_dropdown],
        outputs=[loaded_model_output, stored_models_list, available_models_dropdown]
    ).then(
        update_outputs,
        inputs=[loaded_model_output, stored_models_list],
        outputs=[current_working_model, stored_models_list, available_models_dropdown]
    )

    # Initial update for stored models on startup
    demo.load(
        lambda: ([[f] for f in app.model_storage.list_models()], app.model_storage.list_models()),
        outputs=[stored_models_list, available_models_dropdown]
    )

demo.launch()