import gradio as gr
import os

def parse_prompt_modifiers(prompt: str) -> dict:
    modifiers = {}
    parts = prompt.split(", ")
    for part in parts:
        if ":" in part:
            key, value = part.split(":", 1)
            modifiers[key.strip()] = value.strip()
    return modifiers

def parse_negative_prompt(negative_prompt: str) -> list:
    return [keyword.strip() for keyword in negative_prompt.split(",") if keyword.strip()]

def image_as_text_converter(image_path) -> str:
    if image_path is None:
        return ""
    return f"Visual elements from uploaded image: [description of image {os.path.basename(image_path)}]"

def paired_image_stylizer(style_image_path, target_image_path) -> str:
    if style_image_path is None or target_image_path is None:
        return ""
    return f"Applying style from image {os.path.basename(style_image_path)} to product based on image {os.path.basename(target_image_path)}."

def generate_3d_model(main_prompt: str, negative_prompt: str, reference_image, style_example_image, target_image) -> str:
    main_prompt_modifiers = parse_prompt_modifiers(main_prompt)
    negative_keywords = parse_negative_prompt(negative_prompt)
    image_text_context = image_as_text_converter(reference_image)
    style_context = paired_image_stylizer(style_example_image, target_image)

    output_description = "Generated 3D Model Description:\n"
    output_description += f"- Base Prompt: {main_prompt}\n"
    if main_prompt_modifiers:
        output_description += "- Modifiers: " + ", ".join([f"{k}: {v}" for k, v in main_prompt_modifiers.items()]) + "\n"
    if negative_keywords:
        output_description += "- Excluded: " + ", ".join(negative_keywords) + "\n"
    if image_text_context:
        output_description += f"- Image-as-Text Context: {image_text_context}\n"
    if style_context:
        output_description += f"- Paired Image Style Context: {style_context}\n"

    # Mock 3D generation logic
    if "material" in main_prompt_modifiers:
        output_description += f"- Material: {main_prompt_modifiers['material']}\n"
    if "color" in main_prompt_modifiers:
        output_description += f"- Color: {main_prompt_modifiers['color']}\n"
    if "style" in main_prompt_modifiers:
        output_description += f"- Design Style: {main_prompt_modifiers['style']}\n"

    if "rough" in negative_keywords:
        output_description += f"- Ensures smooth surfaces.\n"
    if "old" in negative_keywords:
        output_description += f"- Ensures modern aesthetics.\n"

    output_description += "\n(This is a conceptual output. In a real application, a 3D model file would be generated and displayed.)"
    return output_description

with gr.Blocks() as demo:
    gr.Markdown("## E-commerce 3D Product Customizer with Advanced Prompting")

    with gr.Row():
        with gr.Column():
            main_prompt_input = gr.Textbox(label="Main Product Description Prompt (e.g., 'A chair, material: wood, color: blue, style: minimalist')", lines=3)
            negative_prompt_input = gr.Textbox(label="Negative Prompt (what to exclude, e.g., 'rough, old, metallic')", lines=2)
            reference_image_input = gr.Image(label="Upload Reference Image (for ImageasText Prompting)", type="filepath")
            style_example_image_input = gr.Image(label="Upload Style Example Image (for PairedImage Prompting)", type="filepath")
            target_image_input = gr.Image(label="Upload Target Image (for PairedImage Prompting)", type="filepath")
            generate_button = gr.Button("Generate 3D Model")
        with gr.Column():
            output_text = gr.Textbox(label="Generated 3D Model Description", lines=15)

    generate_button.click(
        fn=generate_3d_model,
        inputs=[main_prompt_input, negative_prompt_input, reference_image_input, style_example_image_input, target_image_input],
        outputs=output_text
    )

demo.launch()