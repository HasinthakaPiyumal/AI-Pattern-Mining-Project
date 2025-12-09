import gradio as gr

def generate_content(base_prompt, style_modifier, lighting_modifier, medium_modifier):
    """
    Generates a combined prompt based on base input and selected modifiers.
    In a real application, this would also call a generative AI model.
    """
    modifiers = []
    if style_modifier: 
        modifiers.append(f"in the style of {style_modifier}")
    if lighting_modifier:
        modifiers.append(f"{lighting_modifier} lighting")
    if medium_modifier:
        modifiers.append(f"on {medium_modifier}")
    
    final_prompt = base_prompt
    if modifiers:
        final_prompt += ", " + ", ".join(modifiers)

    # Placeholder for actual AI model call
    # In a real application, you would send 'final_prompt' to a model like DALL-E, Stable Diffusion, etc.
    generated_image_placeholder = "https://via.placeholder.com/500x300.png?text=Generated+Image+Placeholder"
    
    return final_prompt, generated_image_placeholder

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AI Creative Content Studio with Prompt Modifiers
        Generate unique images by combining a base prompt with specific modifiers.
        """
    )
    with gr.Row():
        with gr.Column():
            base_prompt_input = gr.Textbox(
                label="Base Prompt", 
                placeholder="e.g., A majestic dragon flying over a mountain"
            )
            style_dropdown = gr.Dropdown(
                ["watercolor painting", "oil painting", "digital art", "sketch", "photorealistic"],
                label="Style Modifier"
            )
            lighting_dropdown = gr.Dropdown(
                ["soft", "dramatic", "neon", "cinematic", "golden hour"],
                label="Lighting Modifier"
            )
            medium_dropdown = gr.Dropdown(
                ["canvas", "paper", "digital screen", "marble"],
                label="Medium Modifier"
            )
            generate_button = gr.Button("Generate Content")
        
        with gr.Column():
            output_prompt = gr.Textbox(label="Final Generated Prompt", interactive=False)
            output_image = gr.Image(label="Generated Content Preview")

    generate_button.click(
        fn=generate_content,
        inputs=[
            base_prompt_input,
            style_dropdown,
            lighting_dropdown,
            medium_dropdown
        ],
        outputs=[
            output_prompt,
            output_image
        ]
    )

demo.launch()