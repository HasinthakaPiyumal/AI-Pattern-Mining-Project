import gradio as gr

def decompose_experiment(experiment_description: str) -> list:
    """
    Simulates the decomposition of an experiment description into a sequence of visualizable steps.
    In a real application, an LLM would perform this.
    """
    steps = []
    lower_description = experiment_description.lower()

    if "photosynthesis" in lower_description:
        steps = [
            {"description": "Plants absorb sunlight through chlorophyll.", "image_prompt": "Sunlight absorbed by plant leaf"},
            {"description": "Water is absorbed from the soil by roots.", "image_prompt": "Plant roots absorbing water"},
            {"description": "Carbon dioxide enters the leaves through stomata.", "image_prompt": "CO2 entering leaf stomata"},
            {"description": "Inside chloroplasts, light energy converts water and CO2 into glucose and oxygen.", "image_prompt": "Chloroplast converting light, water, CO2 to glucose, oxygen"},
            {"description": "Glucose is used as energy for the plant, and oxygen is released.", "image_prompt": "Plant releasing oxygen, using glucose"}
        ]
    elif "water cycle" in lower_description or "hydrologic cycle" in lower_description:
        steps = [
            {"description": "Evaporation: Water turns into vapor and rises into the atmosphere.", "image_prompt": "Water evaporating into clouds"},
            {"description": "Condensation: Water vapor cools and forms clouds.", "image_prompt": "Clouds forming from water vapor"},
            {"description": "Precipitation: Water falls back to Earth as rain, snow, or hail.", "image_prompt": "Rain falling from clouds"},
            {"description": "Collection: Water gathers in rivers, lakes, oceans, or seeps into the ground.", "image_prompt": "River flowing into ocean, ground absorption"}
        ]
    elif "chemical reaction" in lower_description:
        steps = [
            {"description": "Initial reactants are mixed.", "image_prompt": "Two chemicals mixing in a beaker"},
            {"description": "An observable change occurs (e.g., color change, gas formation, heat release).", "image_prompt": "Chemical reaction with color change and bubbles"},
            {"description": "Final products are formed.", "image_prompt": "New chemical products formed"}
        ]
    else:
        steps = [
            {"description": "Step 1: Initial setup of the experiment.", "image_prompt": "Science experiment initial setup"},
            {"description": "Step 2: Observation of an intermediate state.", "image_prompt": "Science experiment intermediate observation"},
            {"description": "Step 3: Recording of results or final state.", "image_prompt": "Science experiment final results"}
        ]
    return steps

def generate_image_placeholder(image_prompt: str) -> str:
    """
    Simulates image generation by returning a placeholder image URL or SVG representation.
    In a real application, this would integrate with DALL-E, Stable Diffusion, etc.
    """
    # Create a simple SVG or use a placeholder image service
    # For simplicity, we'll use a placeholder image service here.
    # Replace spaces with '+' for URL compatibility
    clean_prompt = image_prompt.replace(' ', '+').replace(',', '')
    return f"<img src='https://via.placeholder.com/250x150?text={clean_prompt}' alt='{image_prompt}' style='border:1px solid #ddd; margin-top: 10px; display: block;'>"

def visualize_experiment(experiment_description: str) -> str:
    """
    Orchestrates the visualization of a science experiment using the ChainofImages pattern.
    """
    if not experiment_description.strip():
        return "Please enter a description of a science experiment to visualize."

    experiment_steps = decompose_experiment(experiment_description)
    output_html = []

    output_html.append(f"<h2>Visualization for: {experiment_description}</h2><hr/>")

    for i, step in enumerate(experiment_steps):
        step_number = i + 1
        step_description = step["description"]
        image_prompt = step["image_prompt"]
        generated_image_html = generate_image_placeholder(image_prompt)

        output_html.append(f"<div style='margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 8px;'>")
        output_html.append(f"<h3>Step {step_number}: {step_description}</h3>")
        output_html.append(f"<p><i>Image prompt: \"{image_prompt}\"</i></p>")
        output_html.append(generated_image_html)
        output_html.append("</div>")

    return "".join(output_html)

# Gradio Interface
if __name__ == "__main__":
    demo = gr.Interface(
        fn=visualize_experiment,
        inputs=gr.Textbox(label="Describe the science experiment (e.g., Photosynthesis, Water Cycle, Chemical Reaction)", lines=3, placeholder="e.g., Explain the process of photosynthesis"),
        outputs=gr.HTML(label="Visualized Experiment Steps"),
        title="Interactive Science Experiment Visualizer (Chain of Images)",
        description="Enter a science experiment description, and see its step-by-step visual progression through generated (simulated) images."
    )
    demo.launch()
