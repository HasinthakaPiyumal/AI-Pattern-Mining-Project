import gradio as gr
import os
from genai_service import generate_text

def generate_blog_post(topic: str, style: str) -> str:
    """
    Generates a blog post draft based on the given topic and desired style
    using a Generative AI model.
    """
    if not topic or not style:
        return "Please provide both a topic and a desired style." 

    # Construct the prompt for the GenAI model, applying the Style Prompting pattern
    prompt = f"""
    As a professional content writer, generate a blog post on the topic of '{topic}'.
    The blog post should adhere strictly to the following stylistic guidelines:
    Style: {style}.

    Ensure the tone, vocabulary, and sentence structure align perfectly with the specified style.
    The blog post should be engaging and informative.

    Blog Post Draft:
    """
    
    print(f"Sending prompt to GenAI:\n{prompt}") # For debugging
    generated_content = generate_text(prompt)
    return generated_content

# --- Gradio Interface Setup ---
iface = gr.Interface(
    fn=generate_blog_post,
    inputs=[
        gr.Textbox(label="Blog Topic", placeholder="e.g., The Future of Remote Work"),
        gr.Textbox(label="Desired Style", placeholder="e.g., 'formal and informative', 'casual and humorous', 'sales-driven and persuasive'"),
    ],
    outputs=gr.Textbox(label="Generated Blog Post Draft", lines=15),
    title="Blog Post Style Guide Generator",
    description="Enter a topic and desired style to generate a blog post draft adhering to your stylistic requirements. This application leverages the 'Style Prompting' AI pattern."
)

if __name__ == "__main__":
    # To run this, you'll need to set your OPENAI_API_KEY environment variable
    # Example: export OPENAI_API_KEY='your_api_key_here'
    # Or uncomment the line below for direct setting (NOT recommended for production)
    # os.environ["OPENAI_API_KEY"] = "YOUR_ACTUAL_OPENAI_API_KEY"

    iface.launch()