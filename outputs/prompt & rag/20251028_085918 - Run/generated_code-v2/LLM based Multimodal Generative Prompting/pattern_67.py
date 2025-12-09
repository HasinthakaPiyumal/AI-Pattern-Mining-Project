import gradio as gr
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# --- Mock Data and Services ---

# Simulate a furniture catalog
furniture_catalog = {
    "sofa_modern": {"name": "Modern Grey Sofa", "description": "A sleek, contemporary grey sofa.", "color": "grey"},
    "sofa_classic": {"name": "Classic Leather Sofa", "description": "A comfortable, traditional leather sofa.", "color": "brown"},
    "coffee_table_wood": {"name": "Wooden Coffee Table", "description": "A rustic wooden coffee table.", "color": "wood"},
    "rug_persian": {"name": "Persian Area Rug", "description": "An ornate red and blue Persian rug.", "color": "red and blue"},
    "lamp_floor": {"name": "Tall Floor Lamp", "description": "A minimalist black floor lamp.", "color": "black"},
    "bookcase": {"name": "Large Bookcase", "description": "A tall, open bookcase.", "color": "white"},
}

def get_furniture_details(item_key):
    return furniture_catalog.get(item_key, {"name": "Unknown Item", "description": "", "color": ""})

# Mock image generation function
def generate_mock_image(prompt, width=700, height=400, background_color="#e0e0e0"):
    img = Image.new('RGB', (width, height), color=background_color)
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20) # Using a common font
    except IOError:
        font = ImageFont.load_default() # Fallback to default font

    text_color = (50, 50, 50)

    # Wrap text for better display
    words = prompt.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if d.textlength(" ".join(current_line), font=font) > width - 40:
            lines.pop()
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    lines.append(" ".join(current_line))

    y_text = 20
    for line in lines:
        d.text((20, y_text), line, fill=text_color, font=font)
        y_text += 25

    # Add a simple border to indicate it's an image area
    d.rectangle([(0, 0), (width - 1, height - 1)], outline="black", width=2)

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Chain of Images Logic ---

def get_current_design_summary(room_description, placed_furniture):
    summary = f"You've described the room as: {room_description}.\n"
    if not placed_furniture:
        summary += "Currently, there's no furniture placed in your design. "
    else:
        summary += "So far, you've added: " + ", ".join([get_furniture_details(item)['name'] for item in placed_furniture]) + ". "
    return summary


def generate_next_prompt_and_suggestion(room_description, placed_furniture, user_feedback):
    summary = get_current_design_summary(room_description, placed_furniture)
    
    next_suggestion = ""
    next_prompt_base = f"Visualizing a room based on: '{room_description}'. "

    if not placed_furniture:
        # First step: visualize the empty room or suggest a main piece
        next_prompt = f"{next_prompt_base} Let's think image by image. First, let's visualize the empty space and imagine the main area. What kind of sofa would you like to place? (e.g., 'sofa_modern', 'sofa_classic')"
        next_suggestion = "What's the first major furniture piece you'd like to add? (e.g., 'sofa_modern', 'sofa_classic')"
    else:
        # Subsequent steps: incorporate feedback or suggest next items
        if user_feedback and user_feedback.startswith("add_"):
            item_key_to_add = user_feedback[4:] # e.g., 'add_rug_persian'
            if item_key_to_add in furniture_catalog:
                item_details = get_furniture_details(item_key_to_add)
                placed_furniture.append(item_key_to_add)
                current_items_names = [get_furniture_details(item)['name'] for item in placed_furniture]
                next_prompt = f"{next_prompt_base} Let's add '{item_details['name']}'. The room now features: {', '.join(current_items_names)}. How does it look? What's next?"
                next_suggestion = f"The {item_details['name']} has been added. What else would you like to add or change? (e.g., 'add_coffee_table_wood', 'move_sofa')"
            else:
                next_prompt = f"{next_prompt_base} I couldn't find '{item_key_to_add}' in the catalog. Please try a valid item key. Current design: {', '.join([get_furniture_details(item)['name'] for item in placed_furniture])}."
                next_suggestion = "Item not found. Please choose from: " + ", ".join(furniture_catalog.keys()) + ". Or describe a new action."
        elif user_feedback and "remove" in user_feedback:
            item_to_remove = user_feedback.replace("remove ", "").strip()
            initial_len = len(placed_furniture)
            placed_furniture = [item for item in placed_furniture if get_furniture_details(item)['name'].lower() != item_to_remove.lower() and item.lower() != item_to_remove.lower()]
            if len(placed_furniture) < initial_len:
                current_items_names = [get_furniture_details(item)['name'] for item in placed_furniture]
                next_prompt = f"{next_prompt_base} Removed '{item_to_remove}'. The room now features: {', '.join(current_items_names) if current_items_names else 'an empty space'}. What's next?"
                next_suggestion = f"'{item_to_remove}' removed. What's your next design idea?"
            else:
                next_prompt = f"{next_prompt_base} Couldn't find '{item_to_remove}' to remove. Current design: {', '.join([get_furniture_details(item)['name'] for item in placed_furniture])}."
                next_suggestion = f"Couldn't find '{item_to_remove}'. Try 'remove Modern Grey Sofa' or 'remove sofa_modern'."
        elif user_feedback and "move" in user_feedback or "change" in user_feedback:
             # For simplicity, we'll just acknowledge and regenerate the image with the same items
            current_items_names = [get_furniture_details(item)['name'] for item in placed_furniture]
            next_prompt = f"{next_prompt_base} Considering your feedback: '{user_feedback}'. Let's regenerate the image with current items: {', '.join(current_items_names)}. What next?"
            next_suggestion = f"Acknowledged: '{user_feedback}'. What's your next design step? (e.g., 'add_lamp_floor', 'remove sofa_classic')"
        else:
            # If no specific action, prompt for more input
            current_items_names = [get_furniture_details(item)['name'] for item in placed_furniture]
            next_prompt = f"{next_prompt_base} Let's continue. The current design includes: {', '.join(current_items_names)}. What would you like to add, remove, or change?"
            next_suggestion = f"What's your next design step? Current items: {', '.join(current_items_names) if current_items_names else 'empty'}. (e.g., 'add_rug_persian', 'remove sofa_modern')"
    
    return next_prompt, next_suggestion, placed_furniture


def design_assistant_flow(room_description, user_feedback, current_design_state_json):
    # Parse current_design_state from JSON
    import json
    if current_design_state_json:
        current_design_state = json.loads(current_design_state_json)
    else:
        current_design_state = {"placed_furniture": []}
    
    placed_furniture = current_design_state["placed_furniture"]

    next_prompt, next_suggestion, updated_placed_furniture = generate_next_prompt_and_suggestion(
        room_description, placed_furniture, user_feedback
    )

    generated_image_b64 = generate_mock_image(next_prompt)

    # Update the design state
    updated_design_state = {"placed_furniture": updated_placed_furniture}
    
    return f"data:image/png;base64,{generated_image_b64}", next_suggestion, json.dumps(updated_design_state)


# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Interior Design Assistant (Chain of Images)
        Upload a room photo or describe your room, then interactively design by adding/removing furniture.
        The AI will generate 'visual thoughts' (mock images) at each step.
        """
    )

    with gr.Row():
        room_description_input = gr.Textbox(label="Describe your room (e.g., 'a modern living room with large windows')", lines=2, placeholder="e.g., 'A spacious living room with hardwood floors and light grey walls.'")
        current_design_state = gr.State(value="") # To store the placed furniture

    with gr.Row():
        image_output = gr.Image(label="Your Designed Room (AI Visual Thought)", type="pil", height=400)

    with gr.Row():
        user_feedback_input = gr.Textbox(label="Your next design idea or feedback (e.g., 'add_sofa_modern', 'remove rug_persian', 'move sofa to the left')", lines=1)
    
    with gr.Row():
        submit_btn = gr.Button("Generate Next Visual Thought")
        ai_suggestion_output = gr.Textbox(label="AI's Next Suggestion", interactive=False)

    submit_btn.click(
        design_assistant_flow,
        inputs=[room_description_input, user_feedback_input, current_design_state],
        outputs=[image_output, ai_suggestion_output, current_design_state]
    )

    gr.Markdown(
        """
        ### Available Furniture (Use these keys with 'add_'):
        - `sofa_modern`: Modern Grey Sofa
        - `sofa_classic`: Classic Leather Sofa
        - `coffee_table_wood`: Wooden Coffee Table
        - `rug_persian`: Persian Area Rug
        - `lamp_floor`: Tall Floor Lamp
        - `bookcase`: Large Bookcase
        """
    )

demo.launch()