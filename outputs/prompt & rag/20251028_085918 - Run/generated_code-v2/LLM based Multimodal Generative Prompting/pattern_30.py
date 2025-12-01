import gradio as gr
import os

# Placeholder for a text generation model
def generate_story_text(premise):
    if not premise:
        return "Please provide a story premise."
    return f"Based on the premise \"{premise}\", a new scene unfolds: The protagonist, a brave adventurer, finds themselves at the entrance of a forgotten temple, overgrown with ancient vines. A sense of mystery and danger permeates the air."

# Placeholder for an image generation model
def generate_image(prompt):
    # In a real application, this would call an image generation API (e.g., Stable Diffusion)
    # and return a path to a generated image file.
    # For this placeholder, we simulate a file path.
    dummy_image_path = os.path.join("generated_media", f"image_{hash(prompt)}.png")
    # Create a dummy file to avoid Gradio errors, though it won't be a real image
    os.makedirs("generated_media", exist_ok=True)
    with open(dummy_image_path, "w") as f:
        f.write("Dummy image content for: " + prompt)
    return dummy_image_path

# Placeholder for an audio generation model
def generate_audio(prompt):
    # In a real application, this would call an audio generation API (e.g., MusicGen/SoundGen)
    # and return a path to a generated audio file.
    # For this placeholder, we simulate a file path.
    dummy_audio_path = os.path.join("generated_media", f"audio_{hash(prompt)}.mp3")
    # Create a dummy file to avoid Gradio errors, though it won't be real audio
    os.makedirs("generated_media", exist_ok=True)
    with open(dummy_audio_path, "w") as f:
        f.write("Dummy audio content for: " + prompt)
    return dummy_audio_path

def generate_scene(
    story_premise,
    art_style,
    setting,
    character_look,
    lighting,
    music_genre,
    sound_effects,
    emotional_tone
):
    # 1. Generate scene description (story text)
    scene_description = generate_story_text(story_premise)

    # 2. Construct image prompt using modifiers
    image_prompt = (
        f"{scene_description}. {art_style} art style. Set in {setting}. "
        f"Character looks: {character_look}. Lighting: {lighting}."
    )

    # 3. Construct audio prompt using modifiers
    audio_prompt = (
        f"Background music for a scene: {scene_description}. "
        f"Music genre: {music_genre}. Sound effects: {sound_effects}. "
        f"Emotional tone: {emotional_tone}."
    )

    # 4. Generate image and audio using constructed prompts
    generated_image_path = generate_image(image_prompt)
    generated_audio_path = generate_audio(audio_prompt)

    return scene_description, generated_image_path, generated_audio_path


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Interactive Story Weaver")
    gr.Markdown("Enter your story premise and use modifiers to shape the generated scene.")

    with gr.Row():
        with gr.Column():
            story_premise_input = gr.Textbox(label="Story Premise", placeholder="A lone knight on a quest...")
            art_style_input = gr.Dropdown(
                ["realistic", "cartoon", "oil painting", "watercolor", "sci-fi art"],
                label="Art Style",
                value="realistic"
            )
            setting_input = gr.Textbox(label="Setting", placeholder="A mystical forest, an ancient ruin...")
            character_look_input = gr.Textbox(label="Character Look", placeholder="Armored, wizardly, robed...")
            lighting_input = gr.Textbox(label="Lighting", placeholder="Sunlit, dim, dramatic, glowing...")
            music_genre_input = gr.Dropdown(
                ["orchestral", "ambient", "electronic", "folk", "ominous"],
                label="Music Genre",
                value="orchestral"
            )
            sound_effects_input = gr.Textbox(label="Sound Effects", placeholder="Wind howling, dragon's roar, sword clashing...")
            emotional_tone_input = gr.Dropdown(
                ["epic", "calm", "tense", "joyful", "melancholic"],
                label="Emotional Tone",
                value="epic"
            )
            generate_button = gr.Button("Generate Scene")

        with gr.Column():
            output_text = gr.Textbox(label="Generated Story Text", interactive=False)
            output_image = gr.Image(label="Generated Visuals")
            output_audio = gr.Audio(label="Generated Soundscape/Music")

    generate_button.click(
        fn=generate_scene,
        inputs=[
            story_premise_input,
            art_style_input,
            setting_input,
            character_look_input,
            lighting_input,
            music_genre_input,
            sound_effects_input,
            emotional_tone_input
        ],
        outputs=[output_text, output_image, output_audio]
    )

demo.launch()