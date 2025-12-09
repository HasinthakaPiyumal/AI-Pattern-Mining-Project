import json

class PromptInputModule:
    def process_text_prompt(self, text_prompt: str) -> dict:
        keywords = [word for word in text_prompt.lower().split() if len(word) > 3]
        scene_elements = []
        if "forest" in keywords: scene_elements.append("trees")
        if "mountain" in keywords: scene_elements.append("rocks")
        if "river" in keywords: scene_elements.append("water_body")
        if "castle" in keywords: scene_elements.append("structure")
        style = "realistic" if "realistic" in keywords else "fantasy"
        mood = "calm" if "calm" in keywords else "epic"
        return {"type": "text", "keywords": keywords, "elements": scene_elements, "style": style, "mood": mood}

    def process_image_prompt(self, image_data: str) -> dict:
        # Simulated processing: Returns a fixed visual style
        return {"type": "image", "visual_style": "grungy_industrial"}

    def process_3d_object_prompt(self, object_data: str) -> dict:
        # Simulated processing: Returns a fixed object property
        return {"type": "3d_object", "mesh_detail": "high", "material_type": "metallic"}

class GenerativeAICore:
    def synthesize_3d_object(self, processed_prompt: dict) -> dict:
        if processed_prompt.get("type") == "text":
            obj_name = processed_prompt["elements"][0] if processed_prompt["elements"] else "generic_object"
            material = "wood" if "trees" in processed_prompt["elements"] else "stone"
            return {"object_id": f"obj_{obj_name}_001", "model_type": "mesh", "material": material, "complexity": "medium"}
        elif processed_prompt.get("type") == "3d_object":
            return {"object_id": "custom_object_001", "model_type": "imported_mesh", "material": processed_prompt["material_type"], "complexity": processed_prompt["mesh_detail"]}
        return {}

    def generate_texture(self, processed_prompt: dict) -> dict:
        if processed_prompt.get("type") == "text":
            pattern = "leafy" if "trees" in processed_prompt["elements"] else "rocky"
            color = "green" if "forest" in processed_prompt["keywords"] else "gray"
            return {"texture_id": f"tex_{pattern}_001", "pattern": pattern, "color": color, "resolution": "2048x2048"}
        elif processed_prompt.get("type") == "image":
            return {"texture_id": "tex_industrial_001", "pattern": "rusted_metal", "color": "brown_grey", "resolution": "4096x4096"}
        return {}

    def compose_scene(self, objects: list, textures: list, text_prompt_data: dict) -> dict:
        scene_elements_data = []
        for obj in objects:
            scene_elements_data.append({"type": "object", "id": obj["object_id"], "position": [0,0,0], "rotation": [0,0,0], "scale": [1,1,1], "material": obj["material"]})
        for tex in textures:
            scene_elements_data.append({"type": "texture", "id": tex["texture_id"], "usage": "terrain"})

        lighting = "daylight" if text_prompt_data.get("mood") == "calm" else "dramatic"
        environment = text_prompt_data.get("style", "fantasy")

        return {
            "scene_name": "Generated_Environment",
            "description": f"A {environment} scene generated with {lighting} lighting.",
            "elements": scene_elements_data,
            "lighting": lighting,
            "environment_style": environment
        }

class OutputModule:
    def emit_scene_description(self, scene_data: dict) -> str:
        return json.dumps(scene_data, indent=4)


class _3DEnvironmentGenerator:
    def __init__(self):
        self.prompt_input_module = PromptInputModule()
        self.generative_ai_core = GenerativeAICore()
        self.output_module = OutputModule()

    def generate_environment(self, text_prompt: str = "", image_prompt_data: str = "", object_prompt_data: str = "") -> str:
        processed_text = {}
        if text_prompt:
            processed_text = self.prompt_input_module.process_text_prompt(text_prompt)

        processed_image = {}
        if image_prompt_data:
            processed_image = self.prompt_input_module.process_image_prompt(image_prompt_data)
        
        processed_3d_object = {}
        if object_prompt_data:
            processed_3d_object = self.prompt_input_module.process_3d_object_prompt(object_prompt_data)

        generated_objects = []
        if processed_text: 
            generated_objects.append(self.generative_ai_core.synthesize_3d_object(processed_text))
        if processed_3d_object:
            generated_objects.append(self.generative_ai_core.synthesize_3d_object(processed_3d_object))

        generated_textures = []
        if processed_text: 
            generated_textures.append(self.generative_ai_core.generate_texture(processed_text))
        if processed_image: 
            generated_textures.append(self.generative_ai_core.generate_texture(processed_image))
        
        scene_composition = self.generative_ai_core.compose_scene(generated_objects, generated_textures, processed_text)
        
        return self.output_module.emit_scene_description(scene_composition)

if __name__ == "__main__":
    generator = _3DEnvironmentGenerator()

    # Example 1: Text prompt only
    print("\n--- Generating with Text Prompt ---")
    scene1 = generator.generate_environment(text_prompt="A calm forest scene with a small river.")
    print(scene1)

    # Example 2: Text prompt and simulated image prompt
    print("\n--- Generating with Text and Image Prompt ---")
    scene2 = generator.generate_environment(text_prompt="An epic mountain range, realistic style.", image_prompt_data="concept_art_grungy_industrial.png")
    print(scene2)

    # Example 3: Text prompt and simulated 3D object prompt
    print("\n--- Generating with Text and 3D Object Prompt ---")
    scene3 = generator.generate_environment(text_prompt="A fantasy castle environment.", object_prompt_data="detailed_medieval_tower.obj")
    print(scene3)
