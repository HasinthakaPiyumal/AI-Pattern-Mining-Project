import os
from typing import List, Dict

# try:
#     from openai import OpenAI
# except ImportError:
#     print("OpenAI library not found. Please install it using 'pip install openai' if you intend to use a real LLM.")

class MockLLM:
    def __init__(self):
        pass

    class MockChatCompletions:
        def create(self, messages, **kwargs):
            user_message = messages[-1]["content"]
            response_content = ""

            if "crucial attributes for 'running shoes'" in user_message.lower():
                response_content = "cushioning level, pronation support, terrain suitability, drop height"
            elif "diverse values for 'cushioning level'" in user_message.lower():
                response_content = "maximum, moderate, responsive, minimalist"
            elif "diverse values for 'pronation support'" in user_message.lower():
                response_content = "neutral, stability, motion control"
            elif "diverse values for 'terrain suitability'" in user_message.lower():
                response_content = "road, trail, track, mixed"
            elif "diverse values for 'drop height'" in user_message.lower():
                response_content = "low (0-4mm), medium (5-8mm), high (9+mm)"
            elif "generate a compelling product description" in user_message.lower():
                # Basic mock for description generation, incorporating attributes
                desc_template = "Experience the ultimate performance with our new running shoes. Featuring {cushioning_level} cushioning, {pronation_support} pronation support, and perfect for {terrain_suitability} terrain. With a {drop_height} drop height, these shoes are designed for excellence."
                
                cushioning_level = "" # Default values if not found in prompt
                pronation_support = ""
                terrain_suitability = ""
                drop_height = ""

                # Simple keyword extraction to fill template
                if "maximum cushioning" in user_message.lower(): cushioning_level = "maximum"
                elif "moderate cushioning" in user_message.lower(): cushioning_level = "moderate"
                elif "responsive cushioning" in user_message.lower(): cushioning_level = "responsive"
                elif "minimalist cushioning" in user_message.lower(): cushioning_level = "minimalist"

                if "neutral pronation support" in user_message.lower(): pronation_support = "neutral"
                elif "stability pronation support" in user_message.lower(): pronation_support = "stability"
                elif "motion control pronation support" in user_message.lower(): pronation_support = "motion control"

                if "road terrain" in user_message.lower(): terrain_suitability = "road"
                elif "trail terrain" in user_message.lower(): terrain_suitability = "trail"
                elif "track terrain" in user_message.lower(): terrain_suitability = "track"
                elif "mixed terrain" in user_message.lower(): terrain_suitability = "mixed"

                if "low drop height" in user_message.lower(): drop_height = "low (0-4mm)"
                elif "medium drop height" in user_message.lower(): drop_height = "medium (5-8mm)"
                elif "high drop height" in user_message.lower(): drop_height = "high (9+mm)"
                
                response_content = desc_template.format(
                    cushioning_level=cushioning_level or "unspecified",
                    pronation_support=pronation_support or "unspecified",
                    terrain_suitability=terrain_suitability or "unspecified",
                    drop_height=drop_height or "unspecified"
                )
            else:
                response_content = "Mock LLM response for: " + user_message[:50] + "..."

            class MockMessage:
                def __init__(self, content):
                    self.content = content

            class MockChoice:
                def __init__(self, message):
                    self.message = message

            return type("MockResponse", (object,), {"choices": [MockChoice(MockMessage(response_content))]})()

    @property
    def chat(self):
        return self.MockChatCompletions()

# Initialize LLM client
# If using actual OpenAI, uncomment the lines below and set OPENAI_API_KEY
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if OPENAI_API_KEY:
#     llm_client = OpenAI(api_key=OPENAI_API_KEY)
# else:
llm_client = MockLLM()

def call_llm(prompt: str) -> str:
    try:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or any other suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates product information."}, 
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error: {e}"

def generate_diversifying_attributes(product_category: str) -> List[str]:
    prompt = f"For '{product_category}', identify 3-5 crucial attributes that, when varied, lead to diverse product descriptions. List them comma-separated."
    response = call_llm(prompt)
    attributes = [attr.strip() for attr in response.split(',') if attr.strip()]
    return attributes

def generate_attribute_variations(attribute: str, product_category: str) -> List[str]:
    prompt = f"For '{product_category}' and the attribute '{attribute}', suggest 2-3 diverse values or variations. List them comma-separated."
    response = call_llm(prompt)
    variations = [var.strip() for var in response.split(',') if var.strip()]
    return variations

def generate_product_description_with_attributes(product_category: str, attribute_variations: Dict[str, str]) -> str:
    attribute_str = ", ".join([f"{attr}: {val}" for attr, val in attribute_variations.items()])
    prompt = f"Generate a compelling product description for '{product_category}'. Emphasize and integrate the following characteristics: {attribute_str}. Make it engaging and concise."
    response = call_llm(prompt)
    return response

def main():
    product_category = "running shoes"

    print(f"\n--- Generating diverse descriptions for: {product_category} ---")

    # Step 1: Generate diversifying attributes
    print("\n1. Identifying diversifying attributes...")
    attributes = generate_diversifying_attributes(product_category)
    print(f"Identified Attributes: {attributes}")

    if not attributes:
        print("No attributes identified. Exiting.")
        return

    # Step 2: Generate variations for each attribute
    print("\n2. Generating variations for each attribute...")
    attribute_variations_map: Dict[str, List[str]] = {}
    for attr in attributes:
        variations = generate_attribute_variations(attr, product_category)
        attribute_variations_map[attr] = variations
        print(f"  '{attr}' variations: {variations}")
    
    if not attribute_variations_map:
        print("No variations generated. Exiting.")
        return

    # Step 3: Generate product descriptions using different attribute combinations
    print("\n3. Generating product descriptions with varied attributes...")
    
    # For demonstration, we'll pick one variation for each attribute and generate a few descriptions
    # In a real application, you might generate many more combinations or use a more sophisticated sampling strategy.
    
    # Example 1: High cushioning, neutral support, road shoes, high drop
    selected_variations_1 = {
        "cushioning level": attribute_variations_map.get("cushioning level", [""])[0], # Maximum
        "pronation support": attribute_variations_map.get("pronation support", [""])[0], # Neutral
        "terrain suitability": attribute_variations_map.get("terrain suitability", [""])[0], # Road
        "drop height": attribute_variations_map.get("drop height", [""])[3] # high (9+mm)
    }
    print(f"\n--- Description 1 (Attributes: {selected_variations_1}) ---")
    desc1 = generate_product_description_with_attributes(product_category, selected_variations_1)
    print(desc1)

    # Example 2: Minimalist cushioning, stability support, trail shoes, low drop
    selected_variations_2 = {
        "cushioning level": attribute_variations_map.get("cushioning level", [""])[3], # Minimalist
        "pronation support": attribute_variations_map.get("pronation support", [""])[1], # Stability
        "terrain suitability": attribute_variations_map.get("terrain suitability", [""])[1], # Trail
        "drop height": attribute_variations_map.get("drop height", [""])[0] # low (0-4mm)
    }
    print(f"\n--- Description 2 (Attributes: {selected_variations_2}) ---")
    desc2 = generate_product_description_with_attributes(product_category, selected_variations_2)
    print(desc2)

    # Example 3: Responsive cushioning, motion control, track shoes, medium drop
    selected_variations_3 = {
        "cushioning level": attribute_variations_map.get("cushioning level", [""])[2], # Responsive
        "pronation support": attribute_variations_map.get("pronation support", [""])[2], # Motion control
        "terrain suitability": attribute_variations_map.get("terrain suitability", [""])[2], # Track
        "drop height": attribute_variations_map.get("drop height", [""])[1] # medium (5-8mm)
    }
    print(f"\n--- Description 3 (Attributes: {selected_variations_3}) ---")
    desc3 = generate_product_description_with_attributes(product_category, selected_variations_3)
    print(desc3)

if __name__ == "__main__":
    main()