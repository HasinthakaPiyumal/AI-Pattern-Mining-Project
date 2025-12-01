
import json
import enum
from textwrap import dedent

# Placeholder for a hypothetical LLM client
# In a real application, this would be replaced with actual LLM integration (e.g., OpenAI, Cohere, Hugging Face Transformers)
class MockLLM:
    def generate(self, prompt: str) -> str:
        # Simulate an LLM response based on the prompt's instructions for formatting
        print(f"\n--- Mock LLM received prompt ---\n{prompt}\n---\n")
        if "JSON" in prompt:
            return json.dumps({
                "product_name": "[Generated Product Name]",
                "description": "This is a beautifully crafted product description in JSON format. It highlights key features and targets the specified audience.",
                "features": ["Feature 1", "Feature 2", "Feature 3"],
                "target_audience": "[Target Audience]",
                "call_to_action": "Buy now!"
            }, indent=2)
        elif "XML" in prompt:
            return dedent("""
                <product>
                    <name>[Generated Product Name]</name>
                    <description>This is an eloquent product description in XML format, emphasizing its unique selling points for its intended users.</description>
                    <features>
                        <feature>Feature One</feature>
                        <feature>Feature Two</feature>
                        <feature>Feature Three</feature>
                    </features>
                    <audience>[Target Audience]</audience>
                    <cta>Order Today!</cta>
                </product>
            """)
        elif "Markdown" in prompt:
            return dedent("""
                # [Generated Product Name]

                This is an outstanding product description presented in clean Markdown. We've highlighted its core benefits and who it's perfect for.

                ## Key Features
                *   **Feature A:** Detailed explanation of feature A.
                *   **Feature B:** Detailed explanation of feature B.
                *   **Feature C:** Detailed explanation of feature C.

                ## Who is this for?
                [Target Audience]

                ## Get Yours Now!
                Click here to purchase and experience the difference.
            """)
        else:
            return "This is a generic product description with no specific formatting instructions applied."


class ProductDetails:
    def __init__(self, name: str, features: list[str], audience: str):
        self.name = name
        self.features = features
        self.audience = audience

class OutputFormat(enum.Enum):
    JSON = "JSON"
    XML = "XML"
    MARKDOWN = "Markdown"

def generate_product_description_prompt(
    product: ProductDetails,
    output_format: OutputFormat
) -> str:
    """Generates a prompt for the LLM to create a product description in the specified format."""

    format_instructions = {
        OutputFormat.JSON: dedent("""
            Your response MUST be a valid JSON object. The JSON should have the following keys: `product_name`, `description`, `features` (an array of strings), `target_audience`, and `call_to_action`.
            Example:
            ```json
            {
              "product_name": "Example Product",
              "description": "A concise description.",
              "features": ["Feature 1", "Feature 2"],
              "target_audience": "General Public",
              "call_to_action": "Learn More"
            }
            ```
            """),
        OutputFormat.XML: dedent("""
            Your response MUST be a valid XML structure. The root element should be `<product>`, and it should contain `<name>`, `<description>`, `<features>` (with `<feature>` child elements), `<audience>`, and `<cta>` elements.
            Example:
            ```xml
            <product>
              <name>Example Gadget</name>
              <description>An amazing gadget.</description>
              <features>
                <feature>Feature X</feature>
                <feature>Feature Y</feature>
              </features>
              <audience>Tech Enthusiasts</audience>
              <cta>Buy Today!</cta>
            </product>
            ```
            """),
        OutputFormat.MARKDOWN: dedent("""
            Your response MUST be formatted using Markdown. Use H1 for the product name, H2 for sections like 'Key Features' and 'Who is this for?', and bullet points for features.
            Example:
            ```markdown
            # Example Product Title

            A brief, engaging introduction to the product.

            ## Key Features
            *   **Feature A:** Benefit of feature A.
            *   **Feature B:** Benefit of feature B.

            ## Who is this for?
            People who love innovation.

            ## Call to Action
            Discover more now!
            ```
            """)
    }

    prompt_template = dedent(f"""
        You are an expert e-commerce copywriter. Your task is to generate a compelling product description.

        Product Name: {product.name}
        Key Features: {', '.join(product.features)}
        Target Audience: {product.audience}

        Please generate the product description in {output_format.value} format.

        {format_instructions[output_format]}
        """)

    return prompt_template.strip()

def generate_formatted_description(
    product: ProductDetails,
    output_format: OutputFormat,
    llm_client: MockLLM
) -> str:
    """Generates a product description using an LLM, ensuring it adheres to the specified output format."""
    prompt = generate_product_description_prompt(product, output_format)
    generated_text = llm_client.generate(prompt)
    return generated_text

if __name__ == "__main__":
    # Initialize the mock LLM client
    llm = MockLLM()

    # Example Product Details
    product_camera = ProductDetails(
        name="Ultra-HD Smart Camera",
        features=["4K Resolution", "AI Motion Tracking", "Cloud Storage", "Night Vision"],
        audience="Homeowners, Small Businesses, Security-conscious individuals"
    )

    product_watch = ProductDetails(
        name="AstroFit Smartwatch",
        features=["Heart Rate Monitor", "GPS Tracking", "Waterproof", "Long Battery Life"],
        audience="Fitness Enthusiasts, Outdoor Adventurers, Tech-savvy Users"
    )

    print("\n--- Generating JSON Output ---")
    json_description = generate_formatted_description(product_camera, OutputFormat.JSON, llm)
    print(json_description)

    print("\n--- Generating XML Output ---")
    xml_description = generate_formatted_description(product_watch, OutputFormat.XML, llm)
    print(xml_description)

    print("\n--- Generating Markdown Output ---")
    markdown_description = generate_formatted_description(product_camera, OutputFormat.MARKDOWN, llm)
    print(markdown_description)

    print("\n--- Generating another Markdown Output ---")
    markdown_description_watch = generate_formatted_description(product_watch, OutputFormat.MARKDOWN, llm)
    print(markdown_description_watch)

