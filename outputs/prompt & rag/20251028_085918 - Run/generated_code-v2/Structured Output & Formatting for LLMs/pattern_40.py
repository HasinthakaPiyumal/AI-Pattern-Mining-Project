import json
from enum import Enum
from fastapi import FastAPI, Body, APIRouter
from pydantic import BaseModel, Field

app = FastAPI()
router = APIRouter()

class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"

class ProductInput(BaseModel):
    name: str = Field(..., example="Wireless Bluetooth Headphones")
    key_features: list[str] = Field(..., example=["Noise-cancelling", "Long battery life", "Comfortable fit"])
    category: str = Field(..., example="Electronics")
    output_format: OutputFormat = Field(..., example=OutputFormat.JSON)

class DescriptionOutput(BaseModel):
    product_title: str
    short_description: str
    bullet_points_features: list[str]
    detailed_description: str
    seo_tags: list[str]

def generate_raw_description(product_input: ProductInput) -> DescriptionOutput:
    # This is a mock AI function. In a real application, this would involve
    # calling a large language model or a fine-tuned AI model.
    title = f"{product_input.name} for {product_input.category}"
    short_desc = f"Experience superior audio quality with our {product_input.name}. Featuring {', '.join(product_input.key_features)}."
    
    detailed_desc = (
        f"Dive into an immersive audio experience with our state-of-the-art {product_input.name}. "
        f"Designed with {', '.join(product_input.key_features)}, these headphones provide crystal-clear sound "
        f"and deep bass, perfect for music lovers and professionals alike. "
        f"Enjoy hours of uninterrupted listening with its robust battery life and ergonomic design."
    )
    
    seo_t = [f"{product_input.name.lower().replace(' ', '-')}", f"{product_input.category.lower()}"]
    seo_t.extend([f.lower().replace(' ', '-') for f in product_input.key_features])

    return DescriptionOutput(
        product_title=title,
        short_description=short_desc,
        bullet_points_features=product_input.key_features,
        detailed_description=detailed_desc,
        seo_tags=seo_t
    )

def format_to_json(data: DescriptionOutput) -> str:
    return json.dumps(data.model_dump(), indent=2)

def format_to_markdown(data: DescriptionOutput) -> str:
    markdown_features = "\n".join([f"* {feature}" for feature in data.bullet_points_features])
    markdown_seo_tags = ", ".join(data.seo_tags)

    return (
        f"# {data.product_title}\n\n"
        f"**Short Description:** {data.short_description}\n\n---\n\n"
        f"## Key Features:\n{markdown_features}\n\n---\n\n"
        f"## Detailed Description:\n{data.detailed_description}\n\n---\n\n"
        f"**SEO Tags:** {markdown_seo_tags}"
    )

@router.post("/generate-description", response_model=str, summary="Generate E-commerce Product Description")
async def generate_description(product_input: ProductInput = Body(..., description="Product information and desired output format")):
    raw_description = generate_raw_description(product_input)
    
    if product_input.output_format == OutputFormat.JSON:
        return format_to_json(raw_description)
    elif product_input.output_format == OutputFormat.MARKDOWN:
        return format_to_markdown(raw_description)
    else:
        # This case should ideally not be reached due to Pydantic enum validation
        return "Unsupported output format"

app.include_router(router)
