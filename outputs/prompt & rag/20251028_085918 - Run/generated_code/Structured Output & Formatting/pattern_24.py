from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class ProductDescription(BaseModel):
    product_name: str
    brand: str
    short_description: str
    features: List[str]
    specifications: Dict[str, Any]

class ProductInput(BaseModel):
    product_name: str
    category: str
    key_features: List[str]
    target_audience: str

app = FastAPI()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert e-commerce copywriter. Generate a product description in JSON format based on the user's input. The JSON must adhere to the ProductDescription schema."),
    ("user", "Product Name: {product_name}\nCategory: {category}\nKey Features: {key_features}\nTarget Audience: {target_audience}\n\nGenerate a detailed product description in JSON format, including product_name, brand, short_description, features (as a list), and specifications (as key-value pairs). The brand can be a placeholder if not provided.")
])

# Using JsonOutputParser for demonstration, although with_structured_output is mentioned
# For direct Pydantic parsing with LangChain, a different setup might be used in newer LangChain versions
# However, the architecture mentions `with_structured_output` for validation. 
# For a single file with direct JSON output, JsonOutputParser is a common way to guide the LLM.
# If `with_structured_output` was to be strictly adhered to with Pydantic output, 
# the chain definition would be slightly different, often involving `RunnableWithMessageHistory` or similar.
# For this simplified example, we'll guide the LLM to output JSON and parse it.

# A more direct approach with structured output as per architecture, if using LangChain's specific feature:
# from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel
# from langchain_core.runnables import RunnableParallel
# If ProductDescription was PydanticV1BaseModel, then:
# structured_llm = llm.with_structured_output(ProductDescription)
# chain = prompt_template | structured_llm

# For this simplified example adhering to the prompt engineering principle:
chain = prompt_template | llm | JsonOutputParser()

@app.post("/generate-description", response_model=ProductDescription)
async def generate_product_description(product_input: ProductInput):
    response = await chain.ainvoke({
        "product_name": product_input.product_name,
        "category": product_input.category,
        "key_features": ", ".join(product_input.key_features),
        "target_audience": product_input.target_audience
    })
    # The JsonOutputParser will return a dictionary, which Pydantic will validate
    return response
