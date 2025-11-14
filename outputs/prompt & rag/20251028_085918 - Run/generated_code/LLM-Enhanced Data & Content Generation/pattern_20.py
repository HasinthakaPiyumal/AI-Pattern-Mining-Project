""" 
This module implements a Personalized E-commerce Product Recommendation and Content Generation System 
leveraging Large Language Models (LLMs) as described in the project architecture. 
It demonstrates functionalities for personalized recommendations, LLM-powered content generation, 
synthetic data generation, and feedback handling, all exposed via a FastAPI backend.

Key components:
- Pydantic models for data validation.
- A simplified recommendation engine.
- An LLM-powered content generator using Hugging Face transformers and LangChain (conceptual).
- A synthetic data generator using LLMs.
- A feedback mechanism for iterative content refinement.
- FastAPI for exposing the functionalities as a REST API.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# Try to import transformers, but handle it gracefully if not installed
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Warning: transformers library not found. LLM functionalities will be simulated.")
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

# Try to import langchain, but handle it gracefully if not installed
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.llms import OpenAI # Placeholder, replace with actual LLM integration
except ImportError:
    print("Warning: langchain_core or langchain_community library not found. LangChain functionalities will be simulated.")
    PromptTemplate = None
    StrOutputParser = None
    OpenAI = None

# --- Environment Variables --- 
load_dotenv() # Load environment variables from .env file

# --- Pydantic Models for Data Validation --- 
class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    price: float
    inventory: int

class User(BaseModel):
    id: str
    name: str
    preferences: List[str] = [] # Categories/tags the user likes
    purchase_history: List[str] = [] # Product IDs purchased

class Feedback(BaseModel):
    user_id: str
    content_id: str # ID of the content being reviewed (e.g., product ID, generated content ID)
    content_type: str # e.g., "product_description", "faq"
    rating: int = Field(..., ge=1, le=5) # 1 to 5 stars
    comment: Optional[str] = None

# --- Simulated Data Store --- 
# In a real application, these would be databases (e.g., PostgreSQL, MongoDB, vector DB)
PRODUCTS_DB: Dict[str, Product] = {
    "p1": Product(id="p1", name="Wireless Bluetooth Headphones", description="Premium over-ear headphones with noise cancellation.", category="Electronics", tags=["audio", "wireless", "noise-cancelling"], price=199.99, inventory=50),
    "p2": Product(id="p2", name="Ergonomic Office Chair", description="Adjustable chair for maximum comfort during long work hours.", category="Home & Office", tags=["furniture", "office", "comfort"], price=349.00, inventory=20),
    "p3": Product(id="p3", name="Smartwatch with Health Tracker", description="Monitor your fitness and health with this advanced smartwatch.", category="Electronics", tags=["wearable", "fitness", "health"], price=129.50, inventory=100),
    "p4": Product(id="p4", name="Organic Green Tea Kit", description="A selection of fine organic green teas for relaxation and wellness.", category="Food & Beverage", tags=["tea", "organic", "wellness"], price=25.00, inventory=75),
    "p5": Product(id="p5", name="Portable SSD 1TB", description="High-speed external solid-state drive for fast data transfer.", category="Electronics", tags=["storage", "data", "portable"], price=110.00, inventory=60),
}

USERS_DB: Dict[str, User] = {
    "u1": User(id="u1", name="Alice", preferences=["Electronics", "audio"], purchase_history=["p1"]),
    "u2": User(id="u2", name="Bob", preferences=["Home & Office", "comfort"], purchase_history=["p2"]),
    "u3": User(id="u3", name="Charlie", preferences=["Food & Beverage", "wellness"], purchase_history=["p4"]),
}

FEEDBACK_DB: List[Feedback] = []

# --- Recommendation Engine (Simplified Content-Based) --- 
class RecommendationEngine:
    def get_recommendations(self, user_id: str, limit: int = 5) -> List[Product]:
        user = USERS_DB.get(user_id)
        if not user:
            return []
        
        # Simple content-based filtering: recommend products matching user preferences (categories/tags)
        recommended_products = []
        for product_id, product in PRODUCTS_DB.items():
            if product_id not in user.purchase_history: # Don't recommend already purchased items
                if any(pref in product.category for pref in user.preferences) or \
                   any(tag in user.preferences for tag in product.tags):
                    recommended_products.append(product)
        
        # Sort by price (example sorting) and take top N
        recommended_products.sort(key=lambda p: p.price, reverse=True)
        return recommended_products[:limit]

# --- LLM-Powered Content Generator --- 
class LLMContentGenerator:
    def __init__(self):
        self.generator = None
        self.llm_chain = None
        self.init_llm_model()

    def init_llm_model(self):
        if pipeline and AutoTokenizer and AutoModelForCausalLM:
            try:
                # Using a smaller model for demonstration, e.g., 'distilgpt2'
                # For more advanced use, replace with a larger model or a hosted API
                print("Loading LLM model (distilgpt2) for content generation...")
                self.generator = pipeline(
                    "text-generation", 
                    model="distilgpt2", 
                    tokenizer="distilgpt2",
                    max_new_tokens=100
                )
                print("LLM model loaded successfully.")

                # LangChain integration (conceptual)
                if PromptTemplate and StrOutputParser and OpenAI:
                    # For a real application, configure OpenAI with your API key
                    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
                    # self.llm = OpenAI(temperature=0.7)
                    self.llm = None # Simulate if no API key or actual LLM is configured
                    
                    product_description_template = PromptTemplate.from_template(
                        """You are an expert e-commerce copywriter. Generate a compelling, SEO-friendly product description for the following product:
                        Product Name: {product_name}
                        Category: {product_category}
                        Existing Description: {existing_description}
                        Key Features/Tags: {product_tags}
                        
                        Focus on benefits, unique selling points, and encourage purchase. Keep it under 200 words.
                        """
                    )
                    self.product_description_chain = product_description_template | StrOutputParser()
                    
                    faq_template = PromptTemplate.from_template(
                        """You are a helpful e-commerce assistant. Generate 3 frequently asked questions (FAQs) and their answers 
                        for the following product, based on its description and category. Make sure the answers are concise and informative.
                        Product Name: {product_name}
                        Category: {product_category}
                        Description: {product_description}
                        
                        Format: Q: [Question]\nA: [Answer]\nQ: [Question]\nA: [Answer]\nQ: [Question]\nA: [Answer]
                        """
                    )
                    self.faq_chain = faq_template | StrOutputParser()

            except Exception as e:
                print(f"Error loading LLM model or LangChain components: {e}")
                self.generator = None
                self.llm_chain = None
        else:
            print("LLM generation will be simulated due to missing libraries.")

    def generate_content(self, product: Product, content_type: str) -> str:
        if not self.generator:
            return f"[Simulated LLM Content for {content_type} for {product.name}] - LLM libraries not loaded."
        
        if content_type == "description":
            if self.llm and self.product_description_chain:
                # Use LangChain if available and configured
                try:
                    return self.product_description_chain.invoke({
                        "product_name": product.name,
                        "product_category": product.category,
                        "existing_description": product.description,
                        "product_tags": ", ".join(product.tags)
                    })
                except Exception as e:
                    print(f"LangChain description generation failed: {e}. Falling back to basic generation.")
                    pass # Fallback to basic generation

            # Fallback or direct transformers generation
            prompt = f"Write a compelling product description for {product.name} (Category: {product.category}, Tags: {', '.join(product.tags)}). Existing description: {product.description}. Focus on benefits and features:"
            return self._generate_text(prompt)

        elif content_type == "faq":
            if self.llm and self.faq_chain:
                # Use LangChain if available and configured
                try:
                    return self.faq_chain.invoke({
                        "product_name": product.name,
                        "product_category": product.category,
                        "product_description": product.description
                    })
                except Exception as e:
                    print(f"LangChain FAQ generation failed: {e}. Falling back to basic generation.")
                    pass # Fallback to basic generation

            # Fallback or direct transformers generation
            prompt = f"Generate 3 frequently asked questions and answers for the product {product.name} (Description: {product.description}):\nQ1:"
            return self._generate_text(prompt)
        
        elif content_type == "marketing_copy":
            prompt = f"Create a short, engaging marketing slogan or ad copy for {product.name} (Category: {product.category}, Description: {product.description}). Highlight its main benefit:"
            return self._generate_text(prompt)
        else:
            return f"Unsupported content type: {content_type}"

    def _generate_text(self, prompt: str) -> str:
        if not self.generator:
            return f"[Simulated Generation: {prompt[:50]}...]"
        try:
            res = self.generator(prompt)
            generated_text = res[0]["generated_text"]
            # Post-process to remove the prompt itself from the output
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()
        except Exception as e:
            print(f"Error during LLM text generation: {e}")
            return f"[Error generating content with LLM: {e}]"

# --- Synthetic Data Generator --- 
class SyntheticDataGenerator:
    def __init__(self, llm_generator: LLMContentGenerator):
        self.llm_generator = llm_generator

    def generate_synthetic_product(self, base_category: str = "General") -> Product:
        if not self.llm_generator.generator:
            print("Simulating synthetic product generation due to missing LLM libraries.")
            new_id = f"p_synth_{len(PRODUCTS_DB) + 1}"
            return Product(
                id=new_id,
                name=f"Simulated Product {new_id}",
                description=f"A randomly generated product in {base_category}.",
                category=base_category,
                tags=["synthetic", base_category.lower()],
                price=float(f"{len(PRODUCTS_DB) * 10 + 9.99:.2f}"),
                inventory=int(len(PRODUCTS_DB) * 5 + 20)
            )

        prompt = f"Generate a unique, compelling new e-commerce product idea in the {base_category} category. \nProvide its Name, a brief Description, and 3-5 relevant Tags (comma-separated). \nFormat: Name: [Product Name]\nDescription: [Product Description]\nTags: [Tag1, Tag2, Tag3]\n"
        
        generated_text = self.llm_generator._generate_text(prompt)
        
        # Parse the generated text into a Product object
        name = "Unknown Synthetic Product"
        description = "No description provided."
        tags = []
        
        for line in generated_text.split('\n'):
            if line.startswith("Name:"):
                name = line.replace("Name:", "").strip()
            elif line.startswith("Description:"):
                description = line.replace("Description:", "").strip()
            elif line.startswith("Tags:"):
                tags = [t.strip() for t in line.replace("Tags:", "").split(',') if t.strip()]

        new_id = f"p_synth_{len(PRODUCTS_DB) + 1}"
        # Add a placeholder price and inventory for synthetic product
        return Product(
            id=new_id,
            name=name,
            description=description,
            category=base_category, # Keep base category or try to infer from LLM output
            tags=tags or [base_category.lower(), "synthetic"],
            price=float(f"{len(PRODUCTS_DB) * 10 + 9.99:.2f}"), # Dummy price
            inventory=int(len(PRODUCTS_DB) * 5 + 20) # Dummy inventory
        )

# --- Feedback Handler --- 
class FeedbackHandler:
    def submit_feedback(self, feedback: Feedback):
        FEEDBACK_DB.append(feedback)
        print(f"Feedback received: User {feedback.user_id} rated {feedback.content_type} for {feedback.content_id} as {feedback.rating} stars.")
        # In a real system, this feedback would be used to fine-tune prompts or models.
        # For this demo, we just store it.

    def get_all_feedback(self) -> List[Feedback]:
        return FEEDBACK_DB

# --- FastAPI Application --- 
app = FastAPI(
    title="LLM-Powered E-commerce Augmentation System",
    description="API for personalized product recommendations, LLM-driven content generation, and synthetic data.",
    version="1.0.0",
)

# Initialize components
recommendation_engine = RecommendationEngine()
llm_content_generator = LLMContentGenerator()
synthetic_data_generator = SyntheticDataGenerator(llm_content_generator)
feedback_handler = FeedbackHandler()

@app.get("/products", response_model=List[Product], summary="Get all products")
async def get_products():
    return list(PRODUCTS_DB.values())

@app.get("/products/{product_id}", response_model=Product, summary="Get product by ID")
async def get_product(product_id: str):
    if product_id not in PRODUCTS_DB:
        raise HTTPException(status_code=404, detail="Product not found")
    return PRODUCTS_DB[product_id]

@app.get("/users", response_model=List[User], summary="Get all users")
async def get_users():
    return list(USERS_DB.values())

@app.get("/recommendations/{user_id}", response_model=List[Product], summary="Get personalized product recommendations for a user")
async def get_product_recommendations(user_id: str):
    if user_id not in USERS_DB:
        raise HTTPException(status_code=404, detail="User not found")
    return recommendation_engine.get_recommendations(user_id)

@app.post("/generate_content/{product_id}/{content_type}", summary="Generate content (description, FAQ, marketing copy) for a product using LLM")
async def generate_product_content(product_id: str, content_type: str):
    if product_id not in PRODUCTS_DB:
        raise HTTPException(status_code=404, detail="Product not found")
    
    product = PRODUCTS_DB[product_id]
    generated_text = llm_content_generator.generate_content(product, content_type)
    return {"product_id": product_id, "content_type": content_type, "generated_content": generated_text}

@app.post("/generate_synthetic_product", response_model=Product, summary="Generate a new synthetic product using LLM (addressing data scarcity)")
async def generate_new_synthetic_product(base_category: str = "General"): 
    new_product = synthetic_data_generator.generate_synthetic_product(base_category)
    PRODUCTS_DB[new_product.id] = new_product # Add to our simulated DB
    return new_product

@app.post("/feedback", summary="Submit user feedback for generated content or products")
async def submit_user_feedback(feedback: Feedback):
    feedback_handler.submit_feedback(feedback)
    return {"message": "Feedback submitted successfully."}

@app.get("/feedback", response_model=List[Feedback], summary="Get all submitted feedback")
async def get_all_feedback():
    return feedback_handler.get_all_feedback()

if __name__ == "__main__":
    import uvicorn
    # To run the API, save this file as main.py and run: uvicorn main:app --reload
    print("\n--- LLM-Powered E-commerce Augmentation System --- ")
    print("To run the API, use the command: uvicorn main:app --reload")
    print(f"API documentation available at: http://127.0.0.1:8000/docs")
    print("Simulated LLM initialization will proceed. Check for 'Warning' messages if libraries are missing.")
    
    # The LLM model will be initialized when LLMContentGenerator is instantiated
    # To avoid blocking, in a real scenario, this might be done asynchronously or pre-loaded
    # For this script, we'll let the constructor handle it.
    uvicorn.run(app, host="127.0.0.1", port=8000)

