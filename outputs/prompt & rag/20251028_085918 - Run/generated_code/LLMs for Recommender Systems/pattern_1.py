import uuid
from typing import List, Dict, Any, Optional
import json

# Mocking ChromaDB components
class MockChromaClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

class MockChromaCollection:
    def __init__(self, name):
        self.name = name
        self.data = [] # Stores {'id': ..., 'embedding': ..., 'metadata': ..., 'document': ...}

    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None, embeddings: Optional[List[List[float]]] = None):
        for i, doc in enumerate(documents):
            item_id = ids[i] if ids else str(uuid.uuid4())
            embedding = embeddings[i] if embeddings else [hash(doc) % 1000 / 1000.0] * 384 # Simple mock embedding
            metadata = metadatas[i] if metadatas else {}
            self.data.append({
                "id": item_id,
                "embedding": embedding,
                "metadata": metadata,
                "document": doc
            })

    def query(self, query_texts: List[str], n_results: int = 10, where: Optional[Dict] = None, include: Optional[List[str]] = None) -> Dict:
        # Simple mock query: just return all documents for now, in a real system this would use embeddings
        results = []
        for doc_item in self.data:
            match = True
            if where:
                for key, value in where.items():
                    if doc_item['metadata'].get(key) != value:
                        match = False
                        break
            if match:
                result_item = {}
                if include is None or "documents" in include:
                    result_item["document"] = doc_item["document"]
                if include is None or "metadatas" in include:
                    result_item["metadata"] = doc_item["metadata"]
                if include is None or "ids" in include:
                    result_item["id"] = doc_item["id"]
                results.append(result_item)
        
        # In a real scenario, this would sort by similarity score
        return {"ids": [r["id"] for r in results[:n_results]],
                "documents": [r["document"] for r in results[:n_results] if "document" in r],
                "metadatas": [r["metadata"] for r in results[:n_results] if "metadata" in r]}


# Mock LangChain components
class MockBaseChatModel:
    def invoke(self, messages: List[Dict], **kwargs) -> str:
        # Simulate LLM response based on messages
        last_message_content = messages[-1]["content"].lower()

        if "search_products_tool" in last_message_content:
            if "laptop" in last_message_content and "gaming" in last_message_content:
                return "```json\n{\"tool_name\": \"search_products_tool\", \"args\": {\"query\": \"gaming laptop\", \"category\": \"Electronics\"}}\n```"
            elif "durable outdoor gear" in last_message_content:
                return "```json\n{\"tool_name\": \"search_products_tool\", \"args\": {\"query\": \"durable outdoor gear\", \"category\": \"Outdoor & Sports\"}}\n```"
            elif "laptops under $1000 with at least 16GB RAM for gaming" in last_message_content:
                 return "```json\n{\"tool_name\": \"search_products_tool\", \"args\": {\"query\": \"gaming laptop\", \"category\": \"Electronics\", \"max_price\": 1000, \"features\": \"16GB RAM\"}}\n```"
            else:
                return f"```json\n{{\"tool_name\": \"search_products_tool\", \"args\": {{\"query\": \"{last_message_content}\"\}}}}\n```"
        elif "get_product_details_tool" in last_message_content:
            product_id_match = last_message_content.split("product_id=")[-1].split(")")[0].strip().replace('"', '')
            return f"```json\n{{\"tool_name\": \"get_product_details_tool\", \"args\": {{\"product_id\": \"{product_id_match}\"\}}}}\n```"
        elif "compare_products_tool" in last_message_content:
             product_ids_str = last_message_content.split("product_ids=")[-1].split(")")[0].strip().replace('"', '').replace('[', '').replace(']', '')
             product_ids = [pid.strip() for pid in product_ids_str.split(',')]
             return f"```json\n{{\"tool_name\": \"compare_products_tool\", \"args\": {{\"product_ids\": {json.dumps(product_ids)}\}}}}\n```"
        elif "check_stock_delivery_tool" in last_message_content:
            product_id_match = last_message_content.split("product_id=")[-1].split(")")[0].strip().replace('"', '')
            return f"```json\n{{\"tool_name\": \"check_stock_delivery_tool\", \"args\": {{\"product_id\": \"{product_id_match}\"\}}}}\n```"
        elif "generate_explanation_tool" in last_message_content:
            parts = last_message_content.split("product_id=")
            product_id = parts[1].split(",")[0].strip().replace('"', '')
            user_prefs = parts[2].split(")")[0].strip().replace('"', '')
            return f"```json\n{{\"tool_name\": \"generate_explanation_tool\", \"args\": {{\"product_id\": \"{product_id}\", \"user_preferences\": \"{user_prefs}\"\}}}}\n```"

        # General chat/explanation responses
        if "explain why" in last_message_content or "justify" in last_message_content:
            return "Based on your preferences, this product is ideal because it offers [key feature 1] and [key feature 2], which aligns with your need for [user need]."
        elif "hello" in last_message_content or "hi" in last_message_content:
            return "Hello! How can I assist you with your shopping today?"
        elif "thank you" in last_message_content:
            return "You're welcome! Let me know if you need anything else."
        elif "recommend" in last_message_content:
             return "I can help with recommendations! What kind of product are you looking for?"
        else:
            return f"I'm not sure how to respond to '{last_message_content}'. Can you rephrase or ask about products?"

class MockAgentExecutor:
    def __init__(self, llm: MockBaseChatModel, tools: List[Any], memory: Any, agent_type: str = "openai-tools"):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory
        self.agent_type = agent_type

    def invoke(self, input_dict: Dict) -> Dict:
        user_query = input_dict["input"]
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        # Simulate LLM deciding to use a tool or respond directly
        # In a real LangChain agent, this would involve prompt engineering
        # and structured output parsing to determine tool calls.
        # Here, we'll try to parse a mock tool call.
        llm_response = self.llm.invoke([{"role": "user", "content": user_query}])
        self.memory.save_context({"input": user_query}, {"output": llm_response})

        try:
            # Attempt to parse the mock tool call format
            if llm_response.startswith("```json") and llm_response.endswith("```"):
                tool_call_str = llm_response.strip("```json\n").strip("\n```")
                tool_call_data = json.loads(tool_call_str)
                tool_name = tool_call_data["tool_name"]
                tool_args = tool_call_data["args"]

                if tool_name in self.tools:
                    print(f"\n>>> Calling tool: {tool_name} with args: {tool_args}")
                    tool_output = self.tools[tool_name].func(**tool_args)
                    response = f"Tool output for {tool_name}: {tool_output}"
                    self.memory.save_context({"input": f"Tool output: {tool_output}"}, {"output": response})
                    return {"output": response}
                else:
                    return {"output": f"LLM tried to call unknown tool: {tool_name}"}
        except (json.JSONDecodeError, KeyError) as e:
            pass # Not a tool call, or malformed, proceed to direct response
        
        # If no tool was called or parsing failed, return direct LLM response
        return {"output": llm_response}


class MockConversationBufferMemory:
    def __init__(self):
        self.buffer = []

    def load_memory_variables(self, inputs: Dict) -> Dict:
        return {"chat_history": self.buffer}

    def save_context(self, inputs: Dict, outputs: Dict):
        self.buffer.append(f"Human: {inputs['input']}")
        self.buffer.append(f"AI: {outputs['output']}")


class Tool:
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func
        self.description = description

# --- Product Data Layer ---

class ProductDataManager:
    def __init__(self, products_data: List[Dict]):
        self.products = {p['id']: p for p in products_data}
        self.chroma_client = MockChromaClient()
        self.product_collection = self.chroma_client.get_or_create_collection("products")
        self._initialize_chroma_db(products_data)

    def _initialize_chroma_db(self, products_data: List[Dict]):
        docs = [p['description'] for p in products_data]
        metadatas = [{
            "id": p['id'], 
            "name": p['name'], 
            "category": p['category'], 
            "price": p['price'], 
            "stock": p['stock'],
            "features": json.dumps(p['features']) # Store features as JSON string
        } for p in products_data]
        ids = [p['id'] for p in products_data]
        self.product_collection.add(documents=docs, metadatas=metadatas, ids=ids)

    def add_product(self, product_id: str, name: str, description: str, category: str, price: float, stock: int, features: List[str]):
        new_product = {
            "id": product_id,
            "name": name,
            "description": description,
            "category": category,
            "price": price,
            "stock": stock,
            "features": features
        }
        self.products[product_id] = new_product
        self.product_collection.add(
            documents=[description],
            metadatas=[{
                "id": product_id, 
                "name": name, 
                "category": category, 
                "price": price, 
                "stock": stock,
                "features": json.dumps(features)
            }],
            ids=[product_id]
        )
        print(f"Product {name} added.")

    def enrich_product_description(self, product_description: str) -> Dict:
        # Mock LLM interaction for enrichment
        # In a real scenario, an LLM would parse the text and extract structured info
        if "laptop" in product_description.lower():
            return {"features": ["High Performance", "Portable", "Long Battery Life"], "category": "Electronics"}
        elif "tent" in product_description.lower():
            return {"features": ["Waterproof", "Lightweight", "Easy Setup"], "category": "Outdoor & Sports"}
        else:
            return {"features": ["General Use"], "category": "Miscellaneous"}

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        return self.products.get(product_id)

    def search_products_semantic(self, query: str) -> List[Dict]:
        # In a real setup, this uses embeddings for similarity search.
        # For mock, we'll do a simple keyword match on documents + return metadata.
        results = self.product_collection.query(query_texts=[query], n_results=5, include=["metadatas", "documents", "ids"])
        found_products = []
        for i in range(len(results["ids"])):
            product_id = results["ids"][i]
            # For mocking, just fetch full product data using the ID, 
            # as semantic search is simplified.
            product_data = self.get_product_by_id(product_id)
            if product_data:
                found_products.append(product_data)

        # Simple keyword matching fallback for semantic mock
        if not found_products:
            query_lower = query.lower()
            for product_id, product in self.products.items():
                if query_lower in product['name'].lower() or query_lower in product['description'].lower():
                    if product not in found_products:
                        found_products.append(product)

        return found_products

    def search_products_filtered(self, category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None, features: Optional[str] = None) -> List[Dict]:
        filtered_products = []
        for product_id, product in self.products.items():
            match = True
            if category and product['category'].lower() != category.lower():
                match = False
            if min_price is not None and product['price'] < min_price:
                match = False
            if max_price is not None and product['price'] > max_price:
                match = False
            if features:
                required_features = [f.strip().lower() for f in features.split(',')]
                product_features_lower = [f.lower() for f in product['features']]
                if not all(rf in product_features_lower for rf in required_features):
                    match = False
            
            if match:
                filtered_products.append(product)
        return filtered_products

# --- Tooling Layer (for LangChain Agent) ---

class EcommerceTools:
    def __init__(self, product_manager: ProductDataManager):
        self.product_manager = product_manager

    def search_products_tool(self, query: str, category: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None, features: Optional[str] = None) -> str:
        """Searches for products based on a natural language query and optional filters. 
        Args: query (str): The natural language search query. category (str): Optional category to filter by. 
        min_price (float): Optional minimum price. max_price (float): Optional maximum price. 
        features (str): Comma-separated string of required features. """
        print(f"[TOOL] Executing search_products_tool with query='{query}', category='{category}', min_price='{min_price}', max_price='{max_price}', features='{features}'")
        
        semantic_results = self.product_manager.search_products_semantic(query)
        
        # If specific filters are provided, apply them
        if category or min_price is not None or max_price is not None or features:
            filtered_results = self.product_manager.search_products_filtered(category, min_price, max_price, features)
            # Combine results, prioritize filtered results if they are more specific
            # For this mock, we'll return filtered if present, otherwise semantic
            if filtered_results:
                results = filtered_results
            else:
                results = semantic_results
        else:
            results = semantic_results

        if results:
            return "Found products:\n" + "\n".join([f"- {p['name']} ({p['id']}): {p['description']} - ${p['price']} (Stock: {p['stock']})" for p in results])
        return "No products found matching your criteria."

    def get_product_details_tool(self, product_id: str) -> str:
        """Retrieves detailed information for a specific product by its ID.
        Args: product_id (str): The ID of the product. """
        print(f"[TOOL] Executing get_product_details_tool for ID: {product_id}")
        product = self.product_manager.get_product_by_id(product_id)
        if product:
            return f"Details for {product['name']} (ID: {product_id}):\nCategory: {product['category']}\nPrice: ${product['price']}\nDescription: {product['description']}\nFeatures: {', '.join(product['features'])}\nStock: {product['stock']}"
        return f"Product with ID {product_id} not found."

    def compare_products_tool(self, product_ids: List[str]) -> str:
        """Compares multiple products given their IDs.
        Args: product_ids (List[str]): A list of product IDs to compare. """
        print(f"[TOOL] Executing compare_products_tool for IDs: {product_ids}")
        products = [self.product_manager.get_product_by_id(pid) for pid in product_ids if self.product_manager.get_product_by_id(pid)]
        if not products:
            return "No valid products found for comparison."
        
        comparison_str = "Product Comparison:\n"
        for p in products:
            comparison_str += f"\n--- {p['name']} ({p['id']}) ---\n"
            comparison_str += f"  Category: {p['category']}\n"
            comparison_str += f"  Price: ${p['price']}\n"
            comparison_str += f"  Features: {', '.join(p['features'])}\n"
            comparison_str += f"  Stock: {p['stock']}\n"
        return comparison_str

    def check_stock_delivery_tool(self, product_id: str) -> str:
        """Checks the stock availability and estimated delivery for a given product ID (mocked).
        Args: product_id (str): The ID of the product. """
        print(f"[TOOL] Executing check_stock_delivery_tool for ID: {product_id}")
        product = self.product_manager.get_product_by_id(product_id)
        if product:
            if product['stock'] > 0:
                return f"Product {product['name']} (ID: {product_id}) is in stock. Estimated delivery: 3-5 business days."
            else:
                return f"Product {product['name']} (ID: {product_id}) is currently out of stock."
        return f"Product with ID {product_id} not found."

    def generate_explanation_tool(self, product_id: str, user_preferences: str) -> str:
        """Generates a personalized explanation for why a product is recommended, based on user preferences (mocked by LLM).
        Args: product_id (str): The ID of the product. user_preferences (str): A description of the user's preferences. """
        print(f"[TOOL] Executing generate_explanation_tool for ID: {product_id} with preferences: {user_preferences}")
        product = self.product_manager.get_product_by_id(product_id)
        if product:
            # In a real system, the LLM would dynamically generate this explanation
            return f"Based on your interest in '{user_preferences}', the {product['name']} is a great choice because its features like {', '.join(product['features'])} directly address your needs for a product in the '{product['category']}' category."
        return f"Could not generate explanation for product ID {product_id} as it was not found."

# --- User Interaction Layer ---

class EcommerceAssistant:
    def __init__(self):
        # Sample Product Data
        products_data = [
            {
                "id": "P001",
                "name": "Gaming Laptop Xtreme",
                "description": "High-performance gaming laptop with RTX 3080 and 32GB RAM.",
                "category": "Electronics",
                "price": 1800.00,
                "stock": 5,
                "features": ["High Performance", "Gaming", "32GB RAM", "RTX 3080", "SSD"]
            },
            {
                "id": "P002",
                "name": "Ultra Portable Laptop",
                "description": "Lightweight laptop perfect for travel and productivity. 8GB RAM.",
                "category": "Electronics",
                "price": 950.00,
                "stock": 12,
                "features": ["Portable", "Lightweight", "8GB RAM", "Long Battery Life"]
            },
            {
                "id": "P003",
                "name": "Family Camping Tent",
                "description": "Spacious, waterproof tent for 4 people, easy to set up.",
                "category": "Outdoor & Sports",
                "price": 250.00,
                "stock": 20,
                "features": ["Waterproof", "Spacious", "Easy Setup", "4-person"]
            },
             {
                "id": "P004",
                "name": "Rugged Hiking Boots",
                "description": "Durable and comfortable boots for long hikes and challenging terrains.",
                "category": "Outdoor & Sports",
                "price": 120.00,
                "stock": 15,
                "features": ["Durable", "Comfortable", "Water-resistant", "Hiking"]
            },
             {
                "id": "P005",
                "name": "Smartwatch Pro",
                "description": "Advanced smartwatch with heart rate monitoring and GPS. Fitness focused.",
                "category": "Electronics",
                "price": 300.00,
                "stock": 30,
                "features": ["Fitness Tracking", "GPS", "Heart Rate Monitor", "Waterproof"]
            }
        ]

        self.product_manager = ProductDataManager(products_data)
        self.llm = MockBaseChatModel()
        self.memory = MockConversationBufferMemory()
        self.tools_instance = EcommerceTools(self.product_manager)

        self.tools = [
            Tool(name="search_products_tool", func=self.tools_instance.search_products_tool,
                 description=self.tools_instance.search_products_tool.__doc__),
            Tool(name="get_product_details_tool", func=self.tools_instance.get_product_details_tool,
                 description=self.tools_instance.get_product_details_tool.__doc__),
            Tool(name="compare_products_tool", func=self.tools_instance.compare_products_tool,
                 description=self.tools_instance.compare_products_tool.__doc__),
            Tool(name="check_stock_delivery_tool", func=self.tools_instance.check_stock_delivery_tool,
                 description=self.tools_instance.check_stock_delivery_tool.__doc__),
            Tool(name="generate_explanation_tool", func=self.tools_instance.generate_explanation_tool,
                 description=self.tools_instance.generate_explanation_tool.__doc__),
        ]

        # LangChain AgentExecutor setup
        # In a real LangChain setup, ChatOpenAI or similar would be used for llm_with_tools
        # For this mock, the MockAgentExecutor directly uses the mock LLM to parse tool calls.
        self.agent = MockAgentExecutor(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            agent_type="openai-tools" # This hints at the expected parsing style for the mock LLM
        )

    def handle_user_query(self, query: str):
        print(f"\nUser: {query}")
        response = self.agent.invoke({"input": query})
        print(f"Assistant: {response['output']}")

def main():
    print("Welcome to the LLM-Enhanced E-commerce Assistant! Type 'exit' to quit.")
    assistant = EcommerceAssistant()

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() == 'exit':
            print("Thank you for shopping with us! Goodbye.")
            break
        assistant.handle_user_query(user_input)

if __name__ == "__main__":
    main()
