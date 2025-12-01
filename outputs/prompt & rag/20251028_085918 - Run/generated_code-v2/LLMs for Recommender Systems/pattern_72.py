from typing import List, Dict, Any, Optional
import json

class MockChatOpenAI:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7, **kwargs):
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: List[Any], **kwargs) -> Any:
        last_message_content = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                last_message_content = msg.content
                break

        if "search_products" in last_message_content:
            return MockAIMessage(content="I found some products. Would you like to know more about them?")
        elif "get_product_details" in last_message_content:
            return MockAIMessage(content="Here are the details for that product.")
        elif "check_inventory" in last_message_content:
            return MockAIMessage(content="The product is in stock.")
        elif "add_to_cart" in last_message_content:
            return MockAIMessage(content="I've added the item to your cart.")
        elif "get_user_order_history" in last_message_content:
            return MockAIMessage(content="Here is your recent order history.")
        elif "tool_code" in last_message_content: # Simulate LLM interpreting tool output
            try:
                tool_output_str = last_message_content.split("Tool Output: ", 1)[1]
                tool_output = json.loads(tool_output_str)
                if isinstance(tool_output, list) and tool_output:
                    if "name" in tool_output[0]:
                        names = [item['name'] for item in tool_output if 'name' in item]
                        return MockAIMessage(content=f"I found these: {', '.join(names)}. Which one are you interested in?")
                    elif "status" in tool_output[0] and tool_output[0].get("status") == "delivered":
                         return MockAIMessage(content="Your recent orders include delivered items. Do you want more details?")
                elif isinstance(tool_output, dict) and "status" in tool_output and tool_output["status"] == "success":
                    return MockAIMessage(content=tool_output.get("message", "Action completed successfully."))
                elif isinstance(tool_output, dict) and "description" in tool_output:
                    return MockAIMessage(content=f"The {tool_output['name']} costs ${tool_output['price']}. Description: {tool_output['description']}")
                elif isinstance(tool_output, dict) and "available_stock" in tool_output:
                    return MockAIMessage(content=f"There are {tool_output['available_stock']} units of {tool_output['product_id']} in stock.")

            except json.JSONDecodeError:
                pass
            except IndexError:
                pass
            except KeyError:
                pass

        return MockAIMessage(content=f"Hello! How can I help you with your shopping today? (Mock LLM response to: '{last_message_content}')")

class MockTool:
    def __init__(self, name: str, func: callable, description: str):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MockAgentExecutor:
    def __init__(self, agent: Any, tools: List[MockTool], verbose: bool = False, handle_parsing_errors: bool = False, **kwargs):
        self.agent = agent
        self.tools = tools
        self.verbose = verbose
        self.handle_parsing_errors = handle_parsing_errors

    def invoke(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        user_input = input.get("input", "")
        chat_history = input.get("chat_history", [])

        if self.verbose:
            print(f"\n--- Agent Input ---\nUser Input: {user_input}\nChat History: {chat_history}\n-------------------")

        # Very simplified mock agent logic: if user input mentions a tool, simulate tool call
        # This mock tries to extract arguments from the user_input in a very basic way.
        for tool in self.tools:
            if tool.name.replace("_", " ") in user_input.lower():
                if self.verbose:
                    print(f"Agent Thought: User input '{user_input}' seems to require tool '{tool.name}'. Simulating tool call.")

                tool_args = {}
                # Basic arg extraction for demonstration
                if tool.name == "search_products":
                    query = next((s.strip() for s in user_input.lower().split("search for")[-1].split("in category")[0].split("products about")[-1].split("find")[-1].split(" ") if s), "generic product")
                    category = next((s.strip() for s in user_input.lower().split("in category")[-1].split(" ") if s), None) if "in category" in user_input.lower() else None
                    tool_args = {"query": query, "category": category}
                elif tool.name == "get_product_details":
                    # Assuming a product ID might be mentioned, or we use a default
                    product_id = next((word for word in user_input.split() if word.startswith('P') and len(word) > 1), "P101")
                    tool_args = {"product_id": product_id}
                elif tool.name == "check_inventory":
                    product_id = next((word for word in user_input.split() if word.startswith('P') and len(word) > 1), "P101")
                    tool_args = {"product_id": product_id}
                elif tool.name == "add_to_cart":
                    product_id = next((word for word in user_input.split() if word.startswith('P') and len(word) > 1), "P101")
                    quantity = int(next((s for s in user_input.split() if s.isdigit()), "1"))
                    tool_args = {"product_id": product_id, "quantity": quantity}
                elif tool.name == "get_user_order_history":
                    tool_args = {"user_id": input.get("user_profile", {}).get("id", "U001")}

                try:
                    tool_output = tool.func(**tool_args)
                    if self.verbose:
                        print(f"Tool Output: {tool_output}")
                    # Pass the tool output back to the LLM to generate a natural language response
                    # Simulate the LLM receiving a message about tool output
                    llm_response = self.agent.invoke(chat_history + [MockHumanMessage(content=f"tool_code: {tool.name}, Tool Output: {json.dumps(tool_output)}")]).content
                    return {"output": llm_response}
                except Exception as e:
                    if self.handle_parsing_errors:
                        if self.verbose:
                            print(f"Tool call error: {e}")
                        return {"output": f"There was an error using the {tool.name} tool. {str(e)}"}
                    raise

        # If no tool is explicitly triggered, just pass the user input to the LLM for general conversation
        llm_response = self.agent.invoke(chat_history + [MockHumanMessage(content=user_input)]).content
        return {"output": llm_response}


class MockHumanMessage:
    def __init__(self, content: str):
        self.content = content

class MockAIMessage:
    def __init__(self, content: str):
        self.content = content

def create_react_agent(llm: Any, tools: List[MockTool], prompt: Any) -> Any:
    class MockReactAgent:
        def __init__(self, llm: Any, tools: List[MockTool], prompt: Any):
            self.llm = llm
            self.tools = tools
            self.prompt = prompt

        def invoke(self, messages: List[Any], **kwargs) -> Any:
            return self.llm.invoke(messages, **kwargs)
    return MockReactAgent(llm, tools, prompt)

class MockChatPromptTemplate:
    def __init__(self, messages: List[Any]):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages_tuples: List[Any]):
        messages = []
        for msg_type, content in messages_tuples:
            if msg_type == "system":
                messages.append(MockSystemMessage(content=content))
            elif msg_type == "human":
                messages.append(MockHumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(MockAIMessage(content=content))
            elif isinstance(content, MessagesPlaceholder):
                messages.append(content)
        return cls(messages)

    def format_messages(self, **kwargs) -> List[Any]:
        formatted_messages = []
        for msg in self.messages:
            if isinstance(msg, MessagesPlaceholder):
                formatted_messages.extend(kwargs.get(msg.variable_name, []))
            elif hasattr(msg, 'content'):
                if "{input}" in msg.content and "input" in kwargs:
                    formatted_messages.append(type(msg)(content=msg.content.format(input=kwargs["input"])))
                else:
                    formatted_messages.append(msg)
            else:
                formatted_messages.append(msg)
        return formatted_messages

class MockSystemMessage:
    def __init__(self, content: str):
        self.content = content

class MessagesPlaceholder:
    def __init__(self, variable_name: str):
        self.variable_name = variable_name

class ECommerceTools:
    def search_products(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if "laptop" in query.lower():
            return [
                {"product_id": "P101", "name": "Dell XPS 13", "price": 1200.00},
                {"product_id": "P102", "name": "MacBook Air M2", "price": 1100.00}
            ]
        elif "book" in query.lower():
            return [
                {"product_id": "B201", "name": "The Lord of the Rings", "price": 25.00},
                {"product_id": "B202", "name": "Dune", "price": 18.00}
            ]
        return []

    def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        if product_id == "P101":
            return {"product_id": "P101", "name": "Dell XPS 13", "price": 1200.00, "description": "High-performance ultrabook with a stunning display.", "in_stock": True}
        elif product_id == "B201":
            return {"product_id": "B201", "name": "The Lord of the Rings", "price": 25.00, "description": "Epic fantasy novel by J.R.R. Tolkien.", "in_stock": True}
        return None

    def check_inventory(self, product_id: str) -> Dict[str, Any]:
        if product_id in ["P101", "B201"]:
            return {"product_id": product_id, "available_stock": 50, "in_stock": True}
        return {"product_id": product_id, "available_stock": 0, "in_stock": False}

    def add_to_cart(self, product_id: str, quantity: int) -> Dict[str, Any]:
        return {"status": "success", "message": f"Added {quantity} of {product_id} to cart."}

    def get_user_order_history(self, user_id: str) -> List[Dict[str, Any]]:
        if user_id == "U001":
            return [
                {"order_id": "ORD001", "items": [{"product_id": "P100", "name": "Wireless Mouse", "quantity": 1}], "total": 25.00, "status": "delivered"},
                {"order_id": "ORD002", "items": [{"product_id": "P101", "name": "Dell XPS 13", "quantity": 1}], "total": 1200.00, "status": "shipped"}
            ]
        return []


class MemoryManager:
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.conversation_histories: Dict[str, List[Any]] = {}

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_profiles.setdefault(user_id, {"preferences": [], "cart": {}, "id": user_id})

    def update_user_profile(self, user_id: str, key: str, value: Any):
        profile = self.get_user_profile(user_id)
        profile[key] = value

    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Any]:
        return self.conversation_histories.setdefault(user_id, [])[-limit:]

    def add_message_to_history(self, user_id: str, message: Any):
        self.conversation_histories.setdefault(user_id, []).append(message)


class LLMPersonalShoppingAssistant:

    def __init__(self, user_id: str = "U001"):
        self.user_id = user_id
        self.ecommerce_tools = ECommerceTools()
        self.memory_manager = MemoryManager()

        self.llm = MockChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

        tools = [
            MockTool(
                name="search_products",
                func=self.ecommerce_tools.search_products,
                description="Searches for products in the e-commerce catalog. Use this when the user asks to find products or browse. Args: query (str), category (optional str)."
            ),
            MockTool(
                name="get_product_details",
                func=self.ecommerce_tools.get_product_details,
                description="Retrieves detailed information about a specific product. Use this when the user asks for details about a product. Args: product_id (str)."
            ),
            MockTool(
                name="check_inventory",
                func=self.ecommerce_tools.check_inventory,
                description="Checks the current stock level for a product. Use this when the user asks about product availability. Args: product_id (str)."
            ),
            MockTool(
                name="add_to_cart",
                func=self.ecommerce_tools.add_to_cart,
                description="Adds a specified quantity of a product to the user's shopping cart. Use this when the user expresses intent to purchase or add to cart. Args: product_id (str), quantity (int)."
            ),
            MockTool(
                name="get_user_order_history",
                func=self.ecommerce_tools.get_user_order_history,
                description="Retrieves the past order history for the current user. Use this when the user asks about their previous orders. Args: user_id (str)."
            ),
        ]

        system_message = "You are an intelligent e-commerce personal shopping assistant. Your goal is to help users find products, get information, and make purchases. Be friendly, helpful, and concise. Use the available tools to assist the user."
        prompt = MockChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.agent_executor = MockAgentExecutor(
            agent=create_react_agent(self.llm, tools, prompt),
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def chat(self, user_input: str) -> str:
        chat_history = self.memory_manager.get_conversation_history(self.user_id)
        user_profile = self.memory_manager.get_user_profile(self.user_id)

        self.memory_manager.add_message_to_history(self.user_id, MockHumanMessage(content=user_input))

        try:
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "user_profile": user_profile
            })
            agent_response_content = response["output"]
        except Exception as e:
            agent_response_content = f"I apologize, but I encountered an issue: {e}. Could you please rephrase your request?"

        self.memory_manager.add_message_to_history(self.user_id, MockAIMessage(content=agent_response_content))

        return agent_response_content

if __name__ == "__main__":
    assistant = LLMPersonalShoppingAssistant(user_id="U001")
    print("Welcome to your Personal Shopping Assistant!")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Goodbye! Happy shopping!")
            break
        
        response = assistant.chat(user_query)
        print(f"Assistant: {response}")
