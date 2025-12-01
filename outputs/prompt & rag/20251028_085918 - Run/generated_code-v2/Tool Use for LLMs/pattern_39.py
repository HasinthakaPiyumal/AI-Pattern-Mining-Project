import json

class ECommerceToolRegistry:
    def product_search(self, query: str):
        if "laptop" in query.lower():
            return {"status": "success", "data": [{"id": "P101", "name": "Dell XPS 13 Laptop", "price": 1200.00}, {"id": "P102", "name": "MacBook Air M2", "price": 1300.00}]}
        elif "headphones" in query.lower():
            return {"status": "success", "data": [{"id": "P201", "name": "Sony WH-1000XM5", "price": 350.00}]}
        else:
            return {"status": "success", "data": []}

    def order_management(self, order_id: str):
        if order_id == "ORD123":
            return {"status": "success", "data": {"order_id": "ORD123", "status": "shipped", "items": ["Dell XPS 13"], "total": 1200.00}}
        else:
            return {"status": "error", "message": "Order not found"}

    def customer_support_ticket(self, issue: str, user_id: str):
        return {"status": "success", "data": {"ticket_id": "TICKET456", "issue": issue, "user_id": user_id, "status": "opened"}}

    def process_payment(self, amount: float, card_info: str):
        if len(card_info) == 16 and card_info.isdigit():
            return {"status": "success", "data": {"transaction_id": "TRX789", "amount": amount, "status": "completed"}}
        else:
            return {"status": "error", "message": "Invalid card information"}

    def recommend_products(self, user_id: str):
        if user_id == "USER001":
            return {"status": "success", "data": [{"id": "P301", "name": "Wireless Mouse", "price": 25.00}, {"id": "P302", "name": "Keyboard", "price": 70.00}]}
        else:
            return {"status": "success", "data": []}

class MockLLM:
    def __init__(self):
        pass

    def predict(self, prompt: str):
        if "search for" in prompt.lower() or "find product" in prompt.lower():
            query_start = prompt.lower().find("query:") + len("query:")
            query_end = prompt.lower().find("\noutput format:")
            query = prompt[query_start:query_end].strip()
            return json.dumps({"tool": "product_search", "args": {"query": query}})
        elif "what is the status of order" in prompt.lower() or "track order" in prompt.lower():
            order_id_start = prompt.lower().find("order id:") + len("order id:")
            order_id_end = prompt.lower().find("\noutput format:")
            order_id = prompt[order_id_start:order_id_end].strip()
            return json.dumps({"tool": "order_management", "args": {"order_id": order_id}})
        elif "open a support ticket" in prompt.lower() or "have an issue" in prompt.lower():
            issue_start = prompt.lower().find("issue:") + len("issue:")
            issue_end = prompt.lower().find("user id:")
            issue = prompt[issue_start:issue_end].strip()
            user_id_start = prompt.lower().find("user id:") + len("user id:")
            user_id_end = prompt.lower().find("\noutput format:")
            user_id = prompt[user_id_start:user_id_end].strip()
            return json.dumps({"tool": "customer_support_ticket", "args": {"issue": issue, "user_id": user_id}})
        elif "make a payment" in prompt.lower() or "process payment" in prompt.lower():
            amount_start = prompt.lower().find("amount:") + len("amount:")
            amount_end = prompt.lower().find("card info:")
            amount = float(prompt[amount_start:amount_end].strip())
            card_info_start = prompt.lower().find("card info:") + len("card info:")
            card_info_end = prompt.lower().find("\noutput format:")
            card_info = prompt[card_info_start:card_info_end].strip()
            return json.dumps({"tool": "process_payment", "args": {"amount": amount, "card_info": card_info}})
        elif "recommend products" in prompt.lower() or "suggest items" in prompt.lower():
            user_id_start = prompt.lower().find("user id:") + len("user id:")
            user_id_end = prompt.lower().find("\noutput format:")
            user_id = prompt[user_id_start:user_id_end].strip()
            return json.dumps({"tool": "recommend_products", "args": {"user_id": user_id}})
        else:
            return json.dumps({"tool": None, "args": {}})

class PromptEngineeringModule:
    def __init__(self, tool_registry: ECommerceToolRegistry):
        self.tool_registry = tool_registry
        self.tool_descriptions = self._get_tool_descriptions()

    def _get_tool_descriptions(self):
        descriptions = []
        for tool_name in dir(self.tool_registry):
            if not tool_name.startswith('_'):
                tool_func = getattr(self.tool_registry, tool_name)
                if callable(tool_func):
                    sig = tool_func.__annotations__
                    params = {k: v.__name__ for k, v in sig.items() if k != 'return'}
                    descriptions.append({"name": tool_name, "description": f"Function to {tool_name.replace('_', ' ')}.", "parameters": params})
        return descriptions

    def construct_prompt(self, user_query: str):
        prompt_parts = []
        prompt_parts.append("You are an AI assistant for an e-commerce platform. Your task is to select the most appropriate tool to fulfill the user's request and extract the necessary arguments. If no tool is suitable, respond with '{{\"tool\": null, \"args\": {{}}}}'.\n\n")
        prompt_parts.append("Available Tools:\n")
        for tool in self.tool_descriptions:
            prompt_parts.append(f"- Name: {tool['name']}\n")
            prompt_parts.append(f"  Description: {tool['description']}\n")
            prompt_parts.append(f"  Parameters: {tool['parameters']}\n")
        prompt_parts.append(f"\nUser Query: {user_query}\n\n")
        prompt_parts.append("Output Format: JSON string with 'tool' (string or null) and 'args' (dictionary of parameter names and values).\nExample: {{\"tool\": \"product_search\", \"args\": {{\"query\": \"laptops\"}}}}\n\n")
        prompt_parts.append("Your Response:")
        return "".join(prompt_parts)

class LLMOrchestrator:
    def __init__(self, mock_llm: MockLLM, tool_registry: ECommerceToolRegistry):
        self.prompt_engineer = PromptEngineeringModule(tool_registry)
        self.mock_llm = mock_llm

    def process_query(self, user_query: str):
        prompt = self.prompt_engineer.construct_prompt(user_query)
        llm_response_str = self.mock_llm.predict(prompt)
        try:
            llm_response = json.loads(llm_response_str)
            return llm_response.get("tool"), llm_response.get("args", {})
        except json.JSONDecodeError:
            return None, {}

class ToolExecutionEngine:
    def __init__(self, tool_registry: ECommerceToolRegistry):
        self.tool_registry = tool_registry

    def execute_tool(self, tool_name: str, args: dict):
        if tool_name and hasattr(self.tool_registry, tool_name):
            tool_func = getattr(self.tool_registry, tool_name)
            try:
                result = tool_func(**args)
                return result
            except TypeError as e:
                return {"status": "error", "message": f"Error executing tool {tool_name}: {e}"}
        return {"status": "error", "message": "Tool not found or invalid tool call."}

class ResponseGenerationLayer:
    def generate_response(self, user_query: str, tool_output: dict):
        if tool_output.get("status") == "success":
            if "product_search" in user_query.lower() and tool_output.get("data"):
                products = ", ".join([p['name'] for p in tool_output['data']])
                return f"Here are some products matching your search: {products}."
            elif "order_management" in tool_output.get("data", {}) and tool_output['data'].get('order_id'):
                return f"Order {tool_output['data']['order_id']} status: {tool_output['data']['status']}. Total: ${tool_output['data']['total']:.2f}."
            elif "ticket_id" in tool_output.get("data", {}):
                return f"Customer support ticket '{tool_output['data']['ticket_id']}' for issue '{tool_output['data']['issue']}' has been opened. We will get back to you shortly."
            elif "transaction_id" in tool_output.get("data", {}):
                return f"Payment of ${tool_output['data']['amount']:.2f} processed successfully with transaction ID: {tool_output['data']['transaction_id']}."
            elif "recommend_products" in user_query.lower() and tool_output.get("data"):
                 products = ", ".join([p['name'] for p in tool_output['data']])
                 return f"Here are some recommendations for you: {products}."
            else:
                return f"Operation successful: {tool_output['data']}"
        elif tool_output.get("status") == "error":
            return f"An error occurred: {tool_output['message']}"
        else:
            return "I'm sorry, I couldn't process your request. Could you please rephrase it?"

class ECommerceAIAssistant:
    def __init__(self):
        self.tool_registry = ECommerceToolRegistry()
        self.mock_llm = MockLLM()
        self.orchestrator = LLMOrchestrator(self.mock_llm, self.tool_registry)
        self.executor = ToolExecutionEngine(self.tool_registry)
        self.response_generator = ResponseGenerationLayer()

    def run(self):
        print("Welcome to the E-commerce AI Assistant! Type 'exit' to quit.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break

            selected_tool, args = self.orchestrator.process_query(user_query)

            if selected_tool:
                tool_output = self.executor.execute_tool(selected_tool, args)
                response = self.response_generator.generate_response(user_query, tool_output)
                print(f"Assistant: {response}")
            else:
                print("Assistant: I couldn't find a suitable tool for your request. Can you be more specific?")

if __name__ == "__main__":
    assistant = ECommerceAIAssistant()
    assistant.run()