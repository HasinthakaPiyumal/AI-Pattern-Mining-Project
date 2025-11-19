import os
from typing import List, Dict, Any

# Placeholder for LangChain imports. Assuming OpenAI models for simplicity.
# If you don't have an OpenAI API key, this code will run with a mock LLM.
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    from langchain_core.messages import HumanMessage, SystemMessage
    from dotenv import load_dotenv
    load_dotenv()
    HAS_LANGCHAIN = True
except ImportError:
    print("LangChain or langchain_openai not installed. Running with mock LLM. Please install 'langchain-openai' and 'python-dotenv' for full functionality.")
    HAS_LANGCHAIN = False

# --- Mock LLM for local testing without API key or LangChain ---
class MockLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages: List[Any], *args, **kwargs) -> HumanMessage:
        last_user_message = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
            elif isinstance(msg, dict) and msg.get('type') == 'human': # For older LangChain versions
                last_user_message = msg['content']
        
        if "hello" in last_user_message.lower():
            response_content = "Hello! How can I assist you with your e-commerce queries today?"
        elif "order status" in last_user_message.lower():
            order_id = last_user_message.split(' ')[-1] # very basic parsing
            response_content = f"Let me check the status for order {order_id}. Please allow a moment."
        elif "return" in last_user_message.lower():
            response_content = "For returns, please visit our returns portal on our website. Would you like a link?"
        elif "complex query" in last_user_message.lower():
            response_content = "This is a complex query. I need to break it down. Let me think step by step: [Step 1] [Step 2] [Final Answer]."
        else:
            response_content = f"I received your message: '{last_user_message}'. How else can I help?"
        
        return HumanMessage(content=response_content)

# --- Ethical Guidelines and E-commerce Data Simulation ---
ETHICAL_GUIDELINES = """
As an AI customer support agent, you must adhere to the following principles:
1.  Always be helpful, polite, and respectful.
2.  Provide accurate and truthful information only.
3.  Never disclose sensitive customer information.
4.  Avoid making assumptions; ask for clarification if needed.
5.  Be impartial and fair in all interactions, avoiding any biases.
6.  If you cannot directly assist, guide the user to the correct resource.
7.  Do not generate harmful, discriminatory, or inappropriate content.
"""

# Simulated E-commerce Database
ECOMMERCE_DATA = {
    "customers": {
        "123": {"name": "Alice Smith", "email": "alice@example.com"},
        "456": {"name": "Bob Johnson", "email": "bob@example.com"},
    },
    "orders": {
        "ORD001": {"customer_id": "123", "item": "Laptop Pro", "status": "Shipped", "tracking": "TRK12345"},
        "ORD002": {"customer_id": "456", "item": "Wireless Mouse", "status": "Processing", "tracking": "N/A"},
        "ORD003": {"customer_id": "123", "item": "USB-C Hub", "status": "Delivered", "tracking": "TRK67890"},
    },
    "products": {
        "Laptop Pro": {"price": "$1200", "description": "High-performance laptop"},
        "Wireless Mouse": {"price": "$25", "description": "Ergonomic wireless mouse"},
        "USB-C Hub": {"price": "$50", "description": "Multi-port USB-C adapter"},
    }
}

def get_customer_info(customer_id: str) -> Dict[str, Any]:
    return ECOMMERCE_DATA["customers"].get(customer_id, {})

def get_order_details(order_id: str) -> Dict[str, Any]:
    return ECOMMERCE_DATA["orders"].get(order_id, {})

def get_product_details(product_name: str) -> Dict[str, Any]:
    return ECOMMERCE_DATA["products"].get(product_name, {})

# --- Customer Support Agent Class ---
class CustomerSupportAgent:
    def __init__(self, temperature: float = 0.7):
        if HAS_LANGCHAIN:
            self.llm = ChatOpenAI(temperature=temperature, model="gpt-4o-mini") # Using a cost-effective model
        else:
            self.llm = MockLLM()
            print("Using MockLLM as LangChain is not available. Install 'langchain-openai' and set OPENAI_API_KEY for full functionality.")

        self.base_system_prompt = SystemMessage(content=f"You are an AI assistant for an e-commerce customer support. "
                                                        f"Your goal is to provide helpful, accurate, and polite responses. "
                                                        f"Always follow these ethical guidelines:\n{ETHICAL_GUIDELINES}")

    def _format_messages(self, system_content: str, user_query: str, few_shot_examples: List[Dict[str, str]] = None) -> List[Any]:
        messages = [SystemMessage(content=system_content)]
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append(HumanMessage(content=example["input"]))
                messages.append(SystemMessage(content=example["output"]))
        messages.append(HumanMessage(content=user_query))
        return messages

    def _invoke_llm_with_retries(self, messages: List[Any], max_retries=3) -> str:
        if not HAS_LANGCHAIN:
            return self.llm.invoke(messages).content
            
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                print(f"LLM invocation failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    return "I am currently experiencing technical difficulties. Please try again later."

    def generate_response_zero_shot(self, query: str) -> str:
        messages = self._format_messages(self.base_system_prompt.content, query)
        return self._invoke_llm_with_retries(messages)

    def generate_response_few_shot(self, query: str, examples: List[Dict[str, str]]) -> str:
        system_content = self.base_system_prompt.content + "\nHere are some examples of how to respond:"
        messages = self._format_messages(system_content, query, few_shot_examples=examples)
        return self._invoke_llm_with_retries(messages)

    def generate_response_template_role_style(self, query: str, role: str = "expert customer service agent", style: str = "friendly and professional") -> str:
        system_content = f"You are an {role} for an e-commerce platform. Your responses should be {style} in tone. " \
                         f"Always follow these ethical guidelines:\n{ETHICAL_GUIDELINES}"
        
        template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_content),
            HumanMessagePromptTemplate.from_template("Customer query: {customer_query}\nAgent response:")
        ])
        
        chain = LLMChain(llm=self.llm, prompt=template)
        
        if not HAS_LANGCHAIN:
            # Mock LLM doesn't use chains, simulate direct invoke
            messages = [SystemMessage(content=system_content), HumanMessage(content=f"Customer query: {query}\nAgent response:")]
            return self._invoke_llm_with_retries(messages)

        return chain.invoke({"customer_query": query})["text"]

    def resolve_complex_query_chain(self, query: str) -> str:
        # Example of a multi-step prompt chain for complex reasoning (e.g., rephrase and respond, metacognitive)
        # Step 1: Understand and rephrase the query
        rephrase_template = ChatPromptTemplate.from_messages([
            self.base_system_prompt,
            HumanMessagePromptTemplate.from_template(
                "The customer has a complex query: '{query}'. First, rephrase the query to ensure full understanding. "
                "Then, break down the problem into smaller, manageable steps."
                "Rephrased Query and Steps:"
            )
        ])
        rephrase_chain = LLMChain(llm=self.llm, prompt=rephrase_template, output_key="rephrased_query_and_steps")

        # Step 2: Plan and gather information (simulated data lookup)
        plan_template = ChatPromptTemplate.from_messages([
            self.base_system_prompt,
            HumanMessagePromptTemplate.from_template(
                "Based on the rephrased query and steps: {rephrased_query_and_steps}, "
                "identify any necessary information from e-commerce data (e.g., customer, order, product details). "
                "If an order ID or customer ID is mentioned, simulate looking it up using our internal functions. "
                "Then, create a plan to address the query, explicitly stating what information is needed and where to get it."
                "Information Needed and Plan:"
            )
        ])
        plan_chain = LLMChain(llm=self.llm, prompt=plan_template, output_key="information_and_plan")

        # Step 3: Execute plan and generate final response
        execute_template = ChatPromptTemplate.from_messages([
            self.base_system_prompt,
            HumanMessagePromptTemplate.from_template(
                "Based on the information gathered and the plan: {information_and_plan}, "
                "and the original query: '{query}', generate a comprehensive and helpful final response to the customer. "
                "Ensure the response is polite, professional, and directly addresses all parts of the query."
                "Final Response:"
            )
        ])
        execute_chain = LLMChain(llm=self.llm, prompt=execute_template, output_key="final_response")

        overall_chain = SequentialChain(
            chains=[rephrase_chain, plan_chain, execute_chain],
            input_variables=["query"],
            output_variables=["rephrased_query_and_steps", "information_and_plan", "final_response"],
            verbose=True # Set to True to see intermediate steps
        )

        if not HAS_LANGCHAIN:
            # Mock LLM doesn't support sequential chains directly, simulate a combined response
            rephrase_mock = f"Rephrased: Customer wants help with a complex issue regarding '{query}'. Steps: 1. Understand. 2. Plan. 3. Respond."
            plan_mock = f"Information needed: Potentially order ID, product name. Plan: Combine facts and generate a helpful response."
            final_mock = f"Thank you for your complex query regarding '{query}'. After careful consideration, here is our detailed response based on your needs.\n{rephrase_mock}\n{plan_mock}\n[Simulated detailed answer]."
            return final_mock

        try:
            response = overall_chain.invoke({"query": query})
            return response["final_response"]
        except Exception as e:
            print(f"Error during complex query chain: {e}")
            return "I apologize, but I encountered an error while processing your complex query. Please try rephrasing it or contact support directly."

    def get_response(self, query: str, prompt_strategy: str = "ZeroShot", role: str = "customer service agent", style: str = "friendly and professional") -> str:
        if prompt_strategy == "ZeroShot":
            return self.generate_response_zero_shot(query)
        elif prompt_strategy == "FewShot":
            # Example few-shot examples (these would ideally be dynamically selected based on query type)
            examples = [
                {"input": "My order 123 is delayed.", "output": "I understand your concern about order 123. Let me check its status for you. Could you confirm the full order number or your email?"},
                {"input": "How do I return a product?", "output": "To initiate a return, please visit our returns portal at [link]. You'll need your order number and email address."}
            ]
            return self.generate_response_few_shot(query, examples)
        elif prompt_strategy == "Template_Role_Style":
            return self.generate_response_template_role_style(query, role=role, style=style)
        elif prompt_strategy == "ComplexQueryChain":
            return self.resolve_complex_query_chain(query)
        else:
            return self.generate_response_zero_shot(query)

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    agent = CustomerSupportAgent()

    print("\n--- ZeroShot Example ---")
    print(agent.get_response("What is the status of my order ORD001?"))

    print("\n--- FewShot Example ---")
    print(agent.get_response("I need help with a refund.", prompt_strategy="FewShot"))

    print("\n--- Template, Role, Style Example (Formal Analyst) ---")
    print(agent.get_response("I have a technical question about the Laptop Pro.", prompt_strategy="Template_Role_Style", role="technical support analyst", style="concise and highly informative"))

    print("\n--- Complex Query Chain Example ---")
    complex_query = "I want to understand why my order ORD001, which contains a Laptop Pro, was shipped to the wrong address, and what steps I need to take to get it resent to the correct one (my account address)."
    print(agent.get_response(complex_query, prompt_strategy="ComplexQueryChain"))
