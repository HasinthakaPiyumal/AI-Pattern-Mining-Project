import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Simulated Knowledge Base
KNOWLEDGE_BASE = {
    "shipping": "Standard shipping takes 5-7 business days. Expedited shipping is available for an extra charge and takes 2-3 business days.",
    "returns": "You can return any item within 30 days of purchase, provided it is in its original condition. Please visit our returns page for more details and to initiate a return.",
    "account password reset": "To reset your account password, please go to the login page and click on 'Forgot Password'. Follow the instructions sent to your registered email address.",
    "product warranty": "All our electronics come with a 1-year manufacturer's warranty. For specific product warranty details, please refer to the product description page or contact support.",
    "payment methods": "We accept major credit cards (Visa, Mastercard, American Express), PayPal, and Apple Pay."
}

class InformationGatheringAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        self.extraction_template = PromptTemplate.from_template(
            """You are an information extraction agent. Your goal is to understand a customer's query, identify key topics or keywords, and use them to find relevant information from a knowledge base.
            
            Customer Query: {query}
            
            Based on the query, identify 2-3 key topics/keywords that would be useful for searching a knowledge base. Return them as a comma-separated list.
            Keywords:"""
        )
        self.summarization_template = PromptTemplate.from_template(
            """You are a summarization agent. Consolidate the customer's original query and the retrieved information into a concise summary.
            
            Customer Query: {original_query}
            Retrieved Information: {retrieved_info}
            
            Provide a structured summary of the customer's problem and any relevant information found. Highlight the core issue.
            Summary:"""
        )

    def _search_knowledge_base(self, keywords: list) -> str:
        found_info = []
        for keyword in keywords:
            for kb_key, kb_value in KNOWLEDGE_BASE.items():
                if keyword.lower() in kb_key.lower():
                    found_info.append(f"'{kb_key}': {kb_value}")
        return "\n".join(found_info) if found_info else "No relevant information found in the knowledge base."

    def gather_information(self, query: str) -> str:
        # Step 1: Extract keywords
        extraction_chain = self.extraction_template | self.llm
        raw_keywords_response = extraction_chain.invoke({"query": query})
        keywords = [k.strip() for k in raw_keywords_response.content.split(',') if k.strip()]
        
        # Step 2: Search knowledge base
        retrieved_info = self._search_knowledge_base(keywords)
        
        # Step 3: Summarize
        summarization_chain = self.summarization_template | self.llm
        summary_response = summarization_chain.invoke({"original_query": query, "retrieved_info": retrieved_info})
        
        return summary_response.content

class ResponsePlanningAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        self.response_template = PromptTemplate.from_template(
            """You are an empathetic and helpful customer support agent. Based on the provided summary of the customer's problem and relevant information, formulate a comprehensive and polite response.
            
            Summary of Customer Problem and Information: {summary}
            
            Craft a response that addresses the customer's issue clearly and empathetically. Start with a polite greeting.
            Customer Response:"""
        )

    def generate_response(self, summary: str) -> str:
        response_chain = self.response_template | self.llm
        final_response = response_chain.invoke({"summary": summary})
        return final_response.content

class CustomerSupportSystem:
    def __init__(self, openai_api_key: str):
        self.info_gathering_agent = InformationGatheringAgent(openai_api_key)
        self.response_planning_agent = ResponsePlanningAgent(openai_api_key)

    def handle_customer_query(self, query: str) -> str:
        # Phase 1: Information Gathering
        print("\n--- Phase 1: Information Gathering ---")
        summary = self.info_gathering_agent.gather_information(query)
        print(f"Generated Summary:\n{summary}")
        
        # Phase 2: Response Planning and Generation
        print("\n--- Phase 2: Response Planning and Generation ---")
        final_response = self.response_planning_agent.generate_response(summary)
        print(f"Generated Final Response:\n{final_response}")
        
        return final_response

if __name__ == "__main__":
    # Ensure your OpenAI API key is set as an environment variable
    # Example: os.environ["OPENAI_API_KEY"] = "your_api_key_here"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")
    else:
        support_system = CustomerSupportSystem(openai_api_key)
        
        print("\n--- Customer Support System Initialized ---")

        customer_query_1 = "I need to return a shirt I bought last week. How do I do that?"
        print(f"\nCustomer Query: {customer_query_1}")
        support_system.handle_customer_query(customer_query_1)

        customer_query_2 = "What are the shipping options available for my order?"
        print(f"\nCustomer Query: {customer_query_2}")
        support_system.handle_customer_query(customer_query_2)
        
        customer_query_3 = "My account is locked, and I can't remember my password. Can you help me?"
        print(f"\nCustomer Query: {customer_query_3}")
        support_system.handle_customer_query(customer_query_3)

        customer_query_4 = "I bought a new headphone, does it have a warranty and what payment methods do you accept?"
        print(f"\nCustomer Query: {customer_query_4}")
        support_system.handle_customer_query(customer_query_4)

        customer_query_5 = "I have a very generic question that might not be in the KB."
        print(f"\nCustomer Query: {customer_query_5}")
        support_system.handle_customer_query(customer_query_5)
