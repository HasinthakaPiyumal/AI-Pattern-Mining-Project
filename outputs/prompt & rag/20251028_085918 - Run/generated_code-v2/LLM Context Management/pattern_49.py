import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

class QueryClassifier:
    def classify(self, query: str) -> str:
        query_lower = query.lower()
        if "order status" in query_lower or \
           "track my order" in query_lower or \
           "shipping time" in query_lower:
            return "straightforward"
        elif "return item" in query_lower or \
             "refund process" in query_lower or \
             "exchange product" in query_lower:
            return "moderate"
        elif "damaged" in query_lower or \
             "wrong item" in query_lower or \
             "multiple issues" in query_lower or \
             "escalate" in query_lower or \
             "urgent" in query_lower:
            return "complex"
        return "moderate" # Default for unclassified queries

class KnowledgeBase:
    def __init__(self):
        self.documents = {
            "order_status": "To check your order status, please visit our 'Order Tracking' page and enter your order number. Orders typically ship within 1-2 business days.",
            "return_policy": "Our return policy allows returns within 30 days of purchase. Items must be in their original condition. To initiate a return, visit our 'Returns Portal' on the website and follow the instructions.",
            "refund_process": "Refunds are processed within 5-7 business days after the returned item is received and inspected. The refund will be credited to your original payment method.",
            "shipping_times": "Standard shipping takes 5-7 business days. Expedited shipping options are available at checkout and deliver within 2-3 business days.",
            "damaged_item_policy": "If you received a damaged item, please contact our support team immediately with your order number and photos of the damage. We will arrange for a replacement or a full refund."
        }

    def retrieve(self, keywords: list[str]) -> str:
        relevant_info = []
        for keyword in keywords:
            for doc_key, doc_content in self.documents.items():
                if keyword in doc_key or keyword in doc_content.lower():
                    relevant_info.append(doc_content)
        return "\n".join(list(set(relevant_info))) if relevant_info else "No specific information found."

class LLMOrchestrator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

    def generate_straightforward_response(self, query: str) -> str:
        prompt_template = PromptTemplate.from_template(
            "You are a helpful e-commerce customer support assistant. Provide a concise and direct answer to the following query:\nQuery: {query}\nAnswer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.invoke({"query": query})
        return response["text"]

    def generate_rag_response(self, query: str, context: str) -> str:
        prompt_template = PromptTemplate.from_template(
            "You are an e-commerce customer support assistant. Use the following context to answer the customer's query. If the context does not contain enough information, state that you cannot fully answer and suggest contacting a human.\nContext: {context}\nQuery: {query}\nAnswer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.invoke({"query": query, "context": context})
        return response["text"]

    def handle_complex_query(self, query: str, context: str = "") -> str:
        summary_prompt = PromptTemplate.from_template(
            "Summarize the following complex customer query and any provided context for a human agent. Also, identify key issues and suggest next steps.\nQuery: {query}\nContext: {context}\nSummary for Human Agent:"
        )
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        summary_output = summary_chain.invoke({"query": query, "context": context})
        return f"Your query is complex and requires specialized attention. I have escalated your case to a human agent with the following summary:\n\n{summary_output['text']}\n\nA human agent will contact you shortly."

class AdaptiveStrategyModule:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.knowledge_base = KnowledgeBase()
        self.llm_orchestrator = LLMOrchestrator()

    def process_query(self, query: str) -> str:
        complexity = self.classifier.classify(query)
        print(f"\nQuery: '{query}' - Classified as: {complexity}")

        if complexity == "straightforward":
            return self.llm_orchestrator.generate_straightforward_response(query)
        elif complexity == "moderate":
            keywords = [word for word in query.lower().split() if len(word) > 3]
            context = self.knowledge_base.retrieve(keywords)
            return self.llm_orchestrator.generate_rag_response(query, context)
        elif complexity == "complex":
            keywords = [word for word in query.lower().split() if len(word) > 3]
            context = self.knowledge_base.retrieve(keywords)
            return self.llm_orchestrator.handle_complex_query(query, context)
        else:
            return "I'm sorry, I couldn't process your request. Please try again or contact human support."

if __name__ == "__main__":
    assistant = AdaptiveStrategyModule()

    print("Welcome to the Smart Customer Support Assistant! Type 'exit' to quit.")

    while True:
        user_query = input("\nHow can I help you today? ")
        if user_query.lower() == 'exit':
            break

        response = assistant.process_query(user_query)
        print(f"Assistant: {response}")

