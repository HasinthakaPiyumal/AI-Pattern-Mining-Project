# customer_support_agent.py

import os
from collections import namedtuple

# --- Simulate Langchain and Chroma components ---
# In a real scenario, you would install these and use them directly.
# For this self-contained example, we'll create lightweight mockups.

# Mock LLM and Embeddings
class MockChatOpenAI:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=None):
        self.model_name = model_name
        self.temperature = temperature
        self.openai_api_key = openai_api_key

    def invoke(self, prompt):
        if "plan a solution" in prompt.lower() and "draft a response" in prompt.lower():
            if "order_number: ORD123" in prompt:
                return "The customer's order ORD123 seems to be delayed. The plan is to apologize, confirm the delay, and offer a refund or re-shipment. Draft response: 'We apologize for the delay with your order ORD123. It's currently experiencing an unexpected hold-up. Would you prefer a full refund or for us to reship your order immediately?'"
            elif "product_issue: broken screen" in prompt:
                return "The customer has a broken screen for product XYZ. The plan is to offer troubleshooting, repair options, or a replacement. Draft response: 'We're sorry to hear about your broken screen for product XYZ. We can offer some troubleshooting steps, arrange a repair, or process a replacement. Which option would you prefer?'"
            elif "billing_issue: incorrect charge" in prompt:
                return "The customer reports an incorrect charge. The plan is to investigate the charge and process a correction. Draft response: 'We understand you're concerned about an incorrect charge. We're investigating this immediately and will correct it. Please allow 1-2 business days for this to reflect.' "
            else:
                return "Based on the collected information, a general solution plan is to acknowledge the customer's query and provide a polite, helpful next step. Draft response: 'Thank you for reaching out. We are processing your request and will get back to you with a detailed solution shortly.'"
        return "Simulated LLM response for: " + prompt


class MockOpenAIEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] * 1536 for _ in texts]

    def embed_query(self, query):
        return [0.5] * 1536


Document = namedtuple("Document", ["page_content", "metadata"])

class MockChroma:
    def __init__(self):
        self.documents = []
        self.collection = {}

    def add_documents(self, docs):
        for doc in docs:
            self.documents.append(doc)
            for keyword in doc.page_content.lower().split():
                self.collection.setdefault(keyword, []).append(doc)

    def as_retriever(self, search_kwargs={"k": 2}):
        class Retriever:
            def get_relevant_documents(self, query):
                relevant_docs = []
                query_keywords = query.lower().split()
                for doc in self.documents:
                    if any(kw in doc.page_content.lower() for kw in query_keywords):
                        relevant_docs.append(doc)
                return relevant_docs[:search_kwargs["k"]]
        return Retriever()


class InformationCollectionModule:
    def __init__(self):
        self.intents = {
            "order_status": ["order", "status", "where is", "delivery", "track"],
            "product_issue": ["problem", "issue", "broken", "faulty", "not working"],
            "billing_issue": ["charge", "bill", "invoice", "incorrect", "dispute"],
            "general_inquiry": ["help", "question", "support", "contact"]
        }
        self.entities = {
            "order_number": r"(ORD[0-9]{3})",
            "product_name": r"(product [A-Z]{3})",
            "issue_type": r"(broken screen|payment error|delayed shipping)"
        }

        self.knowledge_base = MockChroma()
        self._load_mock_knowledge_base()

    def _load_mock_knowledge_base(self):
        docs = [
            Document(page_content="Our shipping policy states that standard delivery takes 5-7 business days. You can track your order using the link provided in your shipping confirmation email.", metadata={"source": "FAQ"}),
            Document(page_content="For product returns, please ensure the item is in its original packaging and contact support within 30 days of purchase.", metadata={"source": "Policy"}),
            Document(page_content="Troubleshooting steps for common software issues often involve restarting your device and checking for updates.", metadata={"source": "Troubleshooting Guide"}),
            Document(page_content="To dispute an incorrect charge, please provide your account details and the transaction ID. We will investigate within 2 business days.", metadata={"source": "Billing FAQ"}),
            Document(page_content="Product XYZ has a 1-year warranty covering manufacturing defects. Physical damage, like a broken screen, is not covered unless purchased with an extended warranty.", metadata={"source": "Product Manual"})
        ]
        self.knowledge_base.add_documents(docs)
        self.kb_retriever = self.knowledge_base.as_retriever()

    def _nlu_process(self, query):
        detected_intent = "general_inquiry"
        detected_entities = {}

        query_lower = query.lower()
        for intent, keywords in self.intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intent = intent
                break

        import re
        for entity_name, pattern in self.entities.items():
            match = re.search(pattern, query)
            if match:
                detected_entities[entity_name] = match.group(0)

        return {"intent": detected_intent, "entities": detected_entities}

    def _crm_data_fetcher(self, customer_id=None, order_number=None):
        crm_data_store = {
            "ORD123": {"customer_id": "CUST001", "status": "Delayed", "last_interaction": "2023-10-26"},
            "CUST001": {"name": "Alice Smith", "email": "alice@example.com", "membership": "Gold"}
        }
        data = {}
        if order_number and order_number in crm_data_store:
            data.update(crm_data_store[order_number])
            if crm_data_store[order_number]["customer_id"] in crm_data_store:
                data.update(crm_data_store[crm_data_store[order_number]["customer_id"]])
        elif customer_id and customer_id in crm_data_store:
            data.update(crm_data_store[customer_id])
        return data

    def _external_api_integrator(self, query_params):
        if "track_order" in query_params:
            order_id = query_params.get("order_id")
            if order_id == "ORD123":
                return {"order_id": "ORD123", "shipping_status": "In Transit - Expected Delay", "last_update": "2023-10-27"}
            else:
                return {"error": "Order not found"}
        return {}

    def collect_information(self, query):
        nlu_output = self._nlu_process(query)
        intent = nlu_output["intent"]
        entities = nlu_output["entities"]

        retrieved_docs = self.kb_retriever.get_relevant_documents(query)
        knowledge_base_info = "\n".join([doc.page_content for doc in retrieved_docs])

        crm_info = {}
        if "order_number" in entities:
            crm_info = self._crm_data_fetcher(order_number=entities["order_number"])
        elif "customer_id" in entities:
            crm_info = self._crm_data_fetcher(customer_id=entities["customer_id"])

        external_api_info = {}
        if intent == "order_status" and "order_number" in entities:
            external_api_info = self._external_api_integrator({"track_order": True, "order_id": entities["order_number"]})

        collected_data = {
            "original_query": query,
            "intent": intent,
            "entities": entities,
            "knowledge_base_info": knowledge_base_info,
            "crm_info": crm_info,
            "external_api_info": external_api_info
        }
        return collected_data


class SolutionPlanningAndResponseGenerationModule:
    def __init__(self):
        self.llm = MockChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def _cognitive_planner(self, collected_data):
        prompt_template = """
You        You are an intelligent customer support agent. Your task is to analyze the provided customer query and collected information, then plan a solution and draft a concise, helpful response.

        Customer Query: {original_query}

        Detected Intent: {intent}
        Detected Entities: {entities}

        Knowledge Base Information:
        {knowledge_base_info}

        CRM Data:
        {crm_info}

        External API Data:
        {external_api_info}

        Based on the above information, first, outline a solution plan. Then, draft a customer response adhering to the following constraints:
        - Be polite and empathetic.
        - Be clear and to the point.
        - Address the customer's specific issue.
        - Suggest concrete next steps or resolutions.
        - Ensure the tone is professional.

        Solution Plan:
        1.

        Draft Response:
        """
        
        crm_str = ", ".join([f"{k}: {v}" for k,v in collected_data["crm_info"].items()])
        external_api_str = ", ".join([f"{k}: {v}" for k,v in collected_data["external_api_info"].items()])

        formatted_prompt = prompt_template.format(
            original_query=collected_data["original_query"],
            intent=collected_data["intent"],
            entities=collected_data["entities"],
            knowledge_base_info=collected_data["knowledge_base_info"] if collected_data["knowledge_base_info"] else "No specific knowledge base information found.",
            crm_info=crm_str if crm_str else "No specific CRM data found.",
            external_api_info=external_api_str if external_api_str else "No specific external API data found."
        )
        
        llm_output = self.llm.invoke(formatted_prompt)
        return llm_output

    def _constraint_adherence_and_refinement(self, draft_response):
        refined_response = draft_response
        if not refined_response.strip().startswith("We apologize") and "delay" in refined_response.lower():
             refined_response = "We apologize for any inconvenience. " + refined_response

        if "solution plan:" in refined_response.lower():
            parts = refined_response.lower().split("draft response:")
            if len(parts) > 1:
                refined_response = parts[1].strip()

        if "please allow" not in refined_response.lower() and ("investigating" in refined_response.lower() or "processing" in refined_response.lower()):
            refined_response += " Please allow 1-2 business days for us to resolve this."

        return refined_response

    def _response_formatter(self, final_response):
        return f"--- Final Customer Response ---\n{final_response}\n------------------------------"

    def generate_response(self, collected_data):
        llm_output = self._cognitive_planner(collected_data)
        
        if "Draft Response:" in llm_output:
            draft_response = llm_output.split("Draft Response:", 1)[1].strip()
        else:
            draft_response = llm_output

        refined_response = self._constraint_adherence_and_refinement(draft_response)
        final_formatted_response = self._response_formatter(refined_response)
        return final_formatted_response


class CustomerSupportAgent:
    def __init__(self):
        self.info_collector = InformationCollectionModule()
        self.solution_generator = SolutionPlanningAndResponseGenerationModule()

    def handle_query(self, query):
        print(f"Agent received query: '{query}'")
        collected_data = self.info_collector.collect_information(query)
        print("\n--- Collected Information ---")
        for k, v in collected_data.items():
            if isinstance(v, dict):
                print(f"{k}: {', '.join([f'{sk}:{sv}' for sk,sv in v.items()])}")
            else:
                print(f"{k}: {v}")
        print("-----------------------------\n")

        final_response = self.solution_generator.generate_response(collected_data)
        print(final_response)
        return final_response

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    queries = [
        "My order ORD123 hasn't arrived yet, what's the status?",
        "I have a problem with product XYZ, the screen is broken.",
        "There's an incorrect charge on my last bill.",
        "I need help with my account."
    ]

    for q in queries:
        agent.handle_query(q)
        print("\n" + "="*80 + "\n")
