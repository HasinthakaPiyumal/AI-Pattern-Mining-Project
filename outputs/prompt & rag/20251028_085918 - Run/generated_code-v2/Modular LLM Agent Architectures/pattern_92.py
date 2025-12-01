import os
from openai import OpenAI

class CustomerSupportAgent:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.customer_context = {}
        self.knowledge_base = {
            "product_issue": "Please try restarting your device and checking your internet connection. If the issue persists, contact technical support.",
            "billing_query": "You can view your latest bill and payment history in your account dashboard. For specific discrepancies, please provide your account number.",
            "order_status": "Please provide your order ID to check the status of your recent purchase.",
            "return_policy": "Our return policy allows returns within 30 days of purchase with the original receipt. Some exclusions apply."
        }

    def _get_crm_data(self, customer_id=None):
        if customer_id:
            return {"customer_id": customer_id, "name": "John Doe", "past_issues": ["billing_query", "product_issue"], "current_plan": "Premium"}
        return {}

    def _query_knowledge_base(self, topic):
        return self.knowledge_base.get(topic, "I couldn't find specific information on that topic. Can you provide more details?")

    def _llm_call(self, prompt, model="gpt-3.5-turbo", max_tokens=150, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM error: {e}"

    def stage_1_information_collection(self, query, customer_id=None):
        print("\n--- Stage 1: Information Collection & Understanding ---")
        self.customer_context = {"original_query": query}

        # 1. Intent Recognition and Entity Extraction
        intent_prompt = f"Analyze the following customer query and identify the primary intent and any key entities (e.g., product name, order ID, customer ID). Respond as 'Intent: <intent>, Entities: <entity1>, <entity2>'. Query: {query}"
        intent_response = self._llm_call(intent_prompt)
        print(f"LLM Intent/Entity Extraction: {intent_response}")

        # Parse intent and entities (simplified)
        self.customer_context["parsed_intent"] = "unknown"
        self.customer_context["extracted_entities"] = {}
        if "Intent:" in intent_response:
            parts = intent_response.split("Intent:", 1)[1].split(", Entities:", 1)
            self.customer_context["parsed_intent"] = parts[0].strip().lower().replace(' ', '_')
            if len(parts) > 1:
                entity_str = parts[1].strip()
                for entity_pair in entity_str.split(', '):
                    if ': ' in entity_pair:
                        k, v = entity_pair.split(': ', 1)
                        self.customer_context["extracted_entities"][k.strip().lower()] = v.strip()
                    else:
                        # Handle simple comma separated entities if no key-value given
                        self.customer_context["extracted_entities"][entity_pair.strip().lower()] = entity_pair.strip()

        # 2. Simulate CRM data retrieval
        if customer_id:
            crm_data = self._get_crm_data(customer_id)
            self.customer_context["crm_data"] = crm_data
            print(f"CRM Data Retrieved: {crm_data}")
        else:
            print("No customer ID provided for CRM data retrieval.")

        # 3. Formulate clarifying questions if needed (simplified heuristic)
        clarifying_questions = []
        if self.customer_context["parsed_intent"] == "unknown" or not self.customer_context["extracted_entities"]:
            clarifying_questions_prompt = f"Based on the query '{query}' and parsed intent '{self.customer_context['parsed_intent']}', what clarifying questions should I ask the customer to understand their issue better? List up to 2 questions, separated by newlines."
            questions = self._llm_call(clarifying_questions_prompt, max_tokens=100)
            clarifying_questions = [q.strip() for q in questions.split('\n') if q.strip()]

        self.customer_context["clarifying_questions"] = clarifying_questions
        print(f"Current Customer Context: {self.customer_context}")
        return self.customer_context

    def stage_2_planning_and_response_generation(self):
        print("\n--- Stage 2: Planning & Response Generation ---")
        context = self.customer_context
        final_response = ""
        resolution_plan = []

        # If clarifying questions exist, ask them instead of resolving
        if context.get("clarifying_questions") and context["clarifying_questions"]:
            print("Clarifying questions detected. Cannot resolve yet. Please provide more information.")
            return "Please provide more information. " + " ".join(context["clarifying_questions"])

        intent = context.get("parsed_intent", "unknown")
        entities = context.get("extracted_entities", {})

        # 1. Formulate a plan based on intent and entities
        plan_prompt = f"Given the customer's intent '{intent}' and entities {entities}, outline a step-by-step resolution plan. Include actions like 'consult knowledge base for <topic>', 'check system status', 'suggest solution'."
        plan_response = self._llm_call(plan_prompt, max_tokens=200)
        resolution_plan = [step.strip() for step in plan_response.split('\n') if step.strip()]
        print(f"Generated Resolution Plan: {resolution_plan}")

        # 2. Execute plan steps (simplified)
        response_parts = []
        for step in resolution_plan:
            if "consult knowledge base for" in step.lower():
                topic = step.lower().split("consult knowledge base for ", 1)[1].replace('.', '').strip()
                kb_info = self._query_knowledge_base(topic.replace(' ', '_')) # Adjust for KB keys
                response_parts.append(f"Based on {topic}: {kb_info}")
            elif "check system status" in step.lower():
                response_parts.append("System status is currently nominal.")
            elif "suggest solution" in step.lower() or "recommend" in step.lower():
                # Use LLM to refine the suggestion based on all context
                solution_prompt = f"Based on the original query '{context['original_query']}', the parsed intent '{intent}', entities {entities}, and any collected CRM data {context.get('crm_data', {})}, provide a concise and helpful solution or recommendation. Focus on addressing the core issue."
                solution = self._llm_call(solution_prompt, max_tokens=150)
                response_parts.append(f"Recommendation: {solution}")
            else:
                response_parts.append(step)

        # 3. Generate final coherent response
        final_response_prompt = f"Combine the following information and resolution plan into a polite and comprehensive customer support response: Original query: {context['original_query']}. Resolved intent: {intent}. Entities: {entities}. CRM data: {context.get('crm_data', {})}. Resolution steps taken: {'; '.join(resolution_plan)}. Information gathered: {'; '.join(response_parts)}. Final response should directly address the customer's query."
        final_response = self._llm_call(final_response_prompt, max_tokens=300)

        print(f"Final Agent Response: {final_response}")
        return final_response

    def handle_customer_query(self, query, customer_id=None):
        print(f"\n--- New Customer Query: {query} (Customer ID: {customer_id}) ---")
        self.stage_1_information_collection(query, customer_id)
        response = self.stage_2_planning_and_response_generation()
        return response

if __name__ == "__main__":
    # Replace with your actual OpenAI API key or set as environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("WARNING: Please set your OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY' with your actual key.")

    agent = CustomerSupportAgent(OPENAI_API_KEY)

    # Example 1: Simple query
    agent.handle_customer_query("My internet is not working.")

    # Example 2: Query with customer ID and known issue
    agent.handle_customer_query("I have a question about my last bill. My customer ID is 12345.", "12345")

    # Example 3: Query needing more clarification
    agent.handle_customer_query("I have an issue with my recent purchase.", "12345")

    # Example 4: Query about returns
    agent.handle_customer_query("What is your return policy?")
