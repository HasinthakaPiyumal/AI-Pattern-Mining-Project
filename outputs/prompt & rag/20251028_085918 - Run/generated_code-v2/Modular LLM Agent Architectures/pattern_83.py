import os
from typing import List, Dict, Any, Optional
import uuid

# --- Pydantic Models (Simplified for single file) ---
class CustomerProfile:
    def __init__(self, customer_id: str, name: str, email: str, vip: bool = False, history: List[str] = None):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.vip = vip
        self.history = history if history is not None else []

class ConversationEntry:
    def __init__(self, speaker: str, text: str, timestamp: str):
        self.speaker = speaker
        self.text = text
        self.timestamp = timestamp

class AgentResponse:
    def __init__(self, response_text: str, sentiment: str, actions_taken: List[str] = None, compliant: bool = True):
        self.response_text = response_text
        self.sentiment = sentiment
        self.actions_taken = actions_taken if actions_taken is not None else []
        self.compliant = compliant


# --- Mock LLM Class (Replace with actual OpenAI/Cohere SDK in production) ---
class MockLLM:
    def __init__(self, model_name: str = "mock-gpt-3.5-turbo"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        if "order status" in prompt.lower():
            return "Your order #12345 is currently being processed and is expected to ship within 2 business days."
        elif "refund policy" in prompt.lower():
            return "Our refund policy allows returns within 30 days of purchase for a full refund, provided the item is unused and in its original packaging."
        elif "technical issue" in prompt.lower():
            return "I understand you're experiencing a technical issue. Please describe it in more detail, and I can either provide troubleshooting steps or connect you to a specialist."
        elif "escalate" in prompt.lower() or "urgent" in prompt.lower():
            return "I'm escalating this to a human agent immediately. Please provide your contact number."
        return f"Hello! I'm a smart support agent. You asked: '{prompt}'. How can I further assist you today?"


# --- Modules ---
class WorkingMemoryModule:
    def __init__(self):
        self.customer_sessions: Dict[str, List[ConversationEntry]] = {}
        self.customer_profiles: Dict[str, CustomerProfile] = {}

    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        return self.customer_profiles.get(customer_id)

    def update_customer_profile(self, profile: CustomerProfile):
        self.customer_profiles[profile.customer_id] = profile

    def get_conversation_history(self, customer_id: str) -> List[ConversationEntry]:
        return self.customer_sessions.get(customer_id, [])

    def add_conversation_entry(self, customer_id: str, entry: ConversationEntry):
        if customer_id not in self.customer_sessions:
            self.customer_sessions[customer_id] = []
        self.customer_sessions[customer_id].append(entry)


class KnowledgeBaseRetrievalModule:
    def __init__(self):
        # In a real scenario, this would initialize a vector DB like Chroma/Pinecone
        # and an embedding model. For this single file, we use a simple dict lookup.
        self.knowledge_base = {
            "shipping": "Standard shipping takes 5-7 business days. Expedited shipping is available for an extra charge and takes 2-3 business days.",
            "returns": "You can return any unused item within 30 days of purchase with your original receipt. Refunds are processed within 5 business days.",
            "contact support": "You can contact our support team via email at support@example.com or call us at 1-800-555-0123. Our lines are open Monday-Friday, 9 AM to 5 PM EST.",
            "warranty": "All our products come with a 1-year limited warranty against manufacturing defects."
        }

    def retrieve_info(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        for keyword, info in self.knowledge_base.items():
            if keyword in query_lower:
                return info
        return None


class SentimentAnalysisModule:
    def __init__(self):
        # In a real scenario, this would load a transformers pipeline.
        # For this example, we use a simple keyword-based sentiment detection.
        pass

    def analyze_sentiment(self, text: str) -> str:
        text_lower = text.lower()
        if "unhappy" in text_lower or "frustrated" in text_lower or "bad" in text_lower or "issue" in text_lower:
            return "negative"
        elif "great" in text_lower or "happy" in text_lower or "good" in text_lower or "excellent" in text_lower:
            return "positive"
        return "neutral"


class PolicyEnforcementModule:
    def __init__(self):
        self.disallowed_phrases = ["guarantee 100% satisfaction", "absolute refund"]
        self.required_disclaimers = {
            "legal": "Disclaimer: Terms and conditions apply. Offers subject to change.",
            "warranty": "Disclaimer: Warranty does not cover misuse or accidental damage."
        }

    def enforce_policy(self, response: str, sentiment: str) -> (str, bool):
        is_compliant = True
        edited_response = response

        # Check for disallowed phrases
        for phrase in self.disallowed_phrases:
            if phrase in edited_response.lower():
                edited_response = edited_response.replace(phrase, "", 1) # Simple replacement
                is_compliant = False

        # Add disclaimers based on context/sentiment (simplified logic)
        if "refund" in edited_response.lower() or "return" in edited_response.lower():
            if self.required_disclaimers["legal"] not in edited_response:
                edited_response += " " + self.required_disclaimers["legal"]
                is_compliant = False # Mark as non-compliant if disclaimer was missing

        if sentiment == "negative" and "escalate" not in edited_response.lower():
            # Encourage escalation if negative sentiment and no escalation mentioned
            edited_response += " Would you like me to connect you with a specialist?"

        return edited_response, is_compliant


class ActionExecutorModule:
    def __init__(self):
        pass

    def create_support_ticket(self, customer_id: str, issue_description: str) -> str:
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        print(f"[ACTION] Creating support ticket {ticket_id} for customer {customer_id}: {issue_description}")
        return ticket_id

    def check_order_status(self, order_id: str) -> str:
        print(f"[ACTION] Checking status for order {order_id}...")
        # Simulate API call
        return f"Order {order_id} is currently in transit, expected delivery on 2023-12-25."

    def schedule_callback(self, customer_id: str, phone_number: str, time: str) -> str:
        print(f"[ACTION] Scheduling callback for {customer_id} at {phone_number} on {time}.")
        return f"Callback scheduled for {customer_id}."

    def execute_action(self, action_name: str, **kwargs) -> Optional[str]:
        if hasattr(self, action_name) and callable(getattr(self, action_name)):
            try:
                return getattr(self, action_name)(**kwargs)
            except TypeError as e:
                print(f"Error executing action {action_name}: {e}")
                return f"Error: Invalid parameters for action {action_name}."
        return None


# --- Orchestrator ---
class SmartCustomerSupportAgent:
    def __init__(self, llm_api_key: Optional[str] = None):
        self.llm = MockLLM() # Use MockLLM for demonstration
        # In a real app:
        # from langchain.chat_models import ChatOpenAI
        # self.llm = ChatOpenAI(temperature=0.7, openai_api_key=llm_api_key) 

        self.working_memory = WorkingMemoryModule()
        self.knowledge_base_retriever = KnowledgeBaseRetrievalModule()
        self.sentiment_analyzer = SentimentAnalysisModule()
        self.policy_enforcer = PolicyEnforcementModule()
        self.action_executor = ActionExecutorModule()

    def _process_llm_response(self, customer_query: str, chat_history: List[ConversationEntry], customer_profile: Optional[CustomerProfile]) -> str:
        # Construct a comprehensive prompt for the LLM
        prompt_parts = []
        if customer_profile:
            prompt_parts.append(f"Customer Profile: Name={customer_profile.name}, Email={customer_profile.email}, VIP={customer_profile.vip}.")
            if customer_profile.history:
                prompt_parts.append(f"Previous interactions: {'; '.join(customer_profile.history)}")
        
        if chat_history:
            prompt_parts.append("Conversation History:")
            for entry in chat_history:
                prompt_parts.append(f"{entry.speaker}: {entry.text}")

        prompt_parts.append(f"Customer Query: {customer_query}")
        prompt_parts.append("Based on the above context, provide a helpful and concise response. Consider if any external tools (like knowledge base lookup or action execution) are needed, and if so, suggest them clearly in your response, e.g., 'KNOWLEDGE_LOOKUP: [topic]', 'ACTION: [action_name]: [param=value]'.")
        
        full_prompt = "\n".join(prompt_parts)
        return self.llm.generate_response(full_prompt)

    def process_customer_query(self, customer_id: str, query: str) -> AgentResponse:
        # 1. Get customer profile and conversation history
        customer_profile = self.working_memory.get_customer_profile(customer_id)
        if not customer_profile:
            # Simulate new customer onboarding
            customer_profile = CustomerProfile(customer_id=customer_id, name=f"Customer {customer_id}", email=f"{customer_id}@example.com")
            self.working_memory.update_customer_profile(customer_profile)

        chat_history = self.working_memory.get_conversation_history(customer_id)

        # 2. Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(query)
        print(f"[DEBUG] Sentiment for '{query}': {sentiment}")

        # 3. LLM Interaction (Orchestration)
        llm_raw_response = self._process_llm_response(query, chat_history, customer_profile)
        processed_response = llm_raw_response
        actions_taken = []

        # Check for tool suggestions in LLM response
        if "KNOWLEDGE_LOOKUP: " in llm_raw_response:
            topic = llm_raw_response.split("KNOWLEDGE_LOOKUP: ")[1].split("\n")[0].strip()
            knowledge_info = self.knowledge_base_retriever.retrieve_info(topic)
            if knowledge_info:
                processed_response = f"According to our knowledge base on {topic}: {knowledge_info}\n{processed_response.replace(f'KNOWLEDGE_LOOKUP: {topic}', '').strip()}"
                actions_taken.append(f"Knowledge lookup for: {topic}")
            else:
                processed_response = processed_response.replace(f'KNOWLEDGE_LOOKUP: {topic}', f'Could not find information on {topic}.')

        if "ACTION: " in llm_raw_response:
            action_str = llm_raw_response.split("ACTION: ")[1].split("\n")[0].strip()
            try:
                action_name_raw, params_str = action_str.split(':', 1)
                action_name = action_name_raw.strip()
                params = {}
                for part in params_str.split(';'):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key.strip()] = value.strip()

                action_result = self.action_executor.execute_action(action_name, **params)
                if action_result:
                    processed_response = f"{action_result}\n{processed_response.replace(f'ACTION: {action_str}', '').strip()}"
                    actions_taken.append(f"Executed action: {action_name} with {params}")
                else:
                    processed_response = processed_response.replace(f'ACTION: {action_str}', f'Failed to execute action: {action_name}.')
            except Exception as e:
                processed_response = processed_response.replace(f'ACTION: {action_str}', f'Error parsing action: {e}.')
                print(f"[ERROR] Failed to parse/execute action: {e}")

        # 4. Policy Enforcement
        final_response_text, is_compliant = self.policy_enforcer.enforce_policy(processed_response, sentiment)
        print(f"[DEBUG] Policy compliance: {is_compliant}")

        # 5. Add to working memory
        self.working_memory.add_conversation_entry(customer_id, ConversationEntry(speaker="Customer", text=query, timestamp="now"))
        self.working_memory.add_conversation_entry(customer_id, ConversationEntry(speaker="Agent", text=final_response_text, timestamp="now"))
        
        # Update customer profile history (simple concat for this example)
        customer_profile.history.append(f"Customer: {query}")
        customer_profile.history.append(f"Agent: {final_response_text}")
        self.working_memory.update_customer_profile(customer_profile)

        return AgentResponse(response_text=final_response_text, sentiment=sentiment, actions_taken=actions_taken, compliant=is_compliant)


# --- Main Application Loop (for demonstration) ---
if __name__ == "__main__":
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    agent = SmartCustomerSupportAgent(llm_api_key=os.getenv("OPENAI_API_KEY"))

    print("\n--- Smart Customer Support Agent Demo ---")
    print("Type 'exit' to end the conversation.")
    print("You can interact with different customer IDs to simulate multiple users.")

    current_customer_id = "user_123"

    while True:
        user_input = input(f"\nCustomer {current_customer_id}> ")

        if user_input.lower() == 'exit':
            break
        if user_input.lower().startswith('switch user '):
            new_user_id = user_input.split(' ', 2)[2].strip()
            current_customer_id = new_user_id
            print(f"Switched to customer ID: {current_customer_id}")
            continue

        response = agent.process_customer_query(current_customer_id, user_input)
        print(f"Agent: {response.response_text}")
        print(f"[INFO] Sentiment: {response.sentiment}, Actions: {', '.join(response.actions_taken) or 'None'}, Compliant: {response.compliant}")

