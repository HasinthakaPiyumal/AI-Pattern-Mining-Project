
import collections

class IntentUnderstandingModule:
    def __init__(self):
        self.intents = {
            "refund_request": ["refund", "money back", "return item"],
            "order_tracking": ["where is my order", "track package", "delivery status"],
            "product_inquiry": ["product details", "specifications", "about this item"],
            "account_update": ["change password", "update address", "my account"],
            "general_query": []
        }
        self.personalization_data = collections.defaultdict(list)

    def classify_intent(self, query: str, user_id: str):
        query_lower = query.lower()
        best_intent = "general_query"
        max_match = 0
        entities = {}

        # Simple keyword-based intent classification
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    count = query_lower.count(keyword)
                    if count > max_match:
                        max_match = count
                        best_intent = intent

        # Incorporate personalization
        if user_id in self.personalization_data:
            for personal_phrase, personal_intent in self.personalization_data[user_id]:
                if personal_phrase.lower() in query_lower:
                    best_intent = personal_intent
                    break

        # Simple entity extraction (placeholder)
        if "order" in query_lower and "id" in query_lower:
            # A very simplistic regex-like extraction
            import re
            match = re.search(r"order\s*(?:id)?\s*(\d+)", query_lower)
            if match: 
                entities["order_id"] = match.group(1)
        
        if "product" in query_lower:
            match = re.search(r"product\s*(\w+)", query_lower)
            if match:
                entities["product_name"] = match.group(1)

        # Simulate confidence
        confidence = 0.8 if best_intent != "general_query" else 0.5
        if not entities and best_intent not in ["general_query", "account_update"]:
            confidence = 0.4 # Lower confidence if entities are missing for action-oriented intents

        return best_intent, entities, confidence

    def detect_ambiguity(self, intent: str, entities: dict, confidence: float):
        if confidence < 0.6:
            if intent == "refund_request" and "order_id" not in entities:
                return True, "It seems you want a refund. Could you please provide your order ID?"
            if intent == "order_tracking" and "order_id" not in entities:
                return True, "To track your order, I need the order ID. Can you provide it?"
            if intent == "product_inquiry" and "product_name" not in entities:
                return True, "What product are you interested in? Please specify the product name."
            return True, "I'm not entirely sure what you mean. Could you please rephrase or provide more details?"
        return False, None

    def update_personalization(self, user_id: str, query: str, actual_intent: str):
        # In a real system, this would involve more sophisticated learning
        # For this demo, we'll just store a direct mapping if an intent was successfully handled.
        if (query, actual_intent) not in self.personalization_data[user_id]:
            self.personalization_data[user_id].append((query, actual_intent))

class ToolExecutionModule:
    def __init__(self):
        pass

    def _get_order_status(self, order_id: str):
        if order_id == "12345":
            return f"Order {order_id} is currently out for delivery and expected by 5 PM today."
        return f"Could not find status for order {order_id}."

    def _initiate_refund(self, order_id: str, reason: str = ""):
        if order_id == "12345":
            return f"Refund for order {order_id} initiated successfully due to: {reason}. It will be processed within 3-5 business days."
        return f"Cannot initiate refund for order {order_id}. Please ensure the order ID is correct."

    def _get_product_info(self, product_name: str):
        if "laptop" in product_name.lower():
            return f"The {product_name} is a high-performance model with 16GB RAM and a 512GB SSD. It costs $1200."
        return f"I couldn't find detailed information for {product_name}."

    def _update_account(self, user_id: str, field: str, value: str):
        return f"Account update request for user {user_id}, field '{field}' with new value '{value}' has been received. Please verify via email."

    def execute_tool(self, intent: str, entities: dict):
        if intent == "order_tracking" and "order_id" in entities:
            return self._get_order_status(entities["order_id"])
        elif intent == "refund_request" and "order_id" in entities:
            reason = entities.get("reason", "customer request")
            return self._initiate_refund(entities["order_id"], reason)
        elif intent == "product_inquiry" and "product_name" in entities:
            return self._get_product_info(entities["product_name"])
        elif intent == "account_update" and "user_id" in entities and "field" in entities and "value" in entities:
            return self._update_account(entities["user_id"], entities["field"], entities["value"])
        return "No specific tool found for this request or missing information."

class ResponseGenerationModule:
    def __init__(self):
        pass

    def generate_response(self, query: str, intent: str, entities: dict, tool_output: str = None, clarification_needed: str = None):
        if clarification_needed:
            return clarification_needed

        if tool_output and tool_output != "No specific tool found for this request or missing information.":
            return tool_output

        if intent == "refund_request":
            return "I understand you're looking for a refund. Can you confirm the order ID?"
        elif intent == "order_tracking":
            return "To track your order, please provide your order ID."
        elif intent == "product_inquiry":
            return "What product are you interested in? I can help you with its details."
        elif intent == "account_update":
            return "I can help you update your account. What information do you need to change?"
        elif intent == "general_query":
            return "How can I assist you today?"
        else:
            return "I'm sorry, I couldn't fully understand your request. Can you please rephrase it?"

class SmartCustomerSupportAgent:
    def __init__(self):
        self.intent_module = IntentUnderstandingModule()
        self.tool_module = ToolExecutionModule()
        self.response_module = ResponseGenerationModule()
        self.chat_history = collections.defaultdict(list)

    def process_query(self, user_id: str, query: str):
        self.chat_history[user_id].append({"role": "user", "content": query})

        intent, entities, confidence = self.intent_module.classify_intent(query, user_id)
        
        is_ambiguous, clarification_question = self.intent_module.detect_ambiguity(intent, entities, confidence)

        if is_ambiguous:
            response = self.response_module.generate_response(query, intent, entities, clarification_needed=clarification_question)
        else:
            tool_output = self.tool_module.execute_tool(intent, entities)
            response = self.response_module.generate_response(query, intent, entities, tool_output=tool_output)
            
            if tool_output != "No specific tool found for this request or missing information.":
                self.intent_module.update_personalization(user_id, query, intent)

        self.chat_history[user_id].append({"role": "agent", "content": response})
        return response

# --- FastAPI Backend Simulation (Conceptual) ---
# from fastapi import FastAPI
# from pydantic import BaseModel
# app = FastAPI()
# agent = SmartCustomerSupportAgent()

# class ChatRequest(BaseModel:
#     user_id: str
#     query: str

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     response = agent.process_query(request.user_id, request.query)
#     return {"response": response, "chat_history": agent.chat_history[request.user_id]}

# --- Streamlit Frontend Simulation (Conceptual) ---
# import streamlit as st

# def streamlit_app():
#     st.title("Smart Customer Support Agent")

#     if "agent" not in st.session_state:
#         st.session_state.agent = SmartCustomerSupportAgent()
#     if "user_id" not in st.session_state:
#         st.session_state.user_id = "test_user_123"

#     st.write(f"Welcome, User {st.session_state.user_id}!")

#     for message in st.session_state.agent.chat_history[st.session_state.user_id]:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("How can I help you?"):
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         response = st.session_state.agent.process_query(st.session_state.user_id, prompt)
#         with st.chat_message("agent"):
#             st.markdown(response)

# if __name__ == "__main__":
#     # For actual running, uncomment the relevant sections
#     # To run FastAPI: uvicorn your_file_name:app --reload
#     # To run Streamlit: streamlit run your_file_name.py
#     # For this generation, we just define the classes and their interaction.

#     # Example interaction for demonstration (without FastAPI/Streamlit running)
    agent_instance = SmartCustomerSupportAgent()
    user_id = "demo_user_001"

    print("--- Demo Conversation ---")

    query1 = "I need to track my order."
    response1 = agent_instance.process_query(user_id, query1)
    print(f"User: {query1}")
    print(f"Agent: {response1}\n")

    query2 = "My order ID is 12345."
    response2 = agent_instance.process_query(user_id, query2)
    print(f"User: {query2}")
    print(f"Agent: {response2}\n")

    query3 = "I want a refund for order 12345."
    response3 = agent_instance.process_query(user_id, query3)
    print(f"User: {query3}")
    print(f"Agent: {response3}\n")

    query4 = "Tell me about a laptop."
    response4 = agent_instance.process_query(user_id, query4)
    print(f"User: {query4}")
    print(f"Agent: {response4}\n")

    query5 = "Change my address to 123 Main St."
    response5 = agent_instance.process_query(user_id, query5)
    print(f"User: {query5}")
    print(f"Agent: {response5}\n")

    query6 = "Something general."
    response6 = agent_instance.process_query(user_id, query6)
    print(f"User: {query6}")
    print(f"Agent: {response6}\n")

    print("--- Chat History (Demo User) ---")
    for msg in agent_instance.chat_history[user_id]:
        print(f"{msg['role'].capitalize()}: {msg['content']}")

