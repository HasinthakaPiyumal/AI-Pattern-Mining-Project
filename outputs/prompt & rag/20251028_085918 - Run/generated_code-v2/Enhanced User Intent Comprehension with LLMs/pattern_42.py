import gradio as gr
from pydantic import BaseModel
from loguru import logger
from typing import Optional, Dict, Any

# Simulate Langchain components and a simple "LLM"

# 1. Pydantic Models for Structured Output
class Entities(BaseModel):
    account_number: Optional[str] = None
    issue_description: Optional[str] = None
    plan_name: Optional[str] = None
    contact_info: Optional[str] = None
    query_detail: Optional[str] = None

class IntentOutput(BaseModel):
    intent: str
    entities: Entities
    clarification_needed: bool = False
    clarification_question: Optional[str] = None

# 2. Simulated LLM for Intent Recognition and Entity Extraction
def _simulate_llm_intent_recognition(query: str) -> IntentOutput:
    query_lower = query.lower()

    if "bill" in query_lower or "invoice" in query_lower:
        account_number = next((word for word in query.split() if word.isdigit() and len(word) == 10), None)
        return IntentOutput(intent="billing_inquiry", entities=Entities(account_number=account_number), clarification_needed= not bool(account_number), clarification_question="Could you please provide your 10-digit account number?")
    elif "technical" in query_lower or "internet not working" in query_lower or "problem" in query_lower:
        return IntentOutput(intent="technical_support", entities=Entities(issue_description=query), clarification_needed=False)
    elif "upgrade" in query_lower or "change plan" in query_lower or "new service" in query_lower:
        return IntentOutput(intent="service_upgrade", entities=Entities(plan_name="" if "plan" in query_lower else None), clarification_needed=True, clarification_question="Which plan are you interested in upgrading to?")
    elif "faq" in query_lower or "question" in query_lower or "how to" in query_lower:
        return IntentOutput(intent="get_faq_answer", entities=Entities(query_detail=query), clarification_needed=False)
    else:
        return IntentOutput(intent="general_query", entities=Entities(), clarification_needed=False)

# 3. Simulated Custom Tools
def check_bill(account_number: Optional[str]) -> str:
    if account_number:
        logger.info(f"Simulating bill check for account: {account_number}")
        # In a real app, this would call a backend API
        return f"Your last bill for account {account_number} was $75.50 due on October 26, 2023."
    return "I need your account number to check your bill. Could you please provide it?"

def create_tech_support_ticket(issue_description: str, contact_info: Optional[str] = "not provided") -> str:
    logger.info(f"Simulating tech support ticket for: {issue_description}, contact: {contact_info}")
    # In a real app, this would create a ticket in a CRM/support system
    ticket_id = "TS" + str(hash(issue_description + contact_info) % 10000)
    return f"A technical support ticket (ID: {ticket_id}) has been created for your issue: '{issue_description}'. A representative will contact you shortly."

def upgrade_service(plan_name: Optional[str], account_number: Optional[str]) -> str:
    if plan_name and account_number:
        logger.info(f"Simulating service upgrade for account {account_number} to {plan_name}")
        # In a real app, this would update the customer's service plan
        return f"Your service for account {account_number} has been successfully upgraded to the {plan_name} plan. This will take effect within 24 hours."
    return "To upgrade your service, I need both your desired plan name and your account number."

def get_faq_answer(query_detail: str) -> str:
    logger.info(f"Simulating FAQ search for query: {query_detail}")
    # In a real app, this would query a knowledge base (e.g., Chroma/FAISS)
    if "router setup" in query_detail.lower():
        return "To set up your router, please follow the instructions in the manual, or visit our website for a step-by-step video guide."
    elif "coverage area" in query_detail.lower():
        return "You can check our 5G and fiber optic coverage areas by entering your address on our website under the 'Coverage' section."
    return "I'm sorry, I couldn't find a direct answer in our FAQs for that query. Would you like to speak to a representative?"

# 4. Chatbot Core Logic (Simulating Langchain orchestration)
class Chatbot:
    def __init__(self):
        self.conversation_history = []
        self.current_context: Dict[str, Any] = {}
        self.tools = {
            "check_bill": check_bill,
            "create_tech_support_ticket": create_tech_support_ticket,
            "upgrade_service": upgrade_service,
            "get_faq_answer": get_faq_answer,
        }

    def _get_response(self, user_query: str) -> str:
        # Step 1: LLM for Intent Recognition and Entity Extraction
        intent_output = _simulate_llm_intent_recognition(user_query)
        logger.info(f"Identified Intent: {intent_output.intent}, Entities: {intent_output.entities.dict()}")

        if intent_output.clarification_needed:
            self.current_context = {
                "pending_intent": intent_output.intent,
                "pending_entities": intent_output.entities.dict(),
                "clarification_question": intent_output.clarification_question,
            }
            return intent_output.clarification_question

        # If there's pending context from previous turn, try to resolve
        if self.current_context and not intent_output.clarification_needed: # if current intent detection itself doesn't need clarification
            pending_intent = self.current_context.get("pending_intent")
            pending_entities = self.current_context.get("pending_entities", {})
            clarification_question = self.current_context.get("clarification_question")

            if pending_intent:
                # Attempt to fill missing entities from current user query
                if "account number" in clarification_question.lower() and intent_output.entities.account_number is None:
                    account_num_match = next((word for word in user_query.split() if word.isdigit() and len(word) == 10), None)
                    if account_num_match:
                        pending_entities["account_number"] = account_num_match
                        intent_output.clarification_needed = False # Resolved
                        intent_output.intent = pending_intent # Use the original pending intent
                        intent_output.entities = Entities(**pending_entities)
                elif "plan" in clarification_question.lower() and intent_output.entities.plan_name is None:
                     plan_name_match = next((word for word in user_query.split() if "plan" in word.lower() and len(word) > 4), None)
                     if plan_name_match:
                         pending_entities["plan_name"] = user_query # More robust plan name extraction would be needed
                         intent_output.clarification_needed = False # Resolved
                         intent_output.intent = pending_intent
                         intent_output.entities = Entities(**pending_entities)

                # If clarification still needed after trying to resolve
                if intent_output.clarification_needed:
                    return intent_output.clarification_question # Re-ask clarification if still unresolved from current input
                else: # Clarification resolved, proceed with the original pending intent
                    self.current_context = {}
                    intent_output.intent = pending_intent # Ensure the original intent is used
                    # Entities are already updated in intent_output

        # Step 2: Tool/Action Mapping & Execution
        response_message = "I'm not sure how to handle that. Could you please rephrase?"
        tool_to_call = self.tools.get(intent_output.intent)

        if tool_to_call:
            try:
                # Dynamically call the tool with extracted entities
                valid_params = {k: v for k, v in intent_output.entities.dict().items() if v is not None}
                if intent_output.intent == "create_tech_support_ticket" and "issue_description" not in valid_params:
                    valid_params["issue_description"] = user_query # Fallback if LLM misses it
                response_message = tool_to_call(**valid_params)
            except TypeError as e:
                logger.error(f"Error calling tool {intent_output.intent} with params {valid_params}: {e}")
                response_message = f"There was an error processing your request for {intent_output.intent}. Please try again or contact support."
        elif intent_output.intent == "general_query":
            response_message = "Hello! How can I assist you with your telecommunications needs today?"
        
        self.conversation_history.append((user_query, response_message))
        return response_message

# Initialize Chatbot
chatbot = Chatbot()

# Gradio Interface
def chat_interface(message, history):
    global chatbot
    # Gradio history is a list of lists: [[user_msg, bot_msg], ...]
    # We need to feed the message to our chatbot instance
    response = chatbot._get_response(message)
    return response

# Clear button functionality (optional but good for testing)
def clear_conversation():
    global chatbot
    chatbot = Chatbot() # Reset the chatbot state
    return "", [] # Clear current message and history

logger.remove()
logger.add("chatbot.log", rotation="500 MB", level="INFO")

with gr.Blocks() as demo:
    gr.Markdown("# Telecom Support Chatbot (Simulated)")
    gr.Markdown("This chatbot demonstrates intent understanding, entity extraction, and tool calling.")

    chatbot_output = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear")

    msg.submit(chat_interface, [msg, chatbot_output], [msg, chatbot_output])
    clear.click(clear_conversation, inputs=[], outputs=[msg, chatbot_output])

demo.launch()