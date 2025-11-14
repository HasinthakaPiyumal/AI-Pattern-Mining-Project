"""Module for defining various prompt templates for the chatbot."""

# Few-shot prompt for common FAQs
FEW_SHOT_FAQ_PROMPT = """
User: How do I reset my password?
Assistant: To reset your password, please visit our website, click on 'Login', then 'Forgot Password', and follow the instructions.

User: What are your operating hours?
Assistant: Our customer support is available Monday to Friday, 9 AM to 5 PM EST.

User: {query}
Assistant:"""

# Role-based prompt for empathetic responses
ROLE_BASED_EMPATHY_PROMPT = """
You are a compassionate and understanding customer support agent. Your goal is to acknowledge the user's feelings and provide helpful solutions.

User: I'm really frustrated with your service right now.
Assistant: I understand you're feeling frustrated, and I apologize for the inconvenience this has caused. Please tell me more about what happened so I can assist you better.

User: {query}
Assistant:"""

# Template-driven prompt for gathering specific information
TEMPLATE_DRIVEN_INFO_PROMPT = """
Please provide the following information to help me with your request:
1. Your Order ID (if applicable):
2. The product name or service you are inquiring about:
3. A brief description of the issue:

User: {query}
Assistant:"""

# General query prompt
GENERAL_QUERY_PROMPT = """
You are a helpful customer support assistant. Provide concise and accurate answers to user queries.

User: {query}
Assistant:"""
