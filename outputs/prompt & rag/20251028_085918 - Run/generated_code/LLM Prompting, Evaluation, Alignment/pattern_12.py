from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import os

# --- Configuration ---
# Make sure to set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# --- Pydantic Models ---
class CustomerQuery(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = []

class AssistantResponse(BaseModel):
    response: str
    quality_score: float = 0.0
    feedback_prompt: str = ""

# --- 1. Prompt Engineering Module ---
class PromptEngineer:
    def __init__(self):
        self.templates = {
            "general": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful customer support assistant. Provide concise and accurate information."),
                ("user", "{query}")
            ]),
            "technical": ChatPromptTemplate.from_messages([
                ("system", "You are a technical support specialist. Provide detailed troubleshooting steps and solutions."),
                ("user", "{query}")
            ]),
            "billing": ChatPromptTemplate.from_messages([
                ("system", "You are a billing expert. Provide clear explanations for charges and payment options."),
                ("user", "{query}")
            ]),
            "few_shot": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Here are some examples of how to respond:"),
                ("user", "What is your return policy?"),
                ("assistant", "Our return policy allows returns within 30 days of purchase with a valid receipt."),
                ("user", "{query}")
            ]),
            "role_based": ChatPromptTemplate.from_messages([
                ("system", "You are a friendly and empathetic customer service agent. Your goal is to resolve issues with kindness."),
                ("user", "{query}")
            ]),
        }

    def select_prompt_strategy(self, query: str) -> ChatPromptTemplate:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["technical", "troubleshoot", "error"]):
            return self.templates["technical"]
        elif any(keyword in query_lower for keyword in ["bill", "charge", "payment", "invoice"]):
            return self.templates["billing"]
        elif "return policy" in query_lower or "exchange" in query_lower:
            # Example of using few-shot for a specific, common query type
            return self.templates["few_shot"]
        elif "frustrated" in query_lower or "unhappy" in query_lower:
            return self.templates["role_based"]
        return self.templates["general"]

# --- 2. Large Language Model (LLM) Interface Module ---
class LLMInterface:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

    def generate_response(self, prompt: ChatPromptTemplate, query: str, conversation_history: List[Dict[str, str]]) -> str:
        # Langchain handles history well, but for simplicity, we'll combine here for prompt input
        # In a real app, history would be passed to the LLM's chat method directly
        messages = []
        for msg in conversation_history:
            messages.append((msg["role"], msg["content"]))
        messages.append(("user", query))
        
        # Bind the query to the last user message in the prompt
        # Note: This is a simplified binding. A more robust solution with LangChain would involve LCEL or Runnable.
        formatted_prompt = prompt.format_messages(query=query)
        
        # For this simplified example, we'll just use the last message content with the system message.
        # A full LangChain implementation would use the `invoke` method on the prompt template with LLM.
        # For direct LLM call without complex chain, we'll construct the messages.
        system_message = formatted_prompt[0].content # Assuming system message is first
        user_message = query # We are passing query directly

        response = self.llm.invoke([("system", system_message), ("user", user_message)]).content
        return response

# --- 3. Quality Assurance and Control Module ---
class QualityAssurance:
    def __init__(self, model_name: str = "gpt-4o"):
        self.qa_llm = ChatOpenAI(model_name=model_name, temperature=0.0)

    def autorate_response(self, original_query: str, assistant_response: str) -> float:
        prompt = f"""Rate the quality, relevance, and helpfulness of the assistant's response to the customer's query on a scale of 0 to 10. 
                     0 being completely unhelpful/irrelevant and 10 being perfectly helpful/relevant.
                     Only output the numerical score.

                     Customer Query: {original_query}
                     Assistant Response: {assistant_response}
                     Rating:"""
        try:
            rating_str = self.qa_llm.invoke(prompt).content.strip()
            return float(rating_str) if rating_str.replace('.', '', 1).isdigit() else 0.0
        except Exception as e:
            print(f"Error during autorating: {e}")
            return 0.0

    def check_consistency(self, original_query: str, assistant_response: str) -> bool:
        prompt = f"""Does the following assistant response fully and accurately address the customer's original question?
                     Respond with 'YES' if it does, and 'NO' if it does not.

                     Customer Query: {original_query}
                     Assistant Response: {assistant_response}
                     Fully Addressed?:"""
        response = self.qa_llm.invoke(prompt).content.strip().upper()
        return response == "YES"

    def detect_bias_and_truthfulness(self, query: str, response: str) -> Dict[str, bool]:
        # This is a simplified example. Real adversarial evaluation would be more complex.
        bias_prompt = f"""Analyze the following assistant response for any potential biases, stereotypes, or unfair language.
                           If you find any, state 'BIAS_DETECTED'. Otherwise, state 'NO_BIAS'.
                           Query: {query}
                           Response: {response}
                           Bias Check:"""
        bias_check = self.qa_llm.invoke(bias_prompt).content.strip().upper()

        truth_prompt = f"""Based on general knowledge, is the following statement factually accurate?
                            Respond with 'ACCURATE' or 'INACCURATE'.
                            Statement: {response}
                            Accuracy Check:"""
        truth_check = self.qa_llm.invoke(truth_prompt).content.strip().upper()

        return {"bias_detected": bias_check == "BIAS_DETECTED", "is_truthful": truth_check == "ACCURATE"}

    def enforce_constitutional_ai(self, query: str, response: str) -> str:
        # Example of a simple self-correction prompt based on ethical principles
        ethical_principles = [
            "Do not generate harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
            "Always be helpful, respectful, and honest.",
            "Prioritize user safety and privacy."
        ]
        principles_str = "\n".join([f"- {p}" for p in ethical_principles])

        correction_prompt = f"""You previously generated the following response to the user's query:

                                  Customer Query: {query}
                                  Assistant Response: {response}

                                  Please review this response against the following ethical guidelines and rephrase it if necessary to fully comply with them. 
                                  If the response already complies, state 'COMPLIES'. Otherwise, provide a revised, compliant response.

                                  Ethical Guidelines:
                                  {principles_str}

                                  Revised Response (or 'COMPLIES'):"""
        
        corrected_response = self.qa_llm.invoke(correction_prompt).content.strip()
        if corrected_response == "COMPLIES":
            return response
        return corrected_response

# --- FastAPI Application --- 
app = FastAPI(
    title="Generative AI Customer Support Assistant",
    description="An AI assistant with advanced behavior control and quality assurance."
)

# Initialize modules
prompt_engineer = PromptEngineer()
llm_interface = LLMInterface()
qa_module = QualityAssurance()

@app.post("/chat", response_model=AssistantResponse)
async def chat_with_assistant(customer_query: CustomerQuery):
    # 1. Prompt Engineering
    selected_prompt_template = prompt_engineer.select_prompt_strategy(customer_query.query)
    
    # 2. LLM Interface - Get initial response
    initial_response = llm_interface.generate_response(
        selected_prompt_template, 
        customer_query.query, 
        customer_query.conversation_history
    )

    # 3. Quality Assurance and Control (Iterative Checks/Corrections)
    final_response = initial_response
    quality_score = 0.0
    feedback_prompt = []

    # Autorating
    quality_score = qa_module.autorate_response(customer_query.query, final_response)
    feedback_prompt.append(f"Initial Quality Score: {quality_score:.2f}")

    # Consistency Check
    if not qa_module.check_consistency(customer_query.query, final_response):
        feedback_prompt.append("Response might not fully address the query. Attempting revision.")
        # In a real system, you might regenerate the response here or provide a different prompt
        # For simplicity, we'll just flag it.

    # Bias and Truthfulness Check
    qa_checks = qa_module.detect_bias_and_truthfulness(customer_query.query, final_response)
    if qa_checks["bias_detected"]:
        feedback_prompt.append("Potential bias detected. Applying Constitutional AI principles.")
        final_response = qa_module.enforce_constitutional_ai(customer_query.query, final_response)
    if not qa_checks["is_truthful"]:
        feedback_prompt.append("Response might be factually inaccurate. Applying Constitutional AI principles.")
        final_response = qa_module.enforce_constitutional_ai(customer_query.query, final_response)

    # Constitutional AI Enforcement (final pass if not already applied for bias/truth)
    # This step ensures a final check, especially if no bias/truth issues were initially flagged
    # but a general ethical review is always good.
    revised_by_constitutional_ai = qa_module.enforce_constitutional_ai(customer_query.query, final_response)
    if revised_by_constitutional_ai != final_response:
        final_response = revised_by_constitutional_ai
        feedback_prompt.append("Response revised by Constitutional AI for ethical alignment.")

    # Re-evaluate quality after potential revisions
    final_quality_score = qa_module.autorate_response(customer_query.query, final_response)
    feedback_prompt.append(f"Final Quality Score (after QA): {final_quality_score:.2f}")

    return AssistantResponse(
        response=final_response,
        quality_score=final_quality_score,
        feedback_prompt="\n".join(feedback_prompt)
    )

# To run this application:
# 1. Save the code as `customer_support_assistant.py`
# 2. Install dependencies: `pip install fastapi "uvicorn[standard]" pydantic langchain-openai`
# 3. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY='your_key_here'`
#    (On Windows, use `set OPENAI_API_KEY=your_key_here` in cmd or PowerShell)
# 4. Run the application: `uvicorn customer_support_assistant:app --reload`
# 5. Access the API at http://127.0.0.1:8000/docs for interactive documentation.