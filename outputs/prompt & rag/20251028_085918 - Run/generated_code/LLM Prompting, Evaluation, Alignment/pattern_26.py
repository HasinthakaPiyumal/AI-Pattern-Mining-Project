from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading
import streamlit as st
import requests

# --- Vector Store Simulation ---
EXEMPLARS = {
    "few_shot_example_1": "Query: How do I reset my password? Answer: To reset your password, visit the 'Forgot Password' link on the login page.",
    "few_shot_example_2": "Query: What are your operating hours? Answer: Our customer support operates 24/7.",
    "few_shot_example_3": "Query: Where can I find my order history? Answer: You can find your order history under the 'My Account' section.",
}

def get_few_shot_exemplars(query, num_examples=1):
    if "password" in query.lower():
        return [EXEMPLARS["few_shot_example_1"]]
    elif "hours" in query.lower():
        return [EXEMPLARS["few_shot_example_2"]]
    elif "order" in query.lower():
        return [EXEMPLARS["few_shot_example_3"]]
    return [list(EXEMPLARS.values())[0]]

# --- LLM Integration Layer Simulation ---
class MockLLM:
    def generate(self, prompt, temperature=0.7):
        if "reset my password" in prompt.lower():
            return "To reset your password, please go to the login page and click 'Forgot Password'. Follow the instructions to set a new one."
        elif "operating hours" in prompt.lower():
            return "Our customer support team is available 24 hours a day, 7 days a week to assist you."
        elif "rephrase the following query" in prompt.lower():
            original_part = prompt.split("rephrase the following query: ", 1)[-1].strip()
            return f"Rephrased for clarity: '{original_part.strip('?').strip('.')}' is what you're asking about. Let me help."
        elif "review the original query and your initial response" in prompt.lower():
            if "fully address" in prompt.lower() and "improve" in prompt.lower():
                return "Upon review, the initial response could be more direct. Improved: Our support is always available."
            return "The initial response seems adequate."
        elif "think step-by-step" in prompt.lower():
            return "Thought Process: First, identify keywords. Second, search knowledge base. Third, formulate answer. Final Answer: Step-by-step guidance is coming."
        elif "identify the main topic" in prompt.lower():
            return "The main topic is 'account management'."
        elif "detailed answer to the original query" in prompt.lower():
            return "Based on account management, here is your detailed answer regarding your query."
        elif "ethical" in prompt.lower():
            return "This response adheres to ethical guidelines and is unbiased."
        elif "evaluate the following response" in prompt.lower():
            return "Evaluation: Score 4/5 for accuracy, 5/5 for relevance, 4/5 for helpfulness. Good response."
        elif "needs improvement based on evaluation" in prompt.lower():
            return "Refined response: We understand your concern and are actively working on a solution to improve this."
        return f"This is a simulated response to: '{prompt}'."

mock_llm = MockLLM()

# --- Prompt Engineering Module ---
def zero_shot_prompt(query):
    return f"Please answer the following customer query concisely: {query}"

def few_shot_prompt(query, exemplars):
    examples_str = "\n".join(exemplars)
    return f"Here are some examples:\n{examples_str}\n\nBased on these examples, answer the following customer query: {query}"

def template_based_prompt(query, template_name):
    templates = {
        "greeting": "Hello! How can I assist you with your {{query_topic}} today?",
        "issue_resolution": "I understand you're experiencing an issue with {{issue_details}}. Let me find a solution for you."
    }
    if "hello" in query.lower() or "hi" in query.lower():
        return templates["greeting"].replace("{{query_topic}}", "general inquiry")
    return templates["issue_resolution"].replace("{{issue_details}}", query.split(' ')[0]) if query else templates["issue_resolution"].replace("{{issue_details}}", "your request")

def role_prompt(query, role="support agent"):
    return f"Act as a friendly, knowledgeable, and professional {role}. {query}"

def style_prompt(query, style="professional"):
    return f"Respond in a {style}, clear, and helpful tone: {query}"

def emotion_prompt(query, emotion="empathetic"):
    return f"Show {emotion} and understanding in your response: {query}"

# --- Reasoning Module ---
def rephrase_and_respond(llm, query):
    rephrase_prompt = f"Please rephrase the following customer query concisely: {query}"
    rephrased_query = llm.generate(rephrase_prompt)
    print(f"DEBUG: Rephrased Query: {rephrased_query}")
    response_prompt = f"Now, based on the rephrased query: '{rephrased_query}', provide the best possible answer to the original intent of: {query}"
    return llm.generate(response_prompt)

def rereading_mechanism(llm, original_query, initial_response):
    reread_prompt = f"Original query: {original_query}\nInitial response: {initial_response}\n\nPlease critically review the original query and your initial response. Does the initial response fully and accurately address the query? If not, provide an improved, complete response."
    return llm.generate(reread_prompt)

def metacognitive_prompting(llm, query):
    meta_prompt = f"You are an expert problem solver. Think step-by-step about how to best answer the following customer query: {query}\nAfter your detailed thought process, provide only the final, concise answer."
    return llm.generate(meta_prompt)

def prompt_chain(llm, query):
    step1_prompt = f"First, carefully analyze this customer query and identify the core issue or main topic: {query}"
    step1_result = llm.generate(step1_prompt)
    print(f"DEBUG: Step 1 Result (Topic): {step1_result}")
    step2_prompt = f"Now, based on the core issue identified as '{step1_result}', provide a comprehensive and helpful answer to the original query: {query}"
    return llm.generate(step2_prompt)

# --- Ethical Alignment Module ---
def constitutional_ai_check(llm, response):
    ethical_prompt = f"Review the following response for fairness, absence of bias, helpfulness, and safety, adhering to constitutional AI principles. If any issues are found, suggest specific improvements. Response: {response}"
    return llm.generate(ethical_prompt)

# --- Validation and Evaluation Layer ---
def llm_based_evaluation(llm, query, generated_response):
    eval_prompt = f"As an expert evaluator, assess the following response to a customer query. Provide a confidence score (1-5, 5 being highest) for accuracy, relevance, and overall helpfulness.\nQuery: {query}\nResponse: {generated_response}"
    return llm.generate(eval_prompt)

def round_trip_consistency_check(llm, original_data, generated_data):
    check_prompt = f"Original intent or data: {original_data}\nGenerated output: {generated_data}\nAre these two pieces of information consistent in meaning and intent? Answer Yes/No."
    return llm.generate(check_prompt)

def adversarial_evaluation(llm, query, generated_response):
    adv_prompt = f"As an adversarial tester, find any inaccuracies, misleading information, or potential biases in the following response to the query.\nQuery: {query}\nResponse: {generated_response}"
    return llm.generate(adv_prompt)

# --- LLM Orchestration Layer ---
class LLMOrchestrator:
    def __init__(self, llm_model):
        self.llm = llm_model

    def process_query(self, query, strategy="zero_shot"):
        final_response = ""
        initial_response = ""
        prompt = ""

        if strategy == "few_shot":
            exemplars = get_few_shot_exemplars(query)
            prompt = few_shot_prompt(query, exemplars)
        elif strategy == "template":
            prompt = template_based_prompt(query, "issue_resolution")
        elif strategy == "role_professional":
            prompt = role_prompt(query, role="professional support agent")
        elif strategy == "emotion_empathetic":
            prompt = emotion_prompt(query, emotion="deeply empathetic")
        else:
            prompt = zero_shot_prompt(query)

        initial_response = self.llm.generate(prompt)
        final_response = initial_response

        if strategy == "complex_rephrase":
            final_response = rephrase_and_respond(self.llm, query)
        elif strategy == "review_reread":
            final_response = rereading_mechanism(self.llm, query, initial_response)
        elif strategy == "steps_metacognitive":
            final_response = metacognitive_prompting(self.llm, query)
        elif strategy == "chain_prompt":
            final_response = prompt_chain(self.llm, query)

        ethical_review = constitutional_ai_check(self.llm, final_response)
        print(f"MONITOR: Ethical Review: {ethical_review}")

        validation_result = llm_based_evaluation(self.llm, query, final_response)
        print(f"MONITOR: Validation Result: {validation_result}")

        if "needs improvement" in validation_result.lower() or "1/5" in validation_result or "2/5" in validation_result:
            final_response = self.llm.generate(f"The previous response '{final_response}' for query '{query}' received low evaluation scores. Please refine and improve it for clarity, accuracy, and helpfulness.")

        return final_response

orchestrator = LLMOrchestrator(mock_llm)

# --- FastAPI Backend ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    strategy: str = "zero_shot"

@app.post("/chat")
async def chat_with_assistant(request: QueryRequest):
    response = orchestrator.process_query(request.query, request.strategy)
    return {"response": response}

# --- Streamlit UI ---
STREAMLIT_SERVER_STARTED = False

def run_streamlit_app():
    st.title("Intelligent Customer Support Assistant")

    user_query = st.text_area("Enter your query:", "How do I reset my password?")
    strategy_options = [
        "zero_shot",
        "few_shot",
        "template",
        "role_professional",
        "emotion_empathetic",
        "complex_rephrase",
        "review_reread",
        "steps_metacognitive",
        "chain_prompt"
    ]
    selected_strategy = st.selectbox("Select Orchestration Strategy:", strategy_options, index=0)

    if st.button("Get Assistant Response"):
        with st.spinner("Getting response..."):
            # The FastAPI endpoint expects a single 'strategy' parameter
            # For reasoning methods, we'll pass the specific strategy name and orchestrator handles the internal call.
            response = requests.post("http://127.0.0.1:8000/chat", json={"query": user_query, "strategy": selected_strategy})
            if response.status_code == 200:
                st.write("### Assistant Response:")
                st.info(response.json()["response"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

def run_fastapi_server():
    global STREAMLIT_SERVER_STARTED
    if not STREAMLIT_SERVER_STARTED:
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()
    STREAMLIT_SERVER_STARTED = True
    print("FastAPI server started in a background thread.")
    run_streamlit_app()

