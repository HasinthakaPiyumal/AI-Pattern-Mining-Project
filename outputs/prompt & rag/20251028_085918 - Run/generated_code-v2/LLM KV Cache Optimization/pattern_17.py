import gradio as gr
from vllm import LLM, SamplingParams
import torch

# Configuration
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta" # A smaller, conversational model
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Initialize vLLM
# Setting gpu_memory_utilization to a lower value if GPU is shared or limited.
# enforce_eager=True is good for debugging but slower, remove for production.
# trust_remote_code=True might be needed for some models.
llm = LLM(model=MODEL_NAME, max_model_len=MAX_TOKENS, gpu_memory_utilization=0.8)
sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)

# System prompt - a common prefix that will be reused
SYSTEM_PROMPT = (
    "You are a helpful and friendly customer support assistant for a telecommunications company. "
    "Provide concise and accurate information. If you cannot answer, politely state that and suggest "
    "contacting a human agent."
)

# Simulate a knowledge base / RAG retrieval
def get_retrieved_context(query: str) -> str:
    """
    Simulates retrieving relevant information from a knowledge base based on the query.
    In a real application, this would involve embedding the query and searching a vector DB.
    """
    query = query.lower()
    if "bill" in query or "invoice" in query:
        return (
            "Customers can view their latest bill and past invoices by logging into their "
            "online account on our website. Navigate to the 'Billing' section. "
            "For detailed queries, please refer to the billing FAQ or contact support."
        )
    elif "internet speed" in query or "slow internet" in query:
        return (
            "If your internet speed is slow, first try restarting your router. "
            "Ensure all cables are securely connected. You can also run a speed test "
            "on our website. If problems persist, our technical support can run diagnostics."
        )
    elif "change plan" in query or "upgrade" in query:
        return (
            "To change or upgrade your service plan, please visit the 'My Services' section "
            "in your online account. You can explore available plans and make changes there. "
            "Alternatively, contact our sales team for personalized recommendations."
        )
    elif "contact support" in query or "human agent" in query:
        return (
            "You can contact our customer support team by calling 1-800-TELCO-HELP or by "
            "using the live chat feature on our website during business hours (9 AM - 6 PM EST, Mon-Fri)."
        )
    else:
        return "No specific knowledge base article found for this query. I will try to answer generally."

# Chatbot logic
def predict(message: str, history: list):
    full_conversation_history = ""

    # Always start with the system prompt (KV cache will be reused here)
    current_prompt_parts = [SYSTEM_PROMPT]

    # Append RAG context based on the latest user message
    retrieved_context = get_retrieved_context(message)
    if "No specific knowledge base article found" not in retrieved_context:
        current_prompt_parts.append(f"\n\nRetrieved Context: {retrieved_context}")

    # Append previous conversation turns (KV cache will be reused for these prefixes)
    for human, bot in history:
        full_conversation_history += f"\nCustomer: {human}\nAssistant: {bot}"
    
    current_prompt_parts.append(full_conversation_history)

    # Append current user message
    current_prompt_parts.append(f"\nCustomer: {message}\nAssistant:")
    
    prompt = "".join(current_prompt_parts)

    # vLLM handles KV cache reuse internally for shared prefixes
    outputs = llm.generate(prompt, sampling_params)

    # Extract the generated text
    generated_text = ""
    for output in outputs:
        generated_text = output.outputs[0].text
        # Clean up common model generation artifacts if any
        if generated_text.startswith("Assistant:"):
            generated_text = generated_text[len("Assistant:"):].strip()
        # Ensure it doesn't repeat the user's prompt or cut off unnaturally
        if "\nCustomer:" in generated_text:
            generated_text = generated_text.split("\nCustomer:")[0].strip()

    history.append((message, generated_text))
    return "", history

# Gradio interface
demo = gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me a question about your telecom services", container=False, scale=7),
    title="Telecom Customer Support Chatbot (KV Cache Optimized)",
    description="Ask questions about billing, internet, plans, or general support. "
                "This chatbot leverages KV Cache Reuse for faster responses in multi-turn conversations and common queries.",
    theme="soft",
    examples=[
        ["Where can I find my latest bill?"],
        ["My internet speed is very slow, what should I do?"],
        ["I want to upgrade my data plan."],
        ["How do I contact a human agent?"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)