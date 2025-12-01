import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Base Language Model (Chatbot Core) ---
# Load a pre-trained small LLM
print("Loading base language model (distilgpt2). This may take a moment...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Add a pad token if not present, crucial for batch generation and some models
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))

def generate_chatbot_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    # Generate multiple candidate responses for later selection/ranking
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        num_return_sequences=2,  # Generate 2 options for human comparison
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses[0].replace(prompt, "").strip(), responses[1].replace(prompt, "").strip()

# --- 3. Reward Model (Placeholder) ---
# In a real application, this would be a trained model that predicts human preference
# For this example, we'll use a dummy function.
# It could output a score or a direct preference.

def predict_preference_score(response_text):
    # Simulate a reward model by returning a random score between 0 and 1
    # In a real scenario, this would be based on human feedback data.
    return torch.rand(1).item()

# --- 4. Reinforcement Learning from Human Feedback (RLHF) / Rejection Sampling Logic (Placeholder) ---
# This function would use the Reward Model to select or fine-tune responses.
# For this simplified example, we'll simulate a 'better' response selection.

def optimize_response_with_rm(prompt, response1, response2):
    # In a real RLHF setup, the base LM would be fine-tuned using PPO or similar.
    # For rejection sampling, we'd generate many responses and pick the highest-rewarding one.
    
    score1 = predict_preference_score(response1)
    score2 = predict_preference_score(response2)
    
    if score1 > score2:
        return response1, f"Response 1 (Score: {score1:.2f}) was preferred over Response 2 (Score: {score2:.2f}) by the simulated RM."
    else:
        return response2, f"Response 2 (Score: {score2:.2f}) was preferred over Response 1 (Score: {score1:.2f}) by the simulated RM."

# --- Human Feedback Storage (Placeholder) ---
human_feedback_data = []

def record_feedback(query, response1, response2, preferred_response_idx):
    feedback_entry = {
        "query": query,
        "response1": response1,
        "response2": response2,
        "preferred_response_idx": preferred_response_idx
    }
    human_feedback_data.append(feedback_entry)
    print(f"Feedback recorded: {feedback_entry}")
    return "Thank you for your feedback!"

# --- Gradio Interface ---
def chatbot_interface(user_input):
    # Generate initial responses from the base LLM
    resp1, resp2 = generate_chatbot_response(user_input)
    
    # Simulate the optimization step (e.g., rejection sampling with RM)
    optimized_resp, optimization_notes = optimize_response_with_rm(user_input, resp1, resp2)
    
    # Display the optimized response and allow for human feedback on the *original* two options
    return optimized_resp, resp1, resp2, gr.Radio.update(choices=[("Response 1", 1), ("Response 2", 2)], value=None)


def feedback_submission(query, response1_text, response2_text, preferred_option):
    if preferred_option is None:
        return "Please select a preferred option before submitting."
    
    record_feedback(query, response1_text, response2_text, preferred_option)
    return "Feedback submitted!"

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AI-Powered Customer Support Chatbot with Human-Aligned Response Optimization
        This demo simulates a chatbot that uses a **Reward Model (RM)** and **Reinforcement Learning from Human Feedback (RLHF)** (or Rejection Sampling) 
        to generate better responses. 
        
        **How it works:**
        1.  You ask a question.
        2.  The *Base Chatbot* generates two candidate responses (Response 1, Response 2).
        3.  A **Simulated Reward Model** then 'optimizes' and selects one of these as the **Chatbot's Optimized Response**.
        4.  You can then provide human feedback by selecting which of the *original* two responses you preferred.
        
        *(Note: The Reward Model and RLHF components are simplified/simulated for demonstration purposes.)*
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(label="Your Query", placeholder="How can I reset my password?")
            submit_button = gr.Button("Get Chatbot Response")
            
            chatbot_optimized_response = gr.Textbox(label="Chatbot's Optimized Response (via Simulated RM/RLHF)", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown("### Human Feedback: Which original response was better?")
            original_response1_text = gr.Textbox(label="Original Response 1", interactive=False)
            original_response2_text = gr.Textbox(label="Original Response 2", interactive=False)
            
            preferred_option = gr.Radio(label="Your Preference", choices=[], value=None)
            feedback_button = gr.Button("Submit Feedback")
    
    submit_button.click(
        chatbot_interface,
        inputs=user_query,
        outputs=[chatbot_optimized_response, original_response1_text, original_response2_text, preferred_option]
    )
    
    feedback_button.click(
        feedback_submission,
        inputs=[user_query, original_response1_text, original_response2_text, preferred_option],
        outputs=gr.Textbox(label="Feedback Status", value="")
    )

demo.launch()