import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def paraphrase_prompt(prompt, num_return_sequences=3, temperature=0.7):
    inputs = tokenizer.encode("paraphrase: " + prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=128,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        do_sample=True, # Enable sampling for diverse outputs
        top_k=50, # Sample from top 50 most likely tokens
        top_p=0.95, # Nucleus sampling
        early_stopping=True,
    )

    paraphrases = []
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        if decoded_output.lower().strip() != prompt.lower().strip(): # Avoid returning the exact original prompt
            paraphrases.append(decoded_output)

    # Remove duplicates while preserving order as much as possible, or just use a set if order doesn't matter much
    unique_paraphrases = []
    seen = set()
    for p in paraphrases:
        if p not in seen:
            unique_paraphrases.append(p)
            seen.add(p)
            
    if not unique_paraphrases:
        return ["Could not generate suitable paraphrases."]

    return unique_paraphrases

# Create Gradio interface
iface = gr.Interface(
    fn=paraphrase_prompt,
    inputs=gr.Textbox(lines=5, label="Enter Customer Support Query"),
    outputs=gr.Textbox(lines=10, label="Generated Paraphrases"),
    title="AI Customer Support Prompt Paraphraser",
    description="Enter a customer support query and get several paraphrased versions to improve chatbot understanding or data augmentation."
)

# Launch the Gradio app
iface.launch()