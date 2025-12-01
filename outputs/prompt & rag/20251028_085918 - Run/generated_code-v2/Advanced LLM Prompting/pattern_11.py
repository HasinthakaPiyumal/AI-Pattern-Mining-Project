import gradio as gr
from transformers import pipeline, set_seed

NUM_PARAPHRASES = 3
MAX_PARAPHRASE_LENGTH = 100
MAX_RESPONSE_LENGTH = 150

paraphrase_generator = pipeline(
    "text2text-generation",
    model="t5-small",
    framework="pt",
    device=-1
)

response_generator = pipeline(
    "text-generation",
    model="gpt2",
    framework="pt",
    device=-1
)

set_seed(42)

def paraphrase_prompt(original_prompt: str, num_variations: int = NUM_PARAPHRASES) -> list[str]:
    paraphrased_texts = []
    input_text = f"paraphrase: {original_prompt}"

    for i in range(num_variations):
        generated_output = paraphrase_generator(
            input_text,
            max_length=MAX_PARAPHRASE_LENGTH,
            do_sample=True,
            top_k=50 + i*10,
            temperature=0.7 + i*0.1,
            num_return_sequences=1,
            truncation=True,
            clean_up_tokenization_spaces=True
        )[0]["generated_text"]
        paraphrased_texts.append(generated_output.strip())
    
    unique_paraphrases = []
    seen_texts = {original_prompt.lower()}
    for p in paraphrased_texts:
        if p and p.lower() not in seen_texts:
            unique_paraphrases.append(p)
            seen_texts.add(p.lower())
    
    return unique_paraphrases[:num_variations]

def generate_responses(prompts: list[str]) -> list[str]:
    responses = []
    for prompt in prompts:
        generated_output = response_generator(
            prompt,
            max_new_tokens=MAX_RESPONSE_LENGTH,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            truncation=True,
            clean_up_tokenization_spaces=True
        )[0]["generated_text"]
        
        response_only = generated_output
        if generated_output.startswith(prompt):
            response_only = generated_output[len(prompt):].strip()
        
        response_only = response_only.split('\n')[0].strip()
        if not response_only.endswith(('.', '!', '?')) and ' ' in response_only:
            response_only = response_only.rsplit(' ', 1)[0]
        
        responses.append(response_only.strip())
    return responses

def customer_support_agent(customer_query: str) -> str:
    if not customer_query or not customer_query.strip():
        return "Please enter a customer query."

    paraphrased_prompts = paraphrase_prompt(customer_query, NUM_PARAPHRASES)

    all_prompts_for_response = [customer_query] + paraphrased_prompts

    all_responses = generate_responses(all_prompts_for_response)

    output_str = f"**Original Query:** {customer_query}\n\n"
    output_str += "--- Generated Responses ---\n\n"

    if all_responses:
        output_str += f"**Response based on Original Query:**\n"
        output_str += f"- {all_responses[0]}\n\n"
        
        for i, (paraphrased_p, response_r) in enumerate(zip(paraphrased_prompts, all_responses[1:])):
            output_str += f"**Response based on Paraphrased Query {i+1} (`{paraphrased_p}`):**\n"
            output_str += f"- {response_r}\n\n"
    else:
        output_str += "No responses could be generated.\n\n"

    return output_str

iface = gr.Interface(
    fn=customer_support_agent,
    inputs=gr.Textbox(lines=2, placeholder="Enter customer query here...", label="Customer Query"),
    outputs=gr.Markdown(label="Generated Customer Support Responses"),
    title="Automated Customer Support Response Generator with Paraphrased Prompts",
    description="This application generates multiple diverse customer support responses by first paraphrasing the original query and then generating a response for each variation. This enhances the robustness and diversity of suggested replies."
)

iface.launch()