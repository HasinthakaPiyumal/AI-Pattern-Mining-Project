import gradio as gr

# 1. Simulated Medical Knowledge Graph (KG)
# In a real application, this would be a robust graph database.
medical_knowledge_graph = {
    "fever": [
        "Fever is often a symptom of infection.",
        "Common causes of fever include viral infections (e.g., flu, common cold) and bacterial infections (e.g., strep throat, pneumonia).",
        "Treatment for fever often involves antipyretics like acetaminophen or ibuprofen.",
        "Persistent high fever may indicate a serious underlying condition."
    ],
    "cough": [
        "Cough can be a symptom of respiratory infections (e.g., bronchitis, pneumonia, common cold).",
        "Allergies or asthma can also cause chronic cough.",
        "Productive cough may indicate mucus clearance, while dry cough can be irritating.",
        "Smoking is a significant risk factor for chronic cough."
    ],
    "sore throat": [
        "Sore throat is a common symptom of pharyngitis, often caused by viral infections.",
        "Bacterial causes like Strep throat (Streptococcus pyogenes) require antibiotic treatment.",
        "Symptoms include pain, difficulty swallowing, and sometimes swollen tonsils."
    ],
    "headache": [
        "Headaches can be tension headaches, migraines, or cluster headaches.",
        "Secondary headaches can be caused by underlying conditions like sinus infections or more serious issues.",
        "Dehydration and stress are common triggers for headaches."
    ],
    "fatigue": [
        "Fatigue is a general symptom that can be associated with many conditions.",
        "Lack of sleep, stress, poor diet, and underlying medical conditions like anemia or thyroid issues can cause fatigue."
    ],
    "strep throat": [
        "Strep throat is a bacterial infection of the throat and tonsils.",
        "Symptoms include sudden sore throat, pain when swallowing, fever, red and swollen tonsils, sometimes with white patches.",
        "Diagnosis is typically confirmed by a rapid strep test or throat culture.",
        "Treatment involves antibiotics, usually penicillin or amoxicillin, to prevent complications like rheumatic fever."
    ],
    "viral infection": [
        "Viral infections are caused by viruses and often resolve on their own.",
        "Antibiotics are ineffective against viral infections.",
        "Symptomatic treatment (rest, fluids, fever reducers) is common."
    ],
    "pneumonia": [
        "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.",
        "Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
        "Can be caused by bacteria, viruses, or fungi.",
        "Bacterial pneumonia is often treated with antibiotics."
    ]
}

# 2. Knowledge Retrieval Module
def retrieve_knowledge(query: str) -> list:
    retrieved_facts = []
    query_lower = query.lower()
    for keyword, facts in medical_knowledge_graph.items():
        if keyword in query_lower:
            retrieved_facts.extend(facts)
    return list(set(retrieved_facts)) # Remove duplicates

# 3. Simulated Large Language Model (LLM) Integration
# In a real application, this would involve API calls to a powerful LLM.
def simulate_llm(prompt: str) -> str:
    # This is a very basic simulation. A real LLM would generate dynamic, context-aware text.
    if "fever" in prompt.lower() and "sore throat" in prompt.lower() and "white patches" in prompt.lower() and "strep throat" in prompt.lower():
        reasoning_template = """Chain of Thought:
1. The patient presents with symptoms including fever, sore throat, and pain when swallowing.
2. Retrieved knowledge indicates that a sore throat with fever and white patches on tonsils is highly suggestive of Strep throat.
3. Strep throat is a bacterial infection caused by Streptococcus pyogenes.
4. Diagnosis typically requires a rapid strep test or throat culture.
5. Treatment involves antibiotics to prevent complications.
"""
        recommendation_template = "Recommendation: Consider Strep throat. Recommend a rapid strep test or throat culture for confirmation, followed by appropriate antibiotic treatment if positive (e.g., penicillin or amoxicillin). Also advise symptomatic relief for fever and pain."
    elif "fever" in prompt.lower() and "cough" in prompt.lower():
        reasoning_template = """Chain of Thought:
1. The patient presents with common symptoms of respiratory infection: fever and cough.
2. Retrieved knowledge suggests that fever and cough are common in viral infections (like flu or common cold) or bacterial infections (like bronchitis or pneumonia).
3. Viral infections often resolve spontaneously, while bacterial infections may require antibiotics.
"""
        recommendation_template = "Recommendation: Symptoms suggest a respiratory infection. Differentiate between viral and bacterial causes. Recommend rest, fluids, and antipyretics for symptomatic relief. If symptoms worsen or persist, further diagnostic tests (e.g., chest X-ray, lab tests) may be needed to rule out pneumonia or other bacterial infections."
    elif "headache" in prompt.lower() and "fatigue" in prompt.lower():
        reasoning_template = """Chain of Thought:
1. The patient reports headache and fatigue, which are general and common symptoms.
2. Retrieved knowledge indicates that headaches can be caused by tension, migraines, or secondary conditions, and fatigue can be due to lack of sleep, stress, or underlying medical conditions.
3. Without more specific symptoms or patient history, it is difficult to pinpoint a precise diagnosis.
"""
        recommendation_template = "Recommendation: Investigate lifestyle factors like sleep, stress, and hydration. Suggest over-the-counter pain relief for headache and advise rest. If symptoms persist or worsen, further evaluation to identify underlying causes, such as anemia or thyroid issues, is recommended."
    else:
        reasoning_template = """Chain of Thought:
1. The provided symptoms are general.
2. External knowledge has been retrieved, but more specific information is needed to form a detailed reasoning path.
3. A general approach to common ailments is necessary.
"""
        recommendation_template = "Recommendation: Advise symptomatic relief and monitor for any new or worsening symptoms. If concerns persist, further medical consultation is recommended for a comprehensive assessment."

    # Integrate retrieved knowledge into the response for demonstration
    # In a real LLM, the model would naturally weave this in.
    knowledge_marker = "Knowledge from medical graph:"
    if knowledge_marker in prompt:
        start_index = prompt.find(knowledge_marker) + len(knowledge_marker)
        end_index = prompt.find("Instructions:")
        retrieved_info_in_prompt = prompt[start_index:end_index].strip()
        reasoning_template = reasoning_template.replace("Retrieved knowledge indicates that", f"Retrieved knowledge ({retrieved_info_in_prompt}) indicates that")

    return f"{reasoning_template}\n\n{recommendation_template}"

# 4. Knowledge-Driven Chain-of-Thought (KDCoT) Orchestrator
def kdcot_process(user_query: str) -> tuple[str, str, str]:
    # 1. Retrieve Relevant Knowledge
    retrieved_facts = retrieve_knowledge(user_query)
    retrieved_facts_str = "\n- ".join(retrieved_facts)

    # 2. Construct Sophisticated Prompt for LLM
    prompt = f"""Patient Symptoms/Query: {user_query}

Knowledge from medical graph:
- {retrieved_facts_str}

Instructions: Based ONLY on the provided patient query and the knowledge from the medical graph, generate a step-by-step Chain of Thought leading to a diagnosis or recommendation. Explicitly reference the provided knowledge where applicable in your reasoning. Finally, provide a clear diagnostic or treatment recommendation.

Chain of Thought and Recommendation:
"""

    # 3. LLM Interaction
    llm_response = simulate_llm(prompt)

    # 4. Parse LLM's Response
    reasoning_start = llm_response.find("Chain of Thought:")
    recommendation_start = llm_response.find("Recommendation:")

    if reasoning_start != -1 and recommendation_start != -1:
        reasoning = llm_response[reasoning_start:recommendation_start].strip()
        recommendation = llm_response[recommendation_start:].strip()
    else:
        reasoning = "Could not parse detailed reasoning from LLM response."
        recommendation = llm_response.strip() # Fallback to entire response as recommendation

    return retrieved_facts_str, reasoning, recommendation

# 5. User Interface (UI) with Gradio
if __name__ == "__main__":
    interface = gr.Interface(
        fn=kdcot_process,
        inputs=gr.Textbox(lines=5, label="Enter Patient Symptoms and History (e.g., 'Patient has a high fever, sore throat with white patches, and difficulty swallowing.')"),
        outputs=[
            gr.Textbox(label="Retrieved Medical Knowledge"),
            gr.Textbox(label="LLM's Knowledge-Driven Chain-of-Thought Reasoning"),
            gr.Textbox(label="LLM's Diagnostic/Treatment Recommendation")
        ],
        title="Intelligent Clinical Decision Support System (KDCoT)",
        description="Enter patient symptoms and history to receive a knowledge-driven diagnostic reasoning and recommendation."
    )
    interface.launch(share=True)
