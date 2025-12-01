class MedicalKnowledgeGraphRetrievalModule:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def retrieve_medical_facts(self, query_keywords: list) -> str:
        found_facts = []
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            for category, items in self.knowledge_graph.items():
                if keyword_lower in category.lower():
                    found_facts.append(f"Category: {category}. Information: {items}")
                    continue
                if isinstance(items, dict):
                    for item_name, item_details in items.items():
                        if keyword_lower in item_name.lower():
                            found_facts.append(f"Disease/Symptom: {item_name}. Details: {item_details}")
                        elif isinstance(item_details, str) and keyword_lower in item_details.lower():
                             found_facts.append(f"Disease/Symptom: {item_name}. Details: {item_details}")
                elif isinstance(items, list):
                    for item in items:
                        if keyword_lower in item.lower():
                            found_facts.append(f"Category: {category}. Item: {item}")

        if not found_facts:
            return "No specific medical facts found in the knowledge graph for your query." 
        return "\n".join(list(set(found_facts)))

def simulated_llm(prompt: str) -> str:
    if "headache" in prompt.lower() and "fever" in prompt.lower():
        return "Based on the context, symptoms like headache and fever could indicate a viral infection or flu. Please consult a doctor for a proper diagnosis."
    elif "diabetes" in prompt.lower() and "insulin" in prompt.lower():
        return "The information suggests that insulin is a key treatment for diabetes, helping to regulate blood sugar levels. Regular monitoring and dietary management are also crucial."
    elif "cancer" in prompt.lower() and "chemotherapy" in prompt.lower():
        return "Chemotherapy is a common treatment for cancer, using drugs to destroy cancer cells. The specific type and regimen depend on the cancer type and stage. Further consultation with an oncologist is recommended."
    elif "no specific medical facts found" in prompt.lower():
        return "I couldn't find very specific medical facts for that query in my current knowledge. Please provide more details or ask a different question. However, I can still try to give a general answer based on your query."
    else:
        return f"Understood. Based on your query and the provided context, a general response would be: {prompt.split('Based on the user\'s query and the provided medical context, please provide a diagnostic suggestion or explanation:')[-1].strip()}"

class LLMAdapter:
    def __init__(self, retrieval_module, llm_function):
        self.retrieval_module = retrieval_module
        self.llm_function = llm_function

    def process_query(self, user_query: str) -> str:
        query_keywords = user_query.lower().replace(",", "").split() 
        retrieved_facts = self.retrieval_module.retrieve_medical_facts(query_keywords)

        augmented_prompt = (
            f"User Query: {user_query}\n\n"
            f"Additional Medical Context: {retrieved_facts}\n\n"
            "Based on the user's query and the provided medical context, please provide a diagnostic suggestion or explanation:"
        )
        llm_response = self.llm_function(augmented_prompt)
        return llm_response

if __name__ == "__main__":
    # Simulated Medical Knowledge Graph
    medical_knowledge_graph = {
        "Diseases": {
            "Fever": "An elevated body temperature, often a symptom of infection or inflammation.",
            "Headache": "Pain in the head or face, common causes include stress, fatigue, or illness.",
            "Influenza (Flu)": "A viral infection that attacks the respiratory system (nose, throat, lungs). Symptoms include fever, body aches, headache, and fatigue.",
            "Common Cold": "A viral infectious disease of the upper respiratory tract that primarily affects the nose. Symptoms include cough, sore throat, runny nose, and fever.",
            "Diabetes Mellitus": "A chronic condition that affects the way your body processes blood sugar (glucose). Type 1 involves the immune system attacking insulin-producing cells; Type 2 involves insulin resistance.",
            "Cancer": "A disease caused by an uncontrolled division of abnormal cells in a part of the body."
        },
        "Symptoms": [
            "High temperature", "Chills", "Body aches", "Sore throat", "Cough", 
            "Fatigue", "Nausea", "Vomiting", "Diarrhea", "Shortness of breath",
            "Frequent urination", "Increased thirst", "Unexplained weight loss", "Blurred vision"
        ],
        "Treatments": {
            "Paracetamol": "A common pain reliever and fever reducer.",
            "Ibuprofen": "An NSAID used to reduce pain, fever, and inflammation.",
            "Insulin": "A hormone that regulates the amount of glucose (sugar) in the blood, often used in diabetes management.",
            "Chemotherapy": "Drug treatment that uses powerful chemicals to kill fast-growing cells in your body.",
            "Antibiotics": "Medications that fight bacterial infections."
        },
        "Drug Interactions": {
            "Ibuprofen + Warfarin": "Increased risk of bleeding.",
            "Insulin + Beta-blockers": "Can mask symptoms of hypoglycemia."
        }
    }

    retrieval_module = MedicalKnowledgeGraphRetrievalModule(medical_knowledge_graph)
    adapter = LLMAdapter(retrieval_module, simulated_llm)

    print("Medical Diagnostic Assistant (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = adapter.process_query(user_input)
        print("Assistant:", response)
