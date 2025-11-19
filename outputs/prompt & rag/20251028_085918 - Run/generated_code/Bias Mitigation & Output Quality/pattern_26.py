import random
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CULTURAL_CONTEXTS = {
    "en-US": {
        "greeting": "Hello, how can I assist you with your health query today?",
        "tone": "professional and direct",
        "vocabulary_additions": [],
        "instructions": "Ensure responses are factual and adhere to Western medical standards.",
        "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
    },
    "es-MX": {
        "greeting": "Hola, ¿cómo puedo ayudarte con tu consulta de salud hoy?",
        "tone": "empathetic and supportive",
        "vocabulary_additions": ["paciente", "tratamiento", "síntomas"],
        "instructions": "Be culturally sensitive, use common Spanish medical terms, and acknowledge traditional remedies if relevant (without endorsing them over modern medicine).",
        "disclaimer": "Esta información es sólo para fines educativos y no sustituye el consejo médico profesional."
    },
}

MEDICAL_KNOWLEDGE_BASE = {
    "diabetes management": [
        {"title": "Type 2 Diabetes Guidelines", "content": "Diet, exercise, and medication are key. Regular blood sugar monitoring. Consult a doctor for personalized plan."},
        {"title": "Insulin Therapy Basics", "content": "Different types of insulin, injection sites, dosage adjustments. Requires medical supervision."},
        {"title": "Diabetic Foot Care", "content": "Importance of daily foot checks, proper footwear, avoiding injuries. Seek immediate care for wounds."},
    ],
    "hypertension symptoms": [
        {"title": "High Blood Pressure Overview", "content": "Often no symptoms, hence 'silent killer'. Regular checks are vital. Headaches, dizziness can occur in severe cases."},
        {"title": "Managing Hypertension", "content": "Lifestyle changes (diet, exercise) and medication. Regular monitoring."},
    ],
    "common cold remedies": [
        {"title": "Cold and Flu Treatment", "content": "Rest, fluids, over-the-counter pain relievers. Antibiotics are not effective for viral colds."},
        {"title": "Home Remedies for Cold", "content": "Honey for cough, steam inhalation for congestion. Symptomatic relief."},
    ],
    "headache causes": [
        {"title": "Types of Headaches", "content": "Tension headaches, migraines, cluster headaches. Can be caused by stress, dehydration, eye strain."},
        {"title": "When to worry about a headache", "content": "Sudden severe headache, headache after injury, vision changes, weakness/numbness. Seek immediate medical attention."},
    ]
}

DEMONSTRATIONS_DATA = [
    {"query": "Patient with persistent cough, fever, and fatigue. Possible diagnosis?", "response": "Could indicate a respiratory infection like bronchitis or pneumonia. Recommend chest X-ray and sputum culture.", "demographics": {"age": "adult", "ethnicity": "caucasian", "region": "urban"}},
    {"query": "Elderly patient with sudden chest pain radiating to left arm. What to do?", "response": "Suggest immediate medical evaluation for potential myocardial infarction. Administer aspirin if not contraindicated.", "demographics": {"age": "elderly", "ethnicity": "asian", "region": "rural"}},
    {"query": "Child with skin rash and itching. What are common causes?", "response": "Common causes include allergies, eczema, or viral exanthems. Consider recent exposures and diet changes.", "demographics": {"age": "child", "ethnicity": "african", "region": "suburban"}},
    {"query": "Pregnant woman experiencing nausea and vomiting. Management strategies?", "response": "Recommend small, frequent meals, ginger, and staying hydrated. Consult obstetrician for antiemetics if severe.", "demographics": {"age": "adult", "ethnicity": "hispanic", "region": "urban"}},
]

class LLMSimulator:
    def __init__(self, model_name: str = "SimulatedLLM"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        logging.info(f"LLM simulating generation for prompt: {prompt[:100]}...")
        if "diagnosis" in prompt.lower() and "cough" in prompt.lower():
            return "Based on symptoms, a respiratory infection is possible. Further tests are recommended."
        elif "chest pain" in prompt.lower():
            return "Immediate medical attention is crucial for chest pain. Consider cardiac events."
        elif "cultural context" in prompt.lower() and "diabetes" in prompt.lower():
            return "Managing diabetes involves dietary adjustments. In some cultures, traditional herbal remedies are used, but always consult a doctor before combining with prescribed medication."
        elif "synthetic patient data" in prompt.lower():
            return "Generated synthetic patient: Name: John Doe, Age: 45, Ethnicity: Caucasian, Condition: Hypertension. (This is a placeholder and would be more complex in a real scenario)"
        elif "arguments for" in prompt.lower() and "arguments against" in prompt.lower():
            return "Arguments for X: [reason 1, reason 2]. Arguments against X: [reason A, reason B]."
        else:
            return "I am a simulated AI assistant and can provide general information. For specific medical advice, please consult a healthcare professional."

class PromptEngineer:
    def __init__(self, llm_simulator: LLMSimulator):
        self.llm = llm_simulator

    def _select_balanced_demonstrations(self, num_demos: int = 2, target_demographics: dict = None) -> list:
        if target_demographics:
            balanced_demos = []
            for demo in DEMONSTRATIONS_DATA:
                if all(demo["demographics"].get(k) == v for k, v in target_demographics.items()):
                    balanced_demos.append(demo)
                if len(balanced_demos) >= num_demos:
                    break
            if len(balanced_demos) < num_demos:
                remaining_needed = num_demos - len(balanced_demos)
                available_demos = [d for d in DEMONSTRATIONS_DATA if d not in balanced_demos]
                balanced_demos.extend(random.sample(available_demos, min(remaining_needed, len(available_demos))))
            return balanced_demos
        else:
            if len(DEMONSTRATIONS_DATA) <= num_demos:
                return DEMONSTRATIONS_DATA
            return random.sample(DEMONSTRATIONS_DATA, num_demos)

    def _apply_cultural_awareness(self, prompt: str, locale: str) -> str:
        context = CULTURAL_CONTEXTS.get(locale, CULTURAL_CONTEXTS["en-US"])
        cultural_prefix = f"You are assisting in a {context['tone']} manner. {context['instructions']} "
        if context['vocabulary_additions']:
            cultural_prefix += f"Use medical terms like: {', '.join(context['vocabulary_additions'])}. "
        return f"{cultural_prefix}\n\n{prompt}"

    def _demonstration_ensembling(self, base_prompt: str, demonstrations: list) -> str:
        ensembled_prompt_parts = []
        for i, demo in enumerate(demonstrations):
            ensembled_prompt_parts.append(f"Example {i+1} - Patient Query: {demo['query']}\nExample {i+1} - AI Response: {demo['response']}")
        
        return f"{base_prompt}\n\n{'--- Demonstrations ---\n' + '\n'.join(ensembled_prompt_parts) if ensembled_prompt_parts else ''}"

    def _debate_style_aggregation(self, query: str) -> str:
        logging.info(f"Generating debate-style aggregation for query: {query}")
        pro_args_prompt = f"Provide arguments in favor of the claim: '{query}'."
        con_args_prompt = f"Provide arguments against the claim: '{query}'."

        pro_response = self.llm.generate(pro_args_prompt)
        con_response = self.llm.generate(con_args_prompt)

        return f"Debate on '{query}':\n\nArguments For: {pro_response}\n\nArguments Against: {con_response}"

    def construct_and_execute_prompt(
        self,
        user_query: str,
        locale: str = "en-US",
        use_dense: bool = True,
        use_balanced_demos: bool = True,
        target_demographics: dict = None,
        use_debate_aggregation: bool = False
    ) -> str:
        base_prompt = f"User query: {user_query}"
        
        prompt_with_cultural_context = self._apply_cultural_awareness(base_prompt, locale)

        demonstrations = []
        if use_balanced_demos:
            demonstrations = self._select_balanced_demonstrations(target_demographics=target_demographics)
            logging.info(f"Selected balanced demonstrations: {[d['query'] for d in demonstrations]}")

        final_prompt = prompt_with_cultural_context
        if use_dense:
            final_prompt = self._demonstration_ensembling(final_prompt, demonstrations)

        llm_response = self.llm.generate(final_prompt)

        if use_debate_aggregation:
            debate_output = self._debate_style_aggregation(user_query)
            llm_response = f"{llm_response}\n\n--- Debate Analysis ---\n{debate_output}"

        return llm_response

class KnowledgeBaseManager:
    def __init__(self, knowledge_base: dict):
        self.knowledge_base = knowledge_base

    def retrieve_medical_evidence(self, query: str) -> list:
        logging.info(f"Retrieving evidence for query: {query}")
        relevant_docs = []
        query_lower = query.lower()
        for topic, articles in self.knowledge_base.items():
            if topic.lower() in query_lower or any(word in topic.lower() for word in query_lower.split()):
                for article in articles:
                    relevant_docs.append(f"Source: {article['title']}\nContent: {article['content']}")
        
        for topic, articles in self.knowledge_base.items():
             for article in articles:
                 if any(word in article['content'].lower() for word in query_lower.split()):
                     if f"Source: {article['title']}\nContent: {article['content']}" not in relevant_docs:
                         relevant_docs.append(f"Source: {article['title']}\nContent: {article['content']}")
        
        return relevant_docs[:2]


class BiasMitigationAndSyntheticData:
    def __init__(self, llm_simulator: LLMSimulator):
        self.llm = llm_simulator

    def generate_synthetic_patient_data(self, attributes: dict) -> dict:
        logging.info(f"Generating synthetic data with attributes: {attributes}")
        
        age = attributes.get("age", random.randint(18, 80))
        ethnicity = attributes.get("ethnicity", random.choice(["Caucasian", "African American", "Asian", "Hispanic"]))
        condition = attributes.get("condition", random.choice(["Hypertension", "Diabetes", "Asthma", "Allergies"]))
        gender = attributes.get("gender", random.choice(["Male", "Female", "Other"]))
        
        llm_prompt = f"Generate a detailed synthetic patient profile with the following characteristics: Age: {age}, Ethnicity: {ethnicity}, Condition: {condition}, Gender: {gender}. Focus on diverse attributes to avoid bias."
        
        llm_response_text = self.llm.generate(llm_prompt)
        
        synthetic_patient = {
            "name": f"Synthetic Patient {random.randint(100, 999)}",
            "age": age,
            "ethnicity": ethnicity,
            "gender": gender,
            "condition": condition,
            "details_from_llm": llm_response_text
        }
        return synthetic_patient

    def analyze_and_mitigate_bias(self, llm_response: str) -> str:
        logging.info("Analyzing LLM response for potential bias...")
        mitigated_response = llm_response
        
        if "only recommends male doctors" in llm_response.lower() or "suggests traditional remedies over modern medicine without proper context" in llm_response.lower():
            logging.warning("Potential bias detected! Attempting to mitigate.")
            mitigated_response = llm_response.replace("only recommends male doctors", "recommends qualified doctors")
            mitigated_response = mitigated_response.replace("suggests traditional remedies over modern medicine without proper context", "suggests consulting a doctor before combining traditional remedies with modern medicine")
            mitigated_response += "\n[Bias mitigation applied: Ensured balanced recommendations and contextualized traditional remedies.]"
        elif "recommends only Western treatments" in llm_response.lower():
             logging.warning("Potential Western bias detected! Attempting to mitigate.")
             mitigated_response = llm_response.replace("recommends only Western treatments", "recommends evidence-based treatments, considering global practices where appropriate")
             mitigated_response += "\n[Bias mitigation applied: Expanded treatment recommendations to be globally inclusive.]"
        else:
            logging.info("No significant bias detected in this response (simulated check).")

        return mitigated_response

class GlobalHealthAssistant:
    def __init__(self):
        self.llm_simulator = LLMSimulator()
        self.prompt_engineer = PromptEngineer(self.llm_simulator)
        self.kb_manager = KnowledgeBaseManager(MEDICAL_KNOWLEDGE_BASE)
        self.bias_manager = BiasMitigationAndSyntheticData(self.llm_simulator)
        
    def process_health_query(
        self,
        query: str,
        locale: str = "en-US",
        patient_demographics: dict = None,
        enable_dense: bool = True,
        enable_balanced_demos: bool = True,
        enable_debate_aggregation: bool = False
    ) -> dict:
        logging.info(f"Processing query: '{query}' for locale: '{locale}'")

        retrieved_evidence = self.kb_manager.retrieve_medical_evidence(query)
        evidence_str = "\n".join(retrieved_evidence)
        
        full_query_with_evidence = f"{query}\n\n--- Retrieved Medical Evidence ---\n{evidence_str}"
        if not retrieved_evidence:
            full_query_with_evidence = query

        raw_llm_response = self.prompt_engineer.construct_and_execute_prompt(
            user_query=full_query_with_evidence,
            locale=locale,
            use_dense=enable_dense,
            use_balanced_demos=enable_balanced_demos,
            target_demographics=patient_demographics,
            use_debate_aggregation=enable_debate_aggregation
        )

        final_response = self.bias_manager.analyze_and_mitigate_bias(raw_llm_response)

        logging.info(f"Query: {query}, Response: {final_response[:200]}...")
        
        cultural_disclaimer = CULTURAL_CONTEXTS.get(locale, CULTURAL_CONTEXTS["en-US"])['disclaimer']

        return {
            "query": query,
            "locale": locale,
            "response": final_response,
            "retrieved_evidence": retrieved_evidence,
            "disclaimer": cultural_disclaimer,
            "monitoring_status": "Logged for continuous improvement"
        }

    def generate_patient_data_for_training(self, attributes: dict) -> dict:
        return self.bias_manager.generate_synthetic_patient_data(attributes)

def main():
    print("Initializing Global Health Assistant AI...")
    assistant = GlobalHealthAssistant()

    print("\n--- Testing Health Query (English, default settings) ---")
    query_result_en = assistant.process_health_query(
        query="What are the symptoms and management for high blood pressure?",
        locale="en-US"
    )
    print(json.dumps(query_result_en, indent=2))

    print("\n--- Testing Health Query (Spanish, culturally aware, balanced demos) ---")
    query_result_es = assistant.process_health_query(
        query="Un paciente anciano con tos persistente y fiebre. ¿Qué podría ser?",
        locale="es-MX",
        patient_demographics={"age": "elderly", "ethnicity": "hispanic"},
        enable_balanced_demos=True
    )
    print(json.dumps(query_result_es, indent=2))

    print("\n--- Testing Health Query (English, debate style aggregation) ---")
    query_result_debate = assistant.process_health_query(
        query="Is vitamin C effective in preventing common cold?",
        locale="en-US",
        enable_debate_aggregation=True
    )
    print(json.dumps(query_result_debate, indent=2))

    print("\n--- Testing Synthetic Data Generation ---")
    synthetic_patient = assistant.generate_patient_data_for_training(
        attributes={"age": 60, "ethnicity": "African American", "condition": "Diabetes"}
    )
    print(json.dumps(synthetic_patient, indent=2))

if __name__ == "__main__":
    main()