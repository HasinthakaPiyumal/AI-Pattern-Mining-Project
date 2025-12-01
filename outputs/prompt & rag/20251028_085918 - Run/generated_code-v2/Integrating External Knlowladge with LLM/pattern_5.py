class KnowledgeGraph:
    """
    Simulates a medical Knowledge Graph for retrieving relevant facts.
    In a real-world scenario, this would interact with a graph database
    (e.g., Neo4j, ArangoDB) or a structured medical ontology.
    """
    def __init__(self):
        # Simulate a simple medical KG with a dictionary of facts
        self.medical_facts = {
            "fever": [
                "Fever is a temporary increase in your body temperature, often due to an illness.",
                "Common causes include viral infections (e.g., flu, common cold), bacterial infections (e.g., strep throat), and inflammation.",
                "Treatment often involves rest, fluids, and fever-reducing medications like paracetamol or ibuprofen."
            ],
            "cough": [
                "A cough is a reflex action to clear your airway of irritants and mucus.",
                "Causes can be viral infections (common cold, flu), allergies, asthma, or more serious conditions like bronchitis or pneumonia.",
                "Treatments vary based on the cause; can include cough suppressants, expectorants, or antibiotics for bacterial infections."
            ],
            "sore throat": [
                "A sore throat is pain or irritation of the throat that often worsens when you swallow.",
                "Most sore throats are caused by viral infections, such as the common cold or flu.",
                "Bacterial infections like strep throat also cause sore throats and require antibiotics.",
                "Home remedies include warm liquids, honey, and lozenges."
            ],
            "headache": [
                "A headache is pain in any region of the head.",
                "Common types include tension headaches, migraines, and cluster headaches.",
                "Causes can range from stress and dehydration to more serious conditions.",
                "Pain relievers like aspirin or ibuprofen are common treatments."
            ],
            "fatigue": [
                "Fatigue is extreme tiredness resulting from mental or physical exertion or illness.",
                "It can be a symptom of many underlying conditions, including infections, anemia, thyroid problems, and chronic fatigue syndrome.",
                "Treatment depends on the underlying cause."
            ],
            "pneumonia": [
                "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.",
                "It can be caused by bacteria, viruses, or fungi.",
                "Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
                "Treatment involves antibiotics (for bacterial), antiviral (for viral), and supportive care."
            ],
            "strep throat": [
                "Strep throat is a bacterial infection that can make your throat feel sore and scratchy.",
                "Caused by Streptococcus pyogenes bacteria.",
                "Symptoms include sudden sore throat, pain when swallowing, fever, red spots on the roof of the mouth.",
                "Requires antibiotic treatment to prevent complications."
            ]
        }

    def retrieve_knowledge(self, query: str) -> list[str]:
        """
        Retrieves relevant medical facts from the simulated KG based on a query.
        Matches keywords in the query to known medical conditions or symptoms.
        """
        retrieved_facts = []
        query_keywords = [word.lower() for word in query.split()]

        for keyword, facts in self.medical_facts.items():
            if keyword in query_keywords or any(k in query.lower() for k in keyword.split()):
                retrieved_facts.extend(facts)
        
        # Remove duplicates and return
        return list(set(retrieved_facts))
