class KnowledgeRetriever:
    def __init__(self):
        self.pubmed_mock_db = {
            "fever and cough": [
                "Patient presents with high fever and persistent cough. Possible viral infection. Consider influenza or common cold.",
                "Cough can be a symptom of various respiratory conditions, including bronchitis and pneumonia. Fever often accompanies bacterial infections."
            ],
            "headache and nausea": [
                "Headache accompanied by nausea may indicate migraine or gastrointestinal issues. Assess other neurological symptoms.",
                "Severe headaches can be a sign of tension, cluster headaches, or even more serious conditions like meningitis."
            ]
        }
        self.clinical_guidelines_mock_db = {
            "viral infection treatment": [
                "For viral infections, recommend rest, hydration, and symptomatic relief. Antivirals may be considered in specific cases like severe influenza.",
                "Antibiotics are ineffective against viral infections."
            ],
            "migraine management": [
                "Migraine management includes acute treatment (triptans, NSAIDs) and preventive therapies (beta-blockers, topiramate).",
                "Avoid triggers like certain foods or stress."
            ]
        }

    def retrieve_evidence(self, query):
        print(f"Retrieving evidence for query: '{query}'")
        raw_evidence = []
        for key, articles in self.pubmed_mock_db.items():
            if all(word in query.lower() for word in key.split(' and ')):
                raw_evidence.extend(articles)
        for key, guidelines in self.clinical_guidelines_mock_db.items():
            if all(word in query.lower() for word in key.split(' and ')):
                raw_evidence.extend(guidelines)
        return " ".join(raw_evidence) if raw_evidence else "No direct evidence found."

class EntityLinker:
    def __init__(self):
        self.knowledge_graph = {
            "fever": "Fever is an elevated body temperature and is a common sign of illness.",
            "cough": "A cough is a reflex action to clear the airways of mucus or irritants.",
            "viral infection": "A viral infection is a proliferation of a harmful virus inside the body.",
            "influenza": "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses.",
            "pneumonia": "Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.",
            "headache": "A headache is a pain in any region of the head.",
            "nausea": "Nausea is a sensation of unease and discomfort in the upper stomach with an involuntary urge to vomit.",
            "migraine": "A migraine is a primary headache disorder characterized by recurrent headaches that are moderate to severe.",
            "antibiotics": "Antibiotics are a type of antimicrobial designed to target and kill or stop the growth of bacteria."
        }

    def link_entities(self, raw_evidence):
        print(f"Linking entities in raw evidence...")
        linked_evidence = raw_evidence
        for entity, description in self.knowledge_graph.items():
            if entity in raw_evidence.lower():
                linked_evidence = linked_evidence.replace(entity, f"{entity} ({description})")
        return linked_evidence

class EvidenceChainer:
    def synthesize_evidence(self, linked_evidence, query_keywords):
        print(f"Synthesizing evidence for query keywords: '{', '.join(query_keywords)}'")
        sentences = linked_evidence.split('. ')
        pertinent_sentences = []
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in query_keywords):
                pertinent_sentences.append(sentence)
        
        if not pertinent_sentences:
            return "No highly relevant evidence chains could be formed."

        evidence_chain = " ".join(pertinent_sentences).strip()
        if not evidence_chain.endswith('.'):
            evidence_chain += '.'
        return evidence_chain

# --- Main Application Flow ---
if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    linker = EntityLinker()
    chainer = EvidenceChainer()

    # Example 1: Patient with fever and cough
    patient_query_1 = "patient has fever and cough"
    query_keywords_1 = ["fever", "cough", "viral infection"]
    print(f"\n--- Processing Query: {patient_query_1} ---")

    raw_evidence_1 = retriever.retrieve_evidence(patient_query_1)
    print(f"Raw Evidence:\n{raw_evidence_1}\n")

    linked_evidence_1 = linker.link_entities(raw_evidence_1)
    print(f"Linked Evidence:\n{linked_evidence_1}\n")

    final_evidence_chain_1 = chainer.synthesize_evidence(linked_evidence_1, query_keywords_1)
    print(f"Final Evidence Chain for LLM:\n{final_evidence_chain_1}\n")

    # Example 2: Patient with headache and nausea
    patient_query_2 = "severe headache and nausea"
    query_keywords_2 = ["headache", "nausea", "migraine"]
    print(f"\n--- Processing Query: {patient_query_2} ---")

    raw_evidence_2 = retriever.retrieve_evidence(patient_query_2)
    print(f"Raw Evidence:\n{raw_evidence_2}\n")

    linked_evidence_2 = linker.link_entities(raw_evidence_2)
    print(f"Linked Evidence:\n{linked_evidence_2}\n")

    final_evidence_chain_2 = chainer.synthesize_evidence(linked_evidence_2, query_keywords_2)
    print(f"Final Evidence Chain for LLM:\n{final_evidence_chain_2}\n")

    # Example 3: Query with no direct match
    patient_query_3 = "swollen ankle pain"
    query_keywords_3 = ["ankle", "swollen"]
    print(f"\n--- Processing Query: {patient_query_3} ---")

    raw_evidence_3 = retriever.retrieve_evidence(patient_query_3)
    print(f"Raw Evidence:\n{raw_evidence_3}\n")

    linked_evidence_3 = linker.link_entities(raw_evidence_3)
    print(f"Linked Evidence:\n{linked_evidence_3}\n")

    final_evidence_chain_3 = chainer.synthesize_evidence(linked_evidence_3, query_keywords_3)
    print(f"Final Evidence Chain for LLM:\n{final_evidence_chain_3}\n")
