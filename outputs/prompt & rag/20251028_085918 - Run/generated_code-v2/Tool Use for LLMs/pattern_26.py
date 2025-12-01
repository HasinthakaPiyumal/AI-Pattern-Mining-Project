import networkx as nx
import re

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_medical_fact(self, entity1, relation, entity2):
        self.graph.add_edge(entity1, entity2, relation=relation)

    def search_kg(self, query_keywords, max_depth):
        relevant_facts = set()
        found_nodes = set()

        for keyword in query_keywords:
            for node in self.graph.nodes:
                if keyword.lower() in str(node).lower():
                    found_nodes.add(node)

        if not found_nodes:
            return None

        for start_node in found_nodes:
            for depth in range(1, max_depth + 1):
                for source, target, data in nx.dfs_edges(self.graph, start_node, depth_limit=depth):
                    relevant_facts.add((source, data['relation'], target))

        if not relevant_facts:
            return None
        return list(relevant_facts)

class LargeLanguageModel:
    def query_llm_with_kg_context(self, context, question):
        if context:
            context_str = "; ".join([f"{s} {r} {t}" for s, r, t in context])
            return f"[KG-Backed] Based on knowledge: {context_str}. Regarding '{question}', my analysis suggests... (LLM elaboration on KG data)"
        return "[KG-Backed] I can't provide a specific answer based on the provided context."

    def query_llm_inherent_knowledge(self, question):
        return f"[Inherent Knowledge] Based on my general medical knowledge, regarding '{question}', I can tell you that... (LLM's internal knowledge response)"

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.llm = LargeLanguageModel()
        self.MAX_KG_DEPTH = 2

        # Populate KG with some sample medical facts
        self.kg.add_medical_fact("Fever", "is_symptom_of", "Flu")
        self.kg.add_medical_fact("Cough", "is_symptom_of", "Flu")
        self.kg.add_medical_fact("Flu", "is_treated_by", "Antivirals")
        self.kg.add_medical_fact("Flu", "causes", "Fatigue")
        self.kg.add_medical_fact("Headache", "is_symptom_of", "Migraine")
        self.kg.add_medical_fact("Migraine", "is_treated_by", "Triptans")
        self.kg.add_medical_fact("Aspirin", "interacts_with", "Warfarin")
        self.kg.add_medical_fact("Diabetes", "requires", "Insulin")
        self.kg.add_medical_fact("High Blood Pressure", "is_risk_factor_for", "Heart Disease")

    def _extract_keywords(self, query):
        # A simple keyword extraction using regex for demonstration
        # In a real system, spaCy or similar NLP would be used.
        keywords = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        return list(set(keywords))

    def diagnose_patient(self, query):
        keywords = self._extract_keywords(query)
        print(f"\nProcessing query: '{query}'")
        print(f"Extracted keywords: {keywords}")

        kg_data = self.kg.search_kg(keywords, self.MAX_KG_DEPTH)

        if kg_data:
            print(f"KG found relevant information: {kg_data}")
            answer = self.llm.query_llm_with_kg_context(kg_data, query)
            return f"KG-backed Response: {answer}"
        else:
            print("KG could not find sufficient information within limits. Falling back to LLM inherent knowledge.")
            answer = self.llm.query_llm_inherent_knowledge(query)
            return f"LLM Fallback Response: {answer}"


def main():
    assistant = MedicalDiagnosticAssistant()

    print("\n--- Medical Diagnostic Assistant (Type 'exit' to quit) ---")
    while True:
        user_query = input("\nEnter your medical query: ")
        if user_query.lower() == 'exit':
            break

        response = assistant.diagnose_patient(user_query)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()