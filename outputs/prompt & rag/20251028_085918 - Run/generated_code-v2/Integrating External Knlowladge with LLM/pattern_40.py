import requests
import spacy
import networkx as nx

# Simulated external APIs/Databases
def _simulate_pubmed_api(query):
    if "diabetes" in query.lower():
        return "Article: New insights into Type 2 Diabetes management. \nJournal: NEJM. \nKey finding: SGLT2 inhibitors improve cardiovascular outcomes."
    if "hypertension" in query.lower():
        return "Article: Guidelines for Hypertension treatment. \nJournal: Lancet. \nKey finding: Lifestyle modifications are first-line."
    return f"No specific literature found for '{query}'."

def _simulate_ehr_api(patient_id):
    if patient_id == "P1001":
        return "Patient ID: P1001. \nAge: 62. \nSymptoms: High blood pressure, frequent urination. \nLab Results: Fasting glucose 180 mg/dL, HbA1c 8.5%. \nMedications: Metformin."
    return f"Patient ID '{patient_id}' not found."

def _simulate_drug_db_api(drug_name):
    if drug_name.lower() == "metformin":
        return "Drug: Metformin. \nClass: Biguanide. \nUsage: Type 2 Diabetes. \nSide effects: Nausea, diarrhea. \nInteractions: Alcohol, Cimetidine."
    if drug_name.lower() == "lisinopril":
        return "Drug: Lisinopril. \nClass: ACE inhibitor. \nUsage: Hypertension, Heart failure. \nSide effects: Cough, dizziness. \nInteractions: Potassium-sparing diuretics."
    return f"Drug '{drug_name}' not found."

# Simulated Medical Ontology
MEDICAL_ONTOLOGY = {
    "diabetes": "A chronic metabolic disease characterized by high blood sugar levels.",
    "type 2 diabetes": "A form of diabetes where the body either doesn't produce enough insulin, or it resists insulin.",
    "hypertension": "A condition in which the blood vessels have persistently raised pressure.",
    "metformin": "An oral anti-diabetic drug that lowers blood glucose.",
    "sglt2 inhibitors": "A class of drugs used to treat type 2 diabetes by promoting glucose excretion in urine.",
    "lisinopril": "An ACE inhibitor used to treat high blood pressure and heart failure.",
    "high blood pressure": "Common term for hypertension.",
    "fasting glucose": "A blood test that measures blood sugar after an overnight fast.",
    "hba1c": "A blood test that provides an average of blood sugar control over the past 2-3 months.",
    "frequent urination": "A common symptom of uncontrolled diabetes.",
    "cardiovascular outcomes": "Health results related to the heart and blood vessels."
}

class KnowledgeRetriever:
    def __init__(self):
        pass

    def retrieve_medical_literature(self, query):
        return _simulate_pubmed_api(query)

    def retrieve_ehr_data(self, patient_id):
        return _simulate_ehr_api(patient_id)

    def retrieve_drug_info(self, drug_name):
        return _simulate_drug_db_api(drug_name)

    def retrieve_knowledge(self, query, patient_id=None, drug_names=None):
        evidence = []
        evidence.append(self.retrieve_medical_literature(query))
        if patient_id:
            evidence.append(self.retrieve_ehr_data(patient_id))
        if drug_names:
            for drug in drug_names:
                evidence.append(self.retrieve_drug_info(drug))
        return [e for e in evidence if e and "No specific literature found" not in e and "not found" not in e]

class EntityLinker:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def identify_entities(self, text):
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "DATE", "TIME", "NORP", "FAC", "LOC", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]]
        # Add custom medical terms
        medical_terms = [term for term in MEDICAL_ONTOLOGY if term.lower() in text.lower()]
        return list(set(entities + medical_terms))

    def link_entity_to_ontology(self, entity):
        return MEDICAL_ONTOLOGY.get(entity.lower(), None)

    def build_evidence_graph(self, evidence_list):
        graph = nx.Graph()
        all_entities = set()
        for text in evidence_list:
            entities_in_text = self.identify_entities(text)
            for entity in entities_in_text:
                all_entities.add(entity)
                graph.add_node(entity, description=self.link_entity_to_ontology(entity))
            
            for i in range(len(entities_in_text)):
                for j in range(i + 1, len(entities_in_text)):
                    entity1 = entities_in_text[i]
                    entity2 = entities_in_text[j]
                    if not graph.has_edge(entity1, entity2):
                        graph.add_edge(entity1, entity2, weight=1)
                    else:
                        graph[entity1][entity2]['weight'] += 1
        return graph

class EvidenceChainer:
    def __init__(self):
        pass

    def prune_irrelevant_nodes(self, graph, query_entities, patient_context_entities, threshold=1):
        relevant_nodes = set()
        for entity in query_entities:
            if entity in graph:
                relevant_nodes.add(entity)
        for entity in patient_context_entities:
            if entity in graph:
                relevant_nodes.add(entity)

        nodes_to_remove = [node for node in graph.nodes if node not in relevant_nodes and graph.degree(node) < threshold]
        pruned_graph = graph.copy()
        pruned_graph.remove_nodes_from(nodes_to_remove)
        return pruned_graph

    def synthesize_chains(self, graph, query, max_chain_length=3):
        evidence_chains = []
        query_keywords = set(query.lower().split())

        for node in graph.nodes:
            if node.lower() in query_keywords or any(keyword in node.lower() for keyword in query_keywords):
                desc = graph.nodes[node].get('description', '')
                if desc:
                    evidence_chains.append(f"{node}: {desc}")
                
                # Simple path traversal for chaining
                for neighbor in graph.neighbors(node):
                    neighbor_desc = graph.nodes[neighbor].get('description', '')
                    if neighbor_desc:
                        chain = f"{node} is related to {neighbor}. {neighbor}: {neighbor_desc}"
                        evidence_chains.append(chain)

                    for next_neighbor in graph.neighbors(neighbor):
                        if next_neighbor != node:
                            next_neighbor_desc = graph.nodes[next_neighbor].get('description', '')
                            if next_neighbor_desc:
                                chain = f"{node} is related to {neighbor}, which is related to {next_neighbor}. {next_neighbor}: {next_neighbor_desc}"
                                evidence_chains.append(chain)
        return list(set(evidence_chains))

    def prioritize_chains(self, evidence_chains, query):
        query_lower = query.lower()
        scored_chains = []
        for chain in evidence_chains:
            score = 0
            if query_lower in chain.lower():
                score += 2
            for keyword in query_lower.split():
                if keyword in chain.lower():
                    score += 1
            scored_chains.append((score, chain))
        scored_chains.sort(key=lambda x: x[0], reverse=True)
        return [chain for score, chain in scored_chains]

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.linker = EntityLinker()
        self.chainer = EvidenceChainer()

    def diagnose(self, symptoms, medical_history, diagnostic_question, patient_id=None, current_medications=None):
        query = f"{symptoms} {medical_history} {diagnostic_question}"
        drug_names = current_medications if current_medications else []

        # 1. Knowledge Retrieval
        raw_evidence = self.retriever.retrieve_knowledge(query, patient_id=patient_id, drug_names=drug_names)
        print("\n--- Raw Evidence ---")
        for e in raw_evidence:
            print(e)

        # 2. Entity Linking
        evidence_graph = self.linker.build_evidence_graph(raw_evidence)
        print("\n--- Evidence Graph Nodes (Entities) ---")
        for node in evidence_graph.nodes(data=True):
            print(node)

        # Prepare entities for pruning
        query_entities = self.linker.identify_entities(query)
        patient_context_entities = []
        if patient_id:
            patient_info = self.retriever.retrieve_ehr_data(patient_id)
            patient_context_entities.extend(self.linker.identify_entities(patient_info))
        if current_medications:
            patient_context_entities.extend(current_medications)

        # 3. Evidence Chaining
        pruned_graph = self.chainer.prune_irrelevant_nodes(evidence_graph, query_entities, patient_context_entities)
        print("\n--- Pruned Graph Nodes ---")
        for node in pruned_graph.nodes():
            print(node)

        evidence_chains = self.chainer.synthesize_chains(pruned_graph, diagnostic_question)
        prioritized_chains = self.chainer.prioritize_chains(evidence_chains, diagnostic_question)

        print("\n--- Prioritized Evidence Chains for LLM ---")
        for i, chain in enumerate(prioritized_chains[:5]): # Show top 5 chains
            print(f"{i+1}. {chain}")

        # This 'prioritized_chains' list would then be passed to an LLM Prompt Engine.
        # For demonstration, we'll just return it.
        return prioritized_chains

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    print("\n*** Case 1: Diabetes Diagnosis ***")
    symptoms1 = "Patient reports increased thirst, frequent urination, and fatigue."
    medical_history1 = "Family history of diabetes."
    diagnostic_question1 = "What is the most likely diagnosis and potential treatment?"
    patient_id1 = "P1001"
    current_medications1 = ["Metformin"]

    consolidated_evidence1 = assistant.diagnose(
        symptoms=symptoms1,
        medical_history=medical_history1,
        diagnostic_question=diagnostic_question1,
        patient_id=patient_id1,
        current_medications=current_medications1
    )

    print("\n\n*** Case 2: Hypertension Management ***")
    symptoms2 = "Regular check-up shows consistently high blood pressure readings."
    medical_history2 = "No significant medical history apart from occasional headaches."
    diagnostic_question2 = "What are the initial recommendations for managing hypertension?"
    patient_id2 = "P2002" # Non-existent patient for broader search
    current_medications2 = []

    consolidated_evidence2 = assistant.diagnose(
        symptoms=symptoms2,
        medical_history=medical_history2,
        diagnostic_question=diagnostic_question2,
        patient_id=patient_id2,
        current_medications=current_medications2
    )