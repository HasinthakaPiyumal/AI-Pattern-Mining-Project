class MedicalKnowledgeGraph:
    def __init__(self):
        self.nodes = {
            "D_RareDiseaseA": {"type": "Disease", "name": "Rare Disease A", "symptoms": ["fatigue", "muscle weakness", "rash", "joint pain"], "genes": ["GENE_X", "GENE_Y"], "description": "A complex condition characterized by systemic inflammation and progressive muscle weakness."}, 
            "D_CommonCold": {"type": "Disease", "name": "Common Cold", "symptoms": ["cough", "sore throat", "fever", "runny nose"], "genes": [], "description": "A viral infectious disease of the upper respiratory tract."}, 
            "S_fatigue": {"type": "Symptom", "name": "fatigue"},
            "S_muscle_weakness": {"type": "Symptom", "name": "muscle weakness"},
            "S_rash": {"type": "Symptom", "name": "rash"},
            "S_joint_pain": {"type": "Symptom", "name": "joint pain"},
            "S_cough": {"type": "Symptom", "name": "cough"},
            "S_sore_throat": {"type": "Symptom", "name": "sore throat"},
            "S_fever": {"type": "Symptom", "name": "fever"},
            "S_runny_nose": {"type": "Symptom", "name": "runny nose"},
            "G_GENE_X": {"type": "Gene", "name": "GENE_X"},
            "G_GENE_Y": {"type": "Gene", "name": "GENE_Y"},
            "P_Paper1": {"type": "Paper", "title": "Clinical Manifestations of Rare Disease A", "abstract": "This paper discusses the common symptoms and genetic markers associated with Rare Disease A, including fatigue, muscle weakness, rash, and the involvement of GENE_X and GENE_Y. Early diagnosis is crucial.", "related_entities": ["D_RareDiseaseA", "S_fatigue", "S_muscle_weakness", "S_rash", "G_GENE_X", "G_GENE_Y"]},
            "P_Paper2": {"type": "Paper", "title": "Treatment for Common Cold", "abstract": "Overview of common remedies for symptoms like cough, fever, and runny nose. Focus on symptomatic relief.", "related_entities": ["D_CommonCold", "S_cough", "S_fever", "S_runny_nose"]},
            "P_Paper3": {"type": "Paper", "title": "Genetic Basis of Neuromuscular Disorders", "abstract": "Research on genetic predispositions to muscle weakness and fatigue, often involving genes like GENE_Y.", "related_entities": ["S_muscle_weakness", "S_fatigue", "G_GENE_Y"]}
        }

        self.edges = []
        for node_id, data in self.nodes.items():
            if data["type"] == "Disease":
                for symptom in data["symptoms"]:
                    symptom_id = "S_" + symptom.replace(" ", "_")
                    if symptom_id in self.nodes:
                        self.edges.append({"source": node_id, "target": symptom_id, "relation": "HAS_SYMPTOM"})
                for gene in data["genes"]:
                    gene_id = "G_" + gene
                    if gene_id in self.nodes:
                        self.edges.append({"source": node_id, "target": gene_id, "relation": "ASSOCIATED_GENE"})
            elif data["type"] == "Paper":
                for entity_id in data.get("related_entities", []):
                    if entity_id in self.nodes:
                        self.edges.append({"source": entity_id, "target": node_id, "relation": "MENTIONED_IN_PAPER"})
    
    def get_related_entities(self, entity_id, relation_type=None):
        related = []
        for edge in self.edges:
            if edge["source"] == entity_id and (relation_type is None or edge["relation"] == relation_type):
                related.append(self.nodes[edge["target"]]["name"])
            elif edge["target"] == entity_id and (relation_type is None or edge["relation"] == relation_type):
                related.append(self.nodes[edge["source"]]["name"])
        return list(set(related))

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.mkg = MedicalKnowledgeGraph()

    def _extract_medical_concepts(self, text):
        extracted_concepts = []
        text_lower = text.lower()
        for node_id, data in self.mkg.nodes.items():
            if "name" in data and data["name"].lower() in text_lower:
                extracted_concepts.append(node_id)
            if data["type"] == "Symptom" and data["name"].lower() in text_lower:
                extracted_concepts.append(node_id)
            if data["type"] == "Disease" and data["name"].lower() in text_lower:
                extracted_concepts.append(node_id)
            if data["type"] == "Disease" and "symptoms" in data:
                for symptom in data["symptoms"]:
                    if symptom.lower() in text_lower and "S_" + symptom.replace(" ", "_") not in extracted_concepts:
                        extracted_concepts.append("S_" + symptom.replace(" ", "_"))
        return list(set(extracted_concepts))

    def _calculate_text_overlap(self, text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1.intersection(words2))

    def _lightweight_pruning(self, patient_input_text, N_lightweight=5):
        candidate_scores = {}
        patient_keywords = patient_input_text.lower().split()

        # Semantic Similarity (simulated with keyword overlap)
        for node_id, data in self.mkg.nodes.items():
            text_to_compare = ""
            if "name" in data: text_to_compare += data["name"] + " "
            if "symptoms" in data: text_to_compare += " ".join(data["symptoms"]) + " "
            if "description" in data: text_to_compare += data["description"] + " "
            if "abstract" in data: text_to_compare += data["abstract"] + " "
            
            overlap_score = self._calculate_text_overlap(patient_input_text, text_to_compare)
            
            if node_id not in candidate_scores: candidate_scores[node_id] = 0
            candidate_scores[node_id] += overlap_score * 0.7 # Weighted

            # Simple Lexical Search (simulated BM25)
            for keyword in patient_keywords:
                if keyword in text_to_compare.lower():
                    candidate_scores[node_id] += 1 # Boost for direct keyword match
        
        # Filter for diseases and papers primarily in lightweight pruning
        filtered_candidates = {
            node_id: score 
            for node_id, score in candidate_scores.items() 
            if self.mkg.nodes[node_id]["type"] in ["Disease", "Paper"]
        }

        sorted_candidates = sorted(filtered_candidates.items(), key=lambda item: item[1], reverse=True)
        return [node_id for node_id, score in sorted_candidates[:N_lightweight]]

    def _simulate_llm_response(self, prompt):
        if "Rare Disease A" in prompt and "fatigue" in prompt and "muscle weakness" in prompt:
            return "Based on the symptoms of fatigue and muscle weakness, and considering Rare Disease A's association with these, it is a strong candidate. Further investigation into genetic markers GENE_X and GENE_Y is recommended. The provided context from 'Clinical Manifestations of Rare Disease A' strongly supports this direction."
        elif "Common Cold" in prompt and "cough" in prompt:
            return "The patient's symptoms of cough and sore throat are highly consistent with the Common Cold. The absence of rare disease markers makes this the most probable diagnosis."
        elif "missing information" in prompt:
            return "The LLM suggests investigating genetic test results for GENE_X and GENE_Y to confirm the rare disease hypothesis. Also, a detailed family medical history could be beneficial."
        else:
            return "The LLM is analyzing the provided medical context and patient data to provide a differential diagnosis. It notes the complexity and suggests exploring related research papers for similar cases."

    def diagnose(self, patient_symptoms_text, N_lightweight=5):
        print(f"Patient Input: {patient_symptoms_text}")
        print("\n--- Phase 1: Lightweight Pruning (Initial Candidate Generation) ---")
        
        # 1. Input Module: Extract concepts
        patient_concepts = self._extract_medical_concepts(patient_symptoms_text)
        print(f"Extracted Patient Concepts: {[self.mkg.nodes[c]['name'] for c in patient_concepts if c in self.mkg.nodes]}")

        # 2. Lightweight Pruning
        lightweight_candidates_ids = self._lightweight_pruning(patient_symptoms_text, N_lightweight)
        lightweight_candidates = {cid: self.mkg.nodes[cid] for cid in lightweight_candidates_ids}
        print("Top Lightweight Candidates (Diseases/Papers):")
        for cid, data in lightweight_candidates.items():
            print(f"  - {data['name']} (Type: {data['type']})")

        print("\n--- Phase 2: LLM-Guided Deep Pruning & Reasoning ---")
        final_diagnoses = {}
        
        for candidate_id in lightweight_candidates_ids:
            candidate_data = self.mkg.nodes[candidate_id]
            
            # Context Construction for LLM
            prompt_context = f"Patient Symptoms: {patient_symptoms_text}\n"
            prompt_context += f"Candidate: {candidate_data['name']} (Type: {candidate_data['type']})\n"
            if 'description' in candidate_data: prompt_context += f"Description: {candidate_data['description']}\n"
            if 'symptoms' in candidate_data: prompt_context += f"Associated Symptoms: {', '.join(candidate_data['symptoms'])}\n"
            if 'genes' in candidate_data: prompt_context += f"Associated Genes: {', '.join(candidate_data['genes'])}\n"
            if 'abstract' in candidate_data: prompt_context += f"Paper Abstract: {candidate_data['abstract']}\n"
            
            related_entities = self.mkg.get_related_entities(candidate_id)
            if related_entities: prompt_context += f"Related MKG Entities: {', '.join(related_entities)}\n"
            
            llm_prompt = f"Given the following patient information and a candidate medical entity, evaluate its relevance and potential as a diagnosis. Provide a likelihood assessment and suggested next steps or alternative diagnoses. Focus on explaining the reasoning based on the provided context.\n\n{prompt_context}\n\nBased on this, what is the likelihood of this candidate being the correct diagnosis, and what are the next diagnostic steps?"
            
            # LLM Reasoning (Simulated)
            llm_response = self._simulate_llm_response(llm_prompt)
            print(f"\n--- LLM Analysis for Candidate: {candidate_data['name']} ---")
            print(llm_response)
            final_diagnoses[candidate_data['name']] = llm_response
            
            # Simulated Iterative Refinement (Asking for missing info)
            if "further investigation" in llm_response.lower() or "next diagnostic steps" in llm_response.lower():
                refinement_prompt = f"Based on your previous analysis for {candidate_data['name']}, what specific missing information would be most critical for a definitive diagnosis?"
                refinement_response = self._simulate_llm_response(refinement_prompt)
                print(f"  [LLM Refinement Suggestion]: {refinement_response}")

        print("\n--- Final Diagnostic Summary ---")
        if not final_diagnoses:
            print("No strong diagnostic candidates identified.")
        else:
            for diagnosis, explanation in final_diagnoses.items():
                print(f"Diagnosis Candidate: {diagnosis}\nExplanation: {explanation}\n---")


if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()

    # Test Case 1: Symptoms suggesting a rare disease
    patient_input_1 = "I have severe fatigue, muscle weakness, and a strange rash. I also feel joint pain."
    assistant.diagnose(patient_input_1)

    print("\n=======================================================\n")

    # Test Case 2: Symptoms suggesting a common cold
    patient_input_2 = "I have a cough, sore throat, and a low-grade fever. My nose is also runny."
    assistant.diagnose(patient_input_2)

    print("\n=======================================================\n")

    # Test Case 3: More ambiguous symptoms
    patient_input_3 = "I feel tired all the time and have some general body aches."
    assistant.diagnose(patient_input_3)
