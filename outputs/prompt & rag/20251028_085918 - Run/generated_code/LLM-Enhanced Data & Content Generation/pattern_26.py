import json
from typing import List, Dict

# Mock external libraries/services for demonstration
class KnowledgeGraphDB:
    def __init__(self):
        self.data = {
            "disease_1": {"symptoms": ["fever", "cough"], "treatments": ["rest", "medication A"], "related": ["disease_2"]},
            "disease_2": {"symptoms": ["headache", "nausea"], "treatments": ["fluids", "medication B"], "related": []}
        }

    def retrieve_info(self, query: str) -> List[Dict]:
        results = []
        for k, v in self.data.items():
            if query.lower() in k.lower() or any(query.lower() in s.lower() for s in v.get("symptoms", [])):
                results.append({"entity": k, "info": v})
        return results

class LLM:
    def __init__(self):
        pass

    def reason(self, context: str, question: str) -> str:
        # Simulate LLM reasoning based on context and question
        if "fever and cough" in question.lower() and "disease_1" in context.lower():
            return "Based on the symptoms of fever and cough, and retrieved information on disease_1, a potential diagnosis is disease_1. Treatment options include rest and medication A."
        elif "headache and nausea" in question.lower() and "disease_2" in context.lower():
            return "Given headache and nausea, and information on disease_2, consider disease_2. Treatment includes fluids and medication B."
        return "I need more information or the context does not directly support a diagnosis for that specific query."

class MedicalDiagnosisSystem:
    def __init__(self):
        self.kg_db = KnowledgeGraphDB()
        self.llm = LLM()

    def query_system(self, patient_query: str) -> Dict:
        # 1. Retrieve relevant information from the Knowledge Graph
        retrieved_data = self.kg_db.retrieve_info(patient_query)
        
        context = ""
        if retrieved_data:
            context = "Medical Knowledge Graph Data:\n"
            for item in retrieved_data:
                context += f"Entity: {item['entity']}, Info: {json.dumps(item['info'])}\n"
        
        # 2. Use LLM to reason over retrieved information and user query
        llm_response = self.llm.reason(context, patient_query)

        return {
            "query": patient_query,
            "retrieved_info": retrieved_data,
            "diagnosis_reasoning": llm_response,
            "sources_cited": [item['entity'] for item in retrieved_data] if retrieved_data else []
        }

# Example Usage:
if __name__ == "__main__":
    system = MedicalDiagnosisSystem()

    query1 = "What causes fever and cough, and what are the treatments?"
    response1 = system.query_system(query1)
    print(f"\nQuery: {response1['query']}")
    print(f"Diagnosis/Reasoning: {response1['diagnosis_reasoning']}")
    print(f"Sources: {', '.join(response1['sources_cited'])}")

    query2 = "Patient has headache and nausea. What could it be?"
    response2 = system.query_system(query2)
    print(f"\nQuery: {response2['query']}")
    print(f"Diagnosis/Reasoning: {response2['diagnosis_reasoning']}")
    print(f"Sources: {', '.join(response2['sources_cited'])}")

    query3 = "Symptoms include joint pain."
    response3 = system.query_system(query3)
    print(f"\nQuery: {response3['query']}")
    print(f"Diagnosis/Reasoning: {response3['diagnosis_reasoning']}")
    print(f"Sources: {', '.join(response3['sources_cited'])}")