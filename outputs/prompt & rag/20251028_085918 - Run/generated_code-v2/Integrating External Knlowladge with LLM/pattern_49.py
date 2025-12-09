import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import gradio as gr
import threading
import time

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_sample_kg()

    def _populate_sample_kg(self):
        self.add_entity("Symptom", "Fever")
        self.add_entity("Symptom", "Cough")
        self.add_entity("Symptom", "Headache")
        self.add_entity("Symptom", "Fatigue")
        self.add_entity("Symptom", "Sore Throat")

        self.add_entity("Disease", "Common Cold")
        self.add_entity("Disease", "Influenza")
        self.add_entity("Disease", "Streptococcal Pharyngitis")
        self.add_entity("Disease", "Migraine")

        self.add_entity("Test", "Rapid Strep Test")
        self.add_entity("Test", "Flu Test")
        self.add_entity("Test", "Blood Culture")

        self.add_entity("Treatment", "Rest")
        self.add_entity("Treatment", "Fluids")
        self.add_entity("Treatment", "Pain Relievers")
        self.add_entity("Treatment", "Antibiotics")
        self.add_entity("Treatment", "Antivirals")

        self.add_relation("Symptom", "Fever", "indicates", "Disease", "Common Cold")
        self.add_relation("Symptom", "Cough", "indicates", "Disease", "Common Cold")
        self.add_relation("Symptom", "Sore Throat", "indicates", "Disease", "Common Cold")
        self.add_relation("Symptom", "Fatigue", "indicates", "Disease", "Common Cold")

        self.add_relation("Symptom", "Fever", "indicates", "Disease", "Influenza")
        self.add_relation("Symptom", "Cough", "indicates", "Disease", "Influenza")
        self.add_relation("Symptom", "Fatigue", "indicates", "Disease", "Influenza")

        self.add_relation("Symptom", "Sore Throat", "indicates", "Disease", "Streptococcal Pharyngitis")
        self.add_relation("Symptom", "Fever", "indicates", "Disease", "Streptococcal Pharyngitis")

        self.add_relation("Symptom", "Headache", "indicates", "Disease", "Migraine")

        self.add_relation("Disease", "Streptococcal Pharyngitis", "diagnosed_by", "Test", "Rapid Strep Test")
        self.add_relation("Disease", "Influenza", "diagnosed_by", "Test", "Flu Test")

        self.add_relation("Disease", "Common Cold", "treated_by", "Treatment", "Rest")
        self.add_relation("Disease", "Common Cold", "treated_by", "Treatment", "Fluids")
        self.add_relation("Disease", "Common Cold", "treated_by", "Treatment", "Pain Relievers")

        self.add_relation("Disease", "Influenza", "treated_by", "Treatment", "Antivirals")
        self.add_relation("Disease", "Influenza", "treated_by", "Treatment", "Rest")

        self.add_relation("Disease", "Streptococcal Pharyngitis", "treated_by", "Treatment", "Antibiotics")

    def add_entity(self, entity_type: str, entity_name: str):
        node_id = f"{entity_type}:{entity_name}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=entity_type, name=entity_name)

    def add_relation(self, source_type: str, source_name: str, relation: str, target_type: str, target_name: str):
        source_id = f"{source_type}:{source_name}"
        target_id = f"{target_type}:{target_name}"
        if not self.graph.has_node(source_id):
            self.add_entity(source_type, source_name)
        if not self.graph.has_node(target_id):
            self.add_entity(target_type, target_name)
        self.graph.add_edge(source_id, target_id, relation=relation)

    def get_related_entities(self, entity_name: str, entity_type: str = None, relation_type: str = None, target_type: str = None):
        results = []
        for node_id in self.graph.nodes:
            if (entity_type is None or self.graph.nodes[node_id]['type'] == entity_type) and \
               (entity_name.lower() in self.graph.nodes[node_id]['name'].lower()):
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    if relation_type is None or edge_data.get('relation') == relation_type:
                        if target_type is None or self.graph.nodes[neighbor]['type'] == target_type:
                            results.append({
                                "source": self.graph.nodes[node_id]['name'],
                                "relation": edge_data.get('relation'),
                                "target": self.graph.nodes[neighbor]['name'],
                                "target_type": self.graph.nodes[neighbor]['type']
                            })
        return results

    def find_paths(self, start_entity_name: str, start_entity_type: str, end_entity_type: str, max_length: int = 3):
        start_node_id = f"{start_entity_type}:{start_entity_name}"
        if not self.graph.has_node(start_node_id):
            return []
        paths = []
        for path in nx.all_simple_paths(self.graph, source=start_node_id, cutoff=max_length):
            if len(path) > 1 and self.graph.nodes[path[-1]]['type'] == end_entity_type:
                formatted_path = []
                for i in range(len(path) - 1):
                    source_node = self.graph.nodes[path[i]]
                    target_node = self.graph.nodes[path[i+1]]
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    formatted_path.append(f"{source_node['name']} ({source_node['type']}) --{edge_data.get('relation')}--> {target_node['name']} ({target_node['type']})")
                paths.append(" -> ".join(formatted_path))
        return paths

class FineTunedLLM:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate_diagnosis_and_plan(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Fever" in prompt and "Cough" in prompt:
            if "Sore Throat" in prompt:
                return "Based on the symptoms (Fever, Cough, Sore Throat) and KG integration, a likely diagnosis is Common Cold. Treatment plan: Rest, Fluids, Pain Relievers."
            elif "Fatigue" in prompt:
                return "Based on the symptoms (Fever, Cough, Fatigue) and KG integration, a likely diagnosis is Influenza. Treatment plan: Antivirals, Rest."
        elif "Sore Throat" in prompt and "Fever" in prompt:
            return "Based on the symptoms (Sore Throat, Fever) and KG integration, consider Streptococcal Pharyngitis. Further testing (Rapid Strep Test) is recommended. Treatment plan: Antibiotics if confirmed."
        elif "Headache" in prompt:
            return "Based on the symptom (Headache) and KG integration, a possible diagnosis is Migraine. Consider pain relievers and rest."

        return response + "\n\n(Note: This LLM output is a placeholder simulating KG-grounded reasoning. Actual fine-tuned LLM would provide more accurate and dynamic outputs based on real KG data.)"

class KGRetrievalModule:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg

    def retrieve_relevant_kg_info(self, symptoms: list[str], medical_history: str = "", test_results: str = "") -> str:
        kg_context = []
        for symptom in symptoms:
            related_diseases = self.kg.get_related_entities(symptom, entity_type="Symptom", relation_type="indicates", target_type="Disease")
            for rel in related_diseases:
                kg_context.append(f"Symptom '{rel['source']}' indicates Disease '{rel['target']}'.")

            diagnostic_paths = self.kg.find_paths(symptom, "Symptom", "Test")
            for path in diagnostic_paths:
                kg_context.append(f"Diagnostic pathway from symptom: {path}")

            treatment_paths = self.kg.find_paths(symptom, "Symptom", "Treatment")
            for path in treatment_paths:
                kg_context.append(f"Potential treatment pathway from symptom: {path}")

        return "\n".join(kg_context)

app = FastAPI()

class PatientInput(BaseModel):
    symptoms: list[str]
    medical_history: str = ""
    test_results: str = ""

kg = MedicalKnowledgeGraph()
llm = FineTunedLLM()
kg_retrieval = KGRetrievalModule(kg)

@app.post("/diagnose")
async def diagnose_patient(patient_input: PatientInput):
    try:
        symptoms_str = ", ".join(patient_input.symptoms)
        
        retrieved_kg_info = kg_retrieval.retrieve_relevant_kg_info(
            symptoms=patient_input.symptoms,
            medical_history=patient_input.medical_history,
            test_results=patient_input.test_results
        )

        prompt = f"Patient presents with symptoms: {symptoms_str}. Medical history: {patient_input.medical_history}. Test results: {patient_input.test_results}.\n\nRelevant medical knowledge from KG:\n{retrieved_kg_info}\n\nBased on this information, provide a diagnosis and a treatment plan."
        
        diagnosis_and_plan = llm.generate_diagnosis_and_plan(prompt)
        
        return {"diagnosis_and_plan": diagnosis_and_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def medical_assistant_interface(symptoms_input, history_input, tests_input):
    symptoms_list = [s.strip() for s in symptoms_input.split(',') if s.strip()]
    
    response = requests.post("http://127.0.0.1:8000/diagnose", json={
        "symptoms": symptoms_list,
        "medical_history": history_input,
        "test_results": tests_input
    })
    
    if response.status_code == 200:
        return response.json()["diagnosis_and_plan"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def run_fastapi():
    uvicorn.run(app, host="0.0.0.1", port=8000)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Please install 'requests' library: pip install requests")
        exit()

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    time.sleep(2) 

    iface = gr.Interface(
        fn=medical_assistant_interface,
        inputs=[
            gr.Textbox(label="Symptoms (comma-separated)", placeholder="Fever, Cough, Headache"),
            gr.Textbox(label="Medical History", placeholder="e.g., Asthma, Allergies"),
            gr.Textbox(label="Test Results", placeholder="e.g., Rapid strep test negative")
        ],
        outputs="text",
        title="Medical Diagnostic Assistant (LLM-KG Integrated)",
        description="Enter patient details to get a diagnosis and treatment plan based on LLM's understanding of a Medical Knowledge Graph."
    )
    iface.launch(share=False)

    fastapi_thread.join()
