"""  
medical_diagnostic_assistant.py  
  
This script implements a simplified Medical Diagnostic Assistant using the Modular Reasoning, Knowledge and Language (MRKL) System pattern.  
It simulates an LLM router that orchestrates calls to various external medical tools (EHR, Medical Knowledge Base, Drug Interaction, Medical Calculator, Specialist Consult)  
to provide comprehensive diagnostic assistance.
"""

import json

# --- 1. Simulated External Tools/APIs ---

class Tool:
    """Base class for all simulated external tools."""
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the 'run' method.")

class EHRTool(Tool):
    """Simulates an Electronic Health Record (EHR) system API."""
    def __init__(self):
        super().__init__(
            "EHR_System",
            "Fetches patient history, allergies, medications, and lab results."
        )
        self.patient_data = {
            "P001": {
                "name": "Alice Smith",
                "age": 45,
                "gender": "Female",
                "allergies": ["Penicillin"],
                "medications": ["Metformin (500mg daily)", "Lisinopril (10mg daily)"],
                "history": ["Type 2 Diabetes", "Hypertension"],
                "lab_results": {"blood_sugar": "180 mg/dL", "bp": "140/90 mmHg"}
            },
            "P002": {
                "name": "Bob Johnson",
                "age": 60,
                "gender": "Male",
                "allergies": ["Sulfa drugs"],
                "medications": ["Aspirin (81mg daily)"],
                "history": ["Coronary Artery Disease"],
                "lab_results": {"cholesterol": "220 mg/dL", "bp": "130/85 mmHg"}
            }
        }

    def run(self, patient_id):
        print(f"[EHRTool] Fetching data for patient ID: {patient_id}")
        return self.patient_data.get(patient_id, {"error": "Patient not found"})

class MedicalKnowledgeBaseTool(Tool):
    """Simulates a Medical Knowledge Base (e.g., PubMed, UpToDate)."""
    def __init__(self):
        super().__init__(
            "Medical_Knowledge_Base",
            "Retrieves general medical information, research, and guidelines."
        )
        self.knowledge_base = {
            "diabetes_symptoms": "Frequent urination, increased thirst, unexplained weight loss, blurred vision, fatigue.",
            "hypertension_treatment": "Lifestyle changes (diet, exercise), medications (ACE inhibitors, diuretics, beta-blockers).",
            "appendicitis_diagnosis": "Physical exam, blood tests (elevated WBC), urine test, imaging (ultrasound, CT scan).",
            "common_cold_treatment": "Rest, fluids, over-the-counter pain relievers, decongestants."
        }

    def run(self, query):
        print(f"[MedicalKnowledgeBaseTool] Searching for: {query}")
        # Simple keyword matching for demonstration
        for key, value in self.knowledge_base.items():
            if query.lower() in key.lower() or query.lower() in value.lower():
                return {query: value}
        return {"error": f"No information found for '{query}'."}

class DrugInteractionTool(Tool):
    """Simulates a Drug Interaction API."""
    def __init__(self):
        super().__init__(
            "Drug_Interaction_Checker",
            "Checks for potential adverse drug interactions."
        )
        self.interactions = {
            frozenset(["Metformin", "Lisinopril"]): "Generally safe, monitor kidney function due to Lisinopril. Metformin can rarely cause lactic acidosis.",
            frozenset(["Aspirin", "Warfarin"]): "High risk of bleeding. Concurrent use generally contraindicated unless closely monitored.",
            frozenset(["Decongestant", "Lisinopril"]): "Decongestants can increase blood pressure, potentially counteracting Lisinopril's effects."
        }

    def run(self, medications):
        print(f"[DrugInteractionTool] Checking interactions for: {', '.join(medications)}")
        med_set = frozenset(sorted(m.split(' ')[0] for m in medications))
        for interaction_set, info in self.interactions.items():
            if interaction_set.issubset(med_set) or med_set.issubset(interaction_set):
                return {"interaction_warning": info}
        return {"interaction_warning": "No significant interactions found for the given medications."}

class MedicalCalculatorTool(Tool):
    """Simulates a Medical Calculator API."""
    def __init__(self):
        super().__init__(
            "Medical_Calculator",
            "Performs medical computations like dosage calculations or BMI."
        )

    def run(self, calculation_type, **params):
        print(f"[MedicalCalculatorTool] Performing {calculation_type} calculation with params: {params}")
        if calculation_type == "BMI":
            weight_kg = params.get("weight_kg")
            height_m = params.get("height_m")
            if weight_kg and height_m and height_m > 0:
                bmi = weight_kg / (height_m ** 2)
                return {"BMI": f"{bmi:.2f}"}
            return {"error": "Invalid parameters for BMI calculation. Need weight_kg and height_m.", "params": params}
        elif calculation_type == "BSA": # Body Surface Area (DuBois formula)
            weight_kg = params.get("weight_kg")
            height_cm = params.get("height_cm")
            if weight_kg and height_cm:
                bsa = 0.007184 * (weight_kg**0.425) * (height_cm**0.725)
                return {"BSA_m2": f"{bsa:.2f}"}
            return {"error": "Invalid parameters for BSA calculation. Need weight_kg and height_cm.", "params": params}
        # Add more calculation types as needed
        return {"error": f"Unknown calculation type: {calculation_type}", "params": params}

class SpecialistConsultTool(Tool):
    """Simulates a Specialist Consult database/API."""
    def __init__(self):
        super().__init__(
            "Specialist_Consult",
            "Provides mock specialist advice or referral information."
        )
        self.specialist_advice = {
            "rare_neurological_disorder": "Recommend referral to a neurologist specializing in rare diseases for further evaluation and advanced imaging.",
            "complex_cardiac_case": "Suggest consultation with an interventional cardiologist for angiography and possible stent placement consideration.",
            "unexplained_rash": "Advise referral to a dermatologist for biopsy and specialized allergy testing."
        }

    def run(self, condition):
        print(f"[SpecialistConsultTool] Consulting for condition: {condition}")
        # Simple keyword matching for demonstration
        for key, value in self.specialist_advice.items():
            if condition.lower() in key.lower():
                return {"specialist_recommendation": value}
        return {"specialist_recommendation": f"No specific specialist advice found for '{condition}'. Consider a general referral if symptoms persist."}

# --- 2. LLM Router (Orchestrator) ---

class MedicalDiagnosticAssistant:
    """  
    Acts as the LLM router for the MRKL system, orchestrating calls to various medical tools.  
    """
    def __init__(self):
        self.tools = {
            "ehr": EHRTool(),
            "knowledge_base": MedicalKnowledgeBaseTool(),
            "drug_interaction": DrugInteractionTool(),
            "calculator": MedicalCalculatorTool(),
            "specialist_consult": SpecialistConsultTool()
        }

    def _determine_relevant_tools(self, query):
        """  
        Simulates the LLM's routing logic to determine which tools are relevant  
        based on the query. In a real system, this would be handled by the LLM itself.  
        """
        relevant_tools = []
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ["patient history", "lab results", "medications", "allergies", "patient id"]):
            relevant_tools.append("ehr")
        if any(keyword in query_lower for keyword in ["what is", "symptoms of", "treatment for", "diagnosis of", "info on"]):
            relevant_tools.append("knowledge_base")
        if any(keyword in query_lower for keyword in ["drug interaction", "medication interaction", "side effects combine"]):
            relevant_tools.append("drug_interaction")
        if any(keyword in query_lower for keyword in ["calculate bmi", "body surface area", "dosage"]):
            relevant_tools.append("calculator")
        if any(keyword in query_lower for keyword in ["refer to a specialist", "specialist advice", "complex case"]):
            relevant_tools.append("specialist_consult")

        return list(set(relevant_tools)) # Remove duplicates

    def _extract_tool_parameters(self, query, tool_name):
        """  
        Simulates the LLM's ability to extract parameters for tool calls from the query.  
        This is highly simplified for demonstration.  
        """
        query_lower = query.lower()
        params = {}

        if tool_name == "ehr":
            # Example: "Get patient history for P001"
            import re
            match = re.search(r"patient (?:id|ID) (P\d{3})", query)
            if match: params["patient_id"] = match.group(1).upper()

        elif tool_name == "knowledge_base":
            # Example: "What are the symptoms of diabetes?"
            if "symptoms of" in query_lower: params["query"] = query_lower.split("symptoms of")[-1].strip().replace('?', '') + "_symptoms"
            elif "treatment for" in query_lower: params["query"] = query_lower.split("treatment for")[-1].strip().replace('?', '') + "_treatment"
            elif "diagnosis of" in query_lower: params["query"] = query_lower.split("diagnosis of")[-1].strip().replace('?', '') + "_diagnosis"
            else: params["query"] = query_lower.replace("info on ", "").replace("what is ", "").strip().replace('?', '')

        elif tool_name == "drug_interaction":
            # Example: "Check interactions for Metformin and Lisinopril"
            meds_match = re.search(r"for (.*)", query_lower)
            if meds_match: params["medications"] = [m.strip() for m in meds_match.group(1).split(" and ")]
            else: # Fallback to common meds for demo
                known_meds = ["Metformin", "Lisinopril", "Aspirin", "Warfarin"]
                params["medications"] = [m for m in known_meds if m.lower() in query_lower]
                if not params["medications"] and "medications" in query_lower:
                    params["medications"] = ["Metformin", "Lisinopril"] # Default for demo

        elif tool_name == "calculator":
            if "calculate bmi" in query_lower:
                params["calculation_type"] = "BMI"
                weight_match = re.search(r"weight is (\d+\.?\d*)\s*kg", query_lower)
                height_match = re.search(r"height is (\d+\.?\d*)\s*m", query_lower)
                if weight_match: params["weight_kg"] = float(weight_match.group(1))
                if height_match: params["height_m"] = float(height_match.group(1))
            elif "body surface area" in query_lower:
                params["calculation_type"] = "BSA"
                weight_match = re.search(r"weight is (\d+\.?\d*)\s*kg", query_lower)
                height_match = re.search(r"height is (\d+\.?\d*)\s*cm", query_lower)
                if weight_match: params["weight_kg"] = float(weight_match.group(1))
                if height_match: params["height_cm"] = float(height_match.group(1))

        elif tool_name == "specialist_consult":
            # Example: "I need specialist advice for a rare neurological disorder"
            condition_match = re.search(r"for (.*)", query_lower)
            if condition_match: params["condition"] = condition_match.group(1).strip()
            else: params["condition"] = query_lower.replace("refer to a specialist for ", "").replace("specialist advice on ", "").strip()

        return params

    def _synthesize_response(self, original_query, tool_outputs):
        """  
        Simulates the LLM's ability to synthesize a coherent response from tool outputs.  
        """
        response_parts = [
            f"Assistant's Diagnostic Report for Query: '{original_query}'\n",
            "---"
        ]

        if not tool_outputs:
            response_parts.append("No relevant information found or tools invoked.")
            return "\n".join(response_parts)

        for tool_name, output in tool_outputs.items():
            response_parts.append(f"\n[{self.tools[tool_name].name} Information]:")
            if isinstance(output, dict):
                for key, value in output.items():
                    response_parts.append(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                response_parts.append(f"  {output}")

        response_parts.append("\n---\nDisclaimer: This is a simulated diagnostic assistant and should not replace professional medical advice.")
        return "\n".join(response_parts)

    def process_query(self, query):
        print(f"\nProcessing Query: '{query}'")
        # 1. LLM Router determines relevant tools
        relevant_tools = self._determine_relevant_tools(query)
        print(f"[Router] Identified relevant tools: {', '.join(relevant_tools) if relevant_tools else 'None'}")

        tool_outputs = {}
        # 2. LLM Router extracts parameters and invokes tools
        for tool_name in relevant_tools:
            tool_instance = self.tools[tool_name]
            params = self._extract_tool_parameters(query, tool_name)
            if params:
                try:
                    output = tool_instance.run(**params)
                    tool_outputs[tool_name] = output
                except Exception as e:
                    tool_outputs[tool_name] = {"error": f"Tool {tool_name} failed: {e}"}
            else:
                tool_outputs[tool_name] = {"error": f"Could not extract parameters for {tool_name} from query."}

        # 3. LLM Router synthesizes final response
        final_response = self._synthesize_response(query, tool_outputs)
        return final_response

# --- Main Execution / Demonstration --- 

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Example Queries
    queries = [
        "What is the patient history for P001? Also, check for drug interactions between Metformin and Lisinopril.",
        "What are the symptoms of diabetes and how to treat hypertension?",
        "Calculate BMI for a patient with weight is 70 kg and height is 1.75 m.",
        "I need specialist advice for a rare neurological disorder and information on common cold treatment.",
        "Check drug interactions for Aspirin and Warfarin. Also, get lab results for P002.",
        "What is the body surface area for a patient with weight is 75 kg and height is 180 cm?",
        "Just a simple greeting."
    ]

    for i, query in enumerate(queries):
        print(f"\n=== Query {i+1}/{len(queries)} ===")
        response = assistant.process_query(query)
        print(f"\nFinal Assistant Response:\n{response}")
        print("\n" + "="*50)
