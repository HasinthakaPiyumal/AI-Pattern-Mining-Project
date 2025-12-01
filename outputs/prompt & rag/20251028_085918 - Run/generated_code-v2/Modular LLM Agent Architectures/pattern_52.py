class MedicalDatabaseTool:
    def get_info(self, query):
        if "pneumonia symptoms" in query.lower():
            return "Symptoms of pneumonia include cough, fever, shortness of breath, and chest pain."
        elif "diabetes facts" in query.lower():
            return "Diabetes is a chronic condition that affects how your body turns food into energy."
        return "No specific information found for your query in the medical database."

class DosageCalculatorTool:
    def calculate_dosage(self, drug, weight_kg, age_years):
        if drug.lower() == "amoxicillin":
            if age_years < 12:
                dosage_mg_per_kg = 25
                total_dosage = dosage_mg_per_kg * weight_kg
                return f"For a child of {age_years} years and {weight_kg} kg, the recommended Amoxicillin dosage is {total_dosage} mg per day."
            else:
                return "For adults, Amoxicillin dosage varies; consult a physician."
        return "Dosage calculation not available for this drug."

class ResearchPaperSearchTool:
    def search_papers(self, keywords):
        if "cystic fibrosis gene therapy" in keywords.lower() and "past 6 months" in keywords.lower():
            return [
                {"title": "Advances in Gene Editing for Cystic Fibrosis (Recent)", "summary": "Recent breakthroughs in CRISPR-Cas9 applications."}, 
                {"title": "Clinical Trials of CFTR Modulators (Latest)", "summary": "New phase 3 trials show promising results."}
            ]
        return [{"title": "General Medical Research (Mock)", "summary": "A summary of some general medical research topics."}]

class DrugInteractionCheckerTool:
    def check_interactions(self, drugs):
        drugs_lower = [d.lower() for d in drugs]
        if "ibuprofen" in drugs_lower and "warfarin" in drugs_lower:
            return "Potential severe interaction: Ibuprofen can increase the risk of bleeding when taken with Warfarin."
        elif len(drugs) > 1:
            return f"No significant interactions found between {', '.join(drugs)}. Always consult a pharmacist."
        return "Please provide at least two drugs to check for interactions."

class SymptomCheckerTool:
    def check_symptoms(self, symptoms):
        symptoms_lower = [s.lower() for s in symptoms]
        if "fever" in symptoms_lower and "cough" in symptoms_lower and "shortness of breath" in symptoms_lower:
            return "Potential diagnoses include Pneumonia, Bronchitis, or COVID-19. Consult a doctor for definitive diagnosis."
        elif "headache" in symptoms_lower and "nausea" in symptoms_lower:
            return "Possible diagnoses include Migraine, Tension Headache, or Dehydration."
        return "Based on the provided symptoms, no specific common diagnoses can be suggested without further information."

class LLMRouter:
    def route_query(self, query):
        query_lower = query.lower()
        if "symptoms" in query_lower or "diagnosis" in query_lower:
            symptoms = [s.strip() for s in query_lower.replace("symptoms of", "").replace("diagnose", "").split("and") if s.strip()]
            return {"tool": "symptom_checker", "args": {"symptoms": symptoms}}
        elif "dosage" in query_lower or "calculate drug" in query_lower:
            drug_match = next((word for word in ["amoxicillin", "ibuprofen"] if word in query_lower), None)
            weight_match = next((int(s) for s in query_lower.split() if s.isdigit() and "kg" in query_lower), None)
            age_match = next((int(s) for s in query_lower.split() if s.isdigit() and "year" in query_lower), None)
            return {"tool": "dosage_calculator", "args": {"drug": drug_match, "weight_kg": weight_match, "age_years": age_match}}
        elif "research" in query_lower or "findings" in query_lower:
            return {"tool": "research_paper_search", "args": {"keywords": query}}
        elif "interaction" in query_lower or "drug compatibility" in query_lower:
            drug_list = []
            if "between" in query_lower:
                parts = query_lower.split("between")
                if len(parts) > 1: 
                    drug_part = parts[1].split("and")
                    drug_list = [d.strip() for d in drug_part if d.strip()]
            return {"tool": "drug_interaction_checker", "args": {"drugs": drug_list}}
        elif "what is" in query_lower or "information on" in query_lower:
            return {"tool": "medical_database", "args": {"query": query}}
        return {"tool": "none", "args": {}}

class MedicalAssistant:
    def __init__(self):
        self.llm_router = LLMRouter()
        self.medical_database_tool = MedicalDatabaseTool()
        self.dosage_calculator_tool = DosageCalculatorTool()
        self.research_paper_search_tool = ResearchPaperSearchTool()
        self.drug_interaction_checker_tool = DrugInteractionCheckerTool()
        self.symptom_checker_tool = SymptomCheckerTool()

    def process_query(self, user_query):
        routing_decision = self.llm_router.route_query(user_query)
        tool_name = routing_decision.get("tool")
        tool_args = routing_decision.get("args", {})
        
        result = ""
        if tool_name == "medical_database":
            result = self.medical_database_tool.get_info(tool_args.get("query"))
        elif tool_name == "dosage_calculator":
            drug = tool_args.get("drug")
            weight = tool_args.get("weight_kg")
            age = tool_args.get("age_years")
            if drug and weight and age is not None:
                result = self.dosage_calculator_tool.calculate_dosage(drug, weight, age)
            else:
                result = "Please provide drug, patient weight (kg), and age (years) for dosage calculation."
        elif tool_name == "research_paper_search":
            result = self.research_paper_search_tool.search_papers(tool_args.get("keywords"))
            if result:
                formatted_results = [f"Title: {item['title']}\nSummary: {item['summary']}" for item in result]
                result = "\n\n".join(formatted_results)
            else:
                result = "No research papers found for your keywords."
        elif tool_name == "drug_interaction_checker":
            result = self.drug_interaction_checker_tool.check_interactions(tool_args.get("drugs", []))
        elif tool_name == "symptom_checker":
            result = self.symptom_checker_tool.check_symptoms(tool_args.get("symptoms", []))
        else:
            result = "I'm sorry, I couldn't understand your request or find a suitable tool for it."
        
        return f"Assistant: {result}"

if __name__ == "__main__":
    assistant = MedicalAssistant()

    print(assistant.process_query("What are the symptoms of pneumonia?"))
    print(assistant.process_query("Calculate Amoxicillin dosage for a child weighing 22 kg and 5 years old."))
    print(assistant.process_query("Summarize latest research on cystic fibrosis gene therapy from past 6 months."))
    print(assistant.process_query("Check drug interaction between Ibuprofen and Warfarin."))
    print(assistant.process_query("My symptoms are fever, cough, and shortness of breath. What could it be?"))
    print(assistant.process_query("Information on diabetes facts."))
    print(assistant.process_query("Tell me a joke."))
    print(assistant.process_query("Check drug interaction between Paracetamol and Aspirin."))
    print(assistant.process_query("Calculate drug dosage for a 50 kg adult."))
    print(assistant.process_query("My symptoms are headache and nausea."))
