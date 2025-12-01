import networkx as nx
import json

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_sample_data()

    def _populate_sample_data(self):
        self.add_node("MedicalCondition", "Influenza", {"description": "A common viral infection that can be deadly."})
        self.add_node("Symptom", "Fever", {"severity": "high"})
        self.add_node("Symptom", "Cough", {"type": "dry"})
        self.add_node("Symptom", "Sore Throat", {})
        self.add_node("Symptom", "Fatigue", {})
        self.add_node("MedicalCondition", "Common Cold", {"description": "A viral infectious disease of the upper respiratory tract."})
        self.add_node("MedicalCondition", "Diabetes Type 2", {"description": "A chronic condition that affects the way the body processes blood sugar (glucose)."})
        self.add_node("Symptom", "Increased Thirst", {})
        self.add_node("Symptom", "Frequent Urination", {})
        self.add_node("Drug", "Paracetamol", {"class": "Pain Reliever"})
        self.add_node("Drug", "Amoxicillin", {"class": "Antibiotic"})
        self.add_node("Drug", "Metformin", {"class": "Antidiabetic"})
        self.add_node("Drug", "Insulin", {"class": "Hormone"})
        self.add_node("ResearchPaper", "Flu Vaccine Efficacy 2023", {"year": 2023, "journal": "NEJM", "summary": "Study showing 60% efficacy of current flu vaccine."})
        self.add_node("ResearchPaper", "Metformin Side Effects", {"year": 2022, "journal": "Lancet", "summary": "Review of common gastrointestinal side effects."})

        self.add_relationship("Influenza", "HAS_SYMPTOM", "Fever")
        self.add_relationship("Influenza", "HAS_SYMPTOM", "Cough")
        self.add_relationship("Influenza", "HAS_SYMPTOM", "Fatigue")
        self.add_relationship("Common Cold", "HAS_SYMPTOM", "Cough")
        self.add_relationship("Common Cold", "HAS_SYMPTOM", "Sore Throat")
        self.add_relationship("Diabetes Type 2", "HAS_SYMPTOM", "Increased Thirst")
        self.add_relationship("Diabetes Type 2", "HAS_SYMPTOM", "Frequent Urination")

        self.add_relationship("Paracetamol", "TREATS", "Fever")
        self.add_relationship("Metformin", "TREATS", "Diabetes Type 2")
        self.add_relationship("Insulin", "TREATS", "Diabetes Type 2")

        self.add_relationship("Amoxicillin", "CAUSES_INTERACTION_WITH", "Metformin", {"severity": "moderate"})
        self.add_relationship("Flu Vaccine Efficacy 2023", "RELATES_TO", "Influenza")
        self.add_relationship("Metformin Side Effects", "RELATES_TO", "Metformin")

    def add_node(self, node_type, node_name, properties=None):
        if properties is None:
            properties = {}
        self.graph.add_node(node_name, type=node_type, **properties)

    def add_relationship(self, source, relation_type, target, properties=None):
        if properties is None:
            properties = {}
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, type=relation_type, **properties)
        else:
            pass

    def get_symptoms_for_condition(self, condition):
        symptoms = []
        if condition in self.graph:
            for neighbor in self.graph.neighbors(condition):
                if self.graph.has_edge(condition, neighbor) and self.graph.edges[condition, neighbor]["type"] == "HAS_SYMPTOM":
                    symptoms.append(neighbor)
        return symptoms

    def get_conditions_for_symptoms(self, symptoms):
        matching_conditions = set()
        for symptom in symptoms:
            if symptom in self.graph:
                for neighbor in self.graph.predecessors(symptom):
                    if self.graph.has_edge(neighbor, symptom) and self.graph.edges[neighbor, symptom]["type"] == "HAS_SYMPTOM" and self.graph.nodes[neighbor]["type"] == "MedicalCondition":
                        matching_conditions.add(neighbor)
        return list(matching_conditions)

    def get_drug_interactions(self, drug_list):
        interactions = []
        for i in range(len(drug_list)):
            for j in range(i + 1, len(drug_list)):
                drug1 = drug_list[i]
                drug2 = drug_list[j]
                if drug1 in self.graph and drug2 in self.graph:
                    if self.graph.has_edge(drug1, drug2) and self.graph.edges[drug1, drug2]["type"] == "CAUSES_INTERACTION_WITH":
                        interactions.append({ "drug1": drug1, "drug2": drug2, "severity": self.graph.edges[drug1, drug2].get("severity", "unknown") })
                    if self.graph.has_edge(drug2, drug1) and self.graph.edges[drug2, drug1]["type"] == "CAUSES_INTERACTION_WITH":
                        interactions.append({ "drug1": drug2, "drug2": drug1, "severity": self.graph.edges[drug2, drug1].get("severity", "unknown") })
        return interactions

    def get_latest_research(self, topic):
        research_papers = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["type"] == "ResearchPaper" and topic.lower() in node.lower():
                research_papers.append({ "title": node, **{k:v for k,v in self.graph.nodes[node].items() if k not in ['type']} })
            elif self.graph.nodes[node]["type"] == "ResearchPaper":
                for neighbor in self.graph.neighbors(node):
                    if self.graph.has_edge(node, neighbor) and self.graph.edges[node, neighbor]["type"] == "RELATES_TO" and topic.lower() in neighbor.lower():
                        research_papers.append({ "title": node, **{k:v for k,v in self.graph.nodes[node].items() if k not in ['type']} })

        return research_papers

    def get_condition_details(self, condition_name):
        if condition_name in self.graph and self.graph.nodes[condition_name].get("type") == "MedicalCondition":
            return { "name": condition_name, **{k:v for k,v in self.graph.nodes[condition_name].items() if k not in ['type']} }
        return None


class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs):
        return self.func(**kwargs)

class LLMAgent:
    def __init__(self, kg: KnowledgeGraph, tools: list):
        self.kg = kg
        self.tools = {tool.name: tool for tool in tools}

    def process_query(self, user_query: str):
        user_query_lower = user_query.lower()

        if "symptoms for" in user_query_lower and "condition" in user_query_lower:
            condition = user_query.split("symptoms for ")[1].split(" condition")[0].strip()
            if "medical condition" not in self.tools:
                return "Error: Medical condition lookup tool not available."
            condition_details = self.tools["MedicalConditionLookup"].run(condition_name=condition)
            if condition_details:
                symptoms = self.tools["GetSymptomsForCondition"].run(condition=condition)
                return f"Symptoms for {condition}: {', '.join(symptoms)}."
            else:
                return f"Condition '{condition}' not found."

        elif "conditions for symptoms" in user_query_lower:
            symptom_str = user_query.split("conditions for symptoms ")[1].strip()
            symptoms = [s.strip() for s in symptom_str.split(',')]
            if "SymptomChecker" not in self.tools:
                return "Error: Symptom checker tool not available."
            conditions = self.tools["SymptomChecker"].run(symptoms=symptoms)
            if conditions:
                return f"Possible conditions for {', '.join(symptoms)}: {', '.join(conditions)}."
            else:
                return f"No conditions found for the given symptoms: {', '.join(symptoms)}."

        elif "drug interactions for" in user_query_lower:
            drug_str = user_query.split("drug interactions for ")[1].strip()
            drugs = [d.strip() for d in drug_str.split(',')]
            if "DrugInteractionChecker" not in self.tools:
                return "Error: Drug interaction checker tool not available."
            interactions = self.tools["DrugInteractionChecker"].run(drug_list=drugs)
            if interactions:
                formatted_interactions = []
                for interaction in interactions:
                    formatted_interactions.append(f"{interaction['drug1']} and {interaction['drug2']} (severity: {interaction['severity']})")
                return f"Drug interactions for {', '.join(drugs)}: {'; '.join(formatted_interactions)}."
            else:
                return f"No known drug interactions found for {', '.join(drugs)}."

        elif "latest research on" in user_query_lower:
            topic = user_query.split("latest research on ")[1].strip()
            if "ResearchPaperSearch" not in self.tools:
                return "Error: Research paper search tool not available."
            research = self.tools["ResearchPaperSearch"].run(topic=topic)
            if research:
                research_titles = [r['title'] for r in research]
                return f"Latest research on {topic}: {'; '.join(research_titles)}."
            else:
                return f"No research found on {topic}."

        elif "details of condition" in user_query_lower:
            condition = user_query.split("details of condition ")[1].strip()
            if "MedicalConditionLookup" not in self.tools:
                return "Error: Medical condition lookup tool not available."
            details = self.tools["MedicalConditionLookup"].run(condition_name=condition)
            if details:
                return f"Details for {condition}: {json.dumps(details, indent=2)}."
            else:
                return f"Condition '{condition}' not found."

        return "I can help with medical condition symptoms, conditions for symptoms, drug interactions, or latest research. Please rephrase your query."

if __name__ == "__main__":
    # 1. Initialize Knowledge Graph
    medical_kg = KnowledgeGraph()

    # 2. Define Tools that interact with the KG
    get_symptoms_tool = Tool(
        name="GetSymptomsForCondition",
        description="Retrieves symptoms associated with a given medical condition.",
        func=medical_kg.get_symptoms_for_condition
    )

    symptom_checker_tool = Tool(
        name="SymptomChecker",
        description="Identifies potential medical conditions based on a list of symptoms.",
        func=medical_kg.get_conditions_for_symptoms
    )

    drug_interaction_checker_tool = Tool(
        name="DrugInteractionChecker",
        description="Checks for interactions between a list of specified drugs.",
        func=medical_kg.get_drug_interactions
    )

    research_paper_search_tool = Tool(
        name="ResearchPaperSearch",
        description="Searches for the latest research papers related to a specific medical topic.",
        func=medical_kg.get_latest_research
    )

    medical_condition_lookup_tool = Tool(
        name="MedicalConditionLookup",
        description="Retrieves detailed information about a specific medical condition.",
        func=medical_kg.get_condition_details
    )

    # 3. Initialize LLM Agent with the KG and Tools
    agent_tools = [
        get_symptoms_tool,
        symptom_checker_tool,
        drug_interaction_checker_tool,
        research_paper_search_tool,
        medical_condition_lookup_tool
    ]
    llm_agent = LLMAgent(kg=medical_kg, tools=agent_tools)

    # 4. Simulate User Interaction
    print("\n--- Medical Diagnosis and Treatment Recommendation System ---")
    print("Ask me about medical conditions, symptoms, drug interactions, or research.")
    print("Example queries:")
    print(" - What are the symptoms for Influenza condition?")
    print(" - What are the conditions for symptoms Fever, Cough?")
    print(" - Check drug interactions for Paracetamol, Amoxicillin?")
    print(" - Find latest research on Flu Vaccine?")
    print(" - Tell me details of condition Diabetes Type 2?")

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() == 'exit':
            break
        response = llm_agent.process_query(user_input)
        print(f"Agent: {response}")

    print("Exiting system. Goodbye!")
