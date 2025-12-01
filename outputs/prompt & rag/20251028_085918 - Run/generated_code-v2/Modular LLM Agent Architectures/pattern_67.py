class MedicalKnowledgeModule:
    def __init__(self):
        self.knowledge_base = {
            "metformin side effects": "Metformin can cause nausea, diarrhea, stomach upset, and in rare cases, lactic acidosis. It is primarily used for Type 2 Diabetes.",
            "type 2 diabetes treatment": "Treatment for Type 2 Diabetes often includes lifestyle changes (diet, exercise), oral medications like Metformin, and sometimes insulin. Regular blood sugar monitoring is crucial.",
            "hypertension causes": "Hypertension (high blood pressure) can be caused by genetics, obesity, lack of physical activity, high salt intake, excessive alcohol consumption, and stress.",
            "aspirin uses": "Aspirin is commonly used as a pain reliever, fever reducer, and anti-inflammatory drug. Low doses are also prescribed to prevent heart attacks and strokes."
        }

    def retrieve_medical_facts(self, query):
        relevant_facts = []
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if any(word in query_lower for word in key.split()):
                relevant_facts.append(value)
        return relevant_facts

    def synthesize_medical_context(self, facts):
        if not facts:
            return "No specific medical information found relevant to your query."
        return " ".join(facts) + " Based on this information,"

class LLMSimulator:
    def generate_response(self, original_query, medical_context):
        if "No specific medical information found" in medical_context:
            return f"For your question: '{original_query}', I couldn't find specific medical information in my augmented module. I will try to answer based on my general knowledge.\n(Simulated LLM response): While I don't have specific medical data for that, generally speaking..."
        else:
            return f"Based on the following medical information: {medical_context} answer the question: '{original_query}'\n(Simulated LLM response): Acknowledging the provided medical context, {original_query} can be addressed as follows..."


def main():
    medical_module = MedicalKnowledgeModule()
    llm_simulator = LLMSimulator()

    queries = [
        "What are the side effects of Metformin?",
        "Explain Type 2 Diabetes treatment options.",
        "Causes of high blood pressure?",
        "What is the best treatment for a common cold?"
    ]

    for query in queries:
        print(f"\nUser Query: {query}")
        
        # Step 1: Query goes to Medical Knowledge Module
        retrieved_facts = medical_module.retrieve_medical_facts(query)
        synthesized_context = medical_module.synthesize_medical_context(retrieved_facts)
        
        print(f"Medical Module Output (Context): {synthesized_context}")

        # Step 2: LLM consumes augmented context
        llm_response = llm_simulator.generate_response(query, synthesized_context)
        print(f"Final LLM Response: {llm_response}")

if __name__ == "__main__":
    main()