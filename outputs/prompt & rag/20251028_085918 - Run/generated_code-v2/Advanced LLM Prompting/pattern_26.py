class FewShotFAQSystem:
    def __init__(self):
        self.exemplars = [
            {"question": "What is the new 'AeroGlide' drone's battery life?", "answer": "The AeroGlide drone has a battery life of approximately 30 minutes on a full charge."},
            {"question": "What are the key features of the 'EcoSmart' smart home hub?", "answer": "The EcoSmart hub features voice control, energy monitoring, and compatibility with over 100 smart devices."},
            {"question": "How much does the 'AquaFlow' water filter cost?", "answer": "The AquaFlow water filter costs $49.99 and includes one replaceable cartridge."}
        ]
        self.instruction_prefix = "You are an AI assistant for E-commerce customer support. Answer questions about products based on the provided examples."
        self.exemplar_prefix = "Here are some examples of questions and answers:"

    def _construct_few_shot_prompt(self, user_query):
        prompt_parts = [self.instruction_prefix, self.exemplar_prefix]
        
        for ex in self.exemplars:
            prompt_parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
        
        prompt_parts.append(f"Q: {user_query}\nA:")
        
        return "\n\n".join(prompt_parts)

    def _simulate_llm_response(self, prompt):
        user_query_start_marker = "\nQ: "
        user_query_end_marker = "\nA:"
        
        last_q_index = prompt.rfind(user_query_start_marker)
        
        if last_q_index != -1:
            query_start = last_q_index + len(user_query_start_marker)
            query_end = prompt.rfind(user_query_end_marker, query_start)
            
            if query_end != -1 and query_end > query_start:
                actual_user_query = prompt[query_start:query_end].strip()
                
                for ex in self.exemplars:
                    if actual_user_query.lower() in ex['question'].lower():
                        return ex['answer']
                
        return "Thank you for your question. While I don't have an exact match in my current product FAQs, I am designed to learn from examples. A real AI would now attempt to generate a relevant answer based on the context provided in the prompt."

    def get_answer(self, customer_query):
        full_prompt = self._construct_few_shot_prompt(customer_query)
        simulated_answer = self._simulate_llm_response(full_prompt)
        return simulated_answer

def run_faq_system_demo():
    system = FewShotFAQSystem()
    print("Welcome to the E-commerce Customer Support FAQ System (Few-Shot Prompting Demo)")
    print("Type 'exit' to quit.")

    while True:
        customer_query = input("\nCustomer: ").strip()
        if customer_query.lower() == 'exit':
            break

        assistant_response = system.get_answer(customer_query)
        print(f"Assistant: {assistant_response}")

if __name__ == "__main__":
    run_faq_system_demo()
