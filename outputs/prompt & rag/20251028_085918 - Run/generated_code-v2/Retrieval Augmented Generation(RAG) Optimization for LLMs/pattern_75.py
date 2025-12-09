class SimulatedLLM:
    def __init__(self):
        self.knowledge_base = {
            "what is the capital of france?": "The capital of France is Paris.",
            "who wrote romeo and juliet?": "William Shakespeare wrote Romeo and Juliet.",
            "what is 2+2?": "2 + 2 equals 4.",
            "what is the largest ocean?": "The largest ocean is the Pacific Ocean.",
            "what is the chemical symbol for water?": "The chemical symbol for water is H2O."
        }

    def generate_answer(self, query: str) -> str:
        normalized_query = query.lower().strip()
        if normalized_query in self.knowledge_base:
            return self.knowledge_base[normalized_query]
        else:
            return "I'm sorry, I can only answer very basic common knowledge questions directly from my internal memory, and I don't have information on that specific query."

class EduBotCLI:
    def __init__(self):
        self.llm_service = SimulatedLLM()

    def start(self):
        print("EduBot: Your Quick Common Knowledge Tutor (No Retrieval Mode)")
        print("Ask a simple common knowledge question or type 'exit' to quit.")
        while True:
            user_query = input("\nYour question: ")
            if user_query.lower() == 'exit':
                print("Goodbye!")
                break
            
            answer = self.llm_service.generate_answer(user_query)
            print(f"EduBot: {answer}")

if __name__ == "__main__":
    cli = EduBotCLI()
    cli.start()