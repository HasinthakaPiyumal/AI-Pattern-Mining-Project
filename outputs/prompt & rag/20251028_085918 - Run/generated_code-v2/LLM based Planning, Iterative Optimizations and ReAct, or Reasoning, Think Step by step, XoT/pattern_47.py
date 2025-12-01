from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class SmartCustomerSupportAgent:
    def __init__(self, llm_model_name: str = "gpt-3.5-turbo", max_clarification_turns: int = 3):
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0.0)

        self.DECISION_PROMPT = PromptTemplate(
            template="""Based on the initial customer query: "{initial_query}"
And the current clarifications:
{clarifications_history}
Do you need more information to provide a comprehensive answer? Respond with 'YES' or 'NO' followed by a brief reason.""",
            input_variables=["initial_query", "clarifications_history"],
        )

        self.QUESTION_GENERATION_PROMPT = PromptTemplate(
            template="""Given the initial query: "{initial_query}"
And existing clarifications:
{clarifications_history}
What is a single, clear clarifying question you need to ask the customer to gather more information? Respond only with the question, no introductory phrases.""",
            input_variables=["initial_query", "clarifications_history"],
        )

        self.FINAL_ANSWER_PROMPT = PromptTemplate(
            template="""Considering the initial customer query: "{initial_query}"
And all gathered clarifications:
{clarifications_history}
Please provide a comprehensive and accurate answer to the original query.""",
            input_variables=["initial_query", "clarifications_history"],
        )
        self.max_clarification_turns = max_clarification_turns
        self.initial_query = ""
        self.clarifications_history = []

    def _format_clarifications_history(self) -> str:
        if not self.clarifications_history:
            return "No clarifications yet."
        formatted_history = ""
        for i, (q, a) in enumerate(self.clarifications_history):
            formatted_history += f"Q{i+1}: {q}\nA{i+1}: {a}\n"
        return formatted_history.strip()

    def _call_llm(self, prompt_template: PromptTemplate, input_variables: dict) -> str:
        formatted_prompt = prompt_template.format(**input_variables)
        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()

    def handle_query(self, initial_query: str) -> str:
        self.initial_query = initial_query
        self.clarifications_history = []
        print(f"Agent received query: {initial_query}")

        for turn in range(self.max_clarification_turns):
            current_clarifications = self._format_clarifications_history()
            
            decision_response = self._call_llm(
                self.DECISION_PROMPT,
                {"initial_query": self.initial_query, "clarifications_history": current_clarifications},
            ).upper()

            if "YES" in decision_response:
                print("Agent needs more information.")
                question_to_ask = self._call_llm(
                    self.QUESTION_GENERATION_PROMPT,
                    {"initial_query": self.initial_query, "clarifications_history": current_clarifications},
                )
                print(f"Agent asks: {question_to_ask}")
                customer_answer = input("Your answer: ")
                self.clarifications_history.append((question_to_ask, customer_answer))
            else:
                print("Agent has enough information or reached clarification limit.")
                break
        
        final_answer = self._call_llm(
            self.FINAL_ANSWER_PROMPT,
            {"initial_query": self.initial_query, "clarifications_history": self._format_clarifications_history()},
        )
        print("\n--- Final Answer ---")
        print(final_answer)
        return final_answer

if __name__ == "__main__":
    agent = SmartCustomerSupportAgent(max_clarification_turns=2)
    agent.handle_query("I have a problem with my order.")