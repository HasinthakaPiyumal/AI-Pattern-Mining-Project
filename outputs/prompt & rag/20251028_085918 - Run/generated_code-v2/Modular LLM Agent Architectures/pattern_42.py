import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class WorkingMemoryState(BaseModel):
    user_query: Optional[str] = None
    dialog_history: List[Dict[str, str]] = []  # [{'role': 'user', 'content': '...'}]
    external_evidence: Optional[str] = None
    attempted_solutions: List[str] = []
    llm_candidate_responses: List[str] = []
    policy_decisions: List[str] = []


class WorkingMemory:
    def __init__(self):
        self._state = WorkingMemoryState()

    def update_memory(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                current_value = getattr(self._state, key)
                if isinstance(current_value, list) and isinstance(value, list):
                    current_value.extend(value)
                elif isinstance(current_value, list) and not isinstance(value, list):
                    current_value.append(value)
                else:
                    setattr(self._state, key, value)

    def get_context(self) -> WorkingMemoryState:
        return self._state

    def clear_memory(self):
        self._state = WorkingMemoryState()

    def add_dialog_turn(self, role: str, content: str):
        self._state.dialog_history.append({"role": role, "content": content})


PRODUCT_KNOWLEDGE = {
    "internet connection": {
        "diagnosis": "Check router, cables, and Wi-Fi settings.",
        "solution_steps": [
            "1. Restart your router and modem.",
            "2. Check if all cables are securely connected.",
            "3. Forget and reconnect to your Wi-Fi network.",
            "4. Try connecting with an Ethernet cable."
        ]
    },
    "printer not printing": {
        "diagnosis": "Check power, paper, ink levels, and print queue.",
        "solution_steps": [
            "1. Ensure the printer is turned on and has paper.",
            "2. Check ink or toner levels.",
            "3. Clear the print queue on your computer.",
            "4. Reinstall printer drivers."
        ]
    },
    "software crash": {
        "diagnosis": "Restart application, check for updates, verify system requirements.",
        "solution_steps": [
            "1. Close and reopen the application.",
            "2. Check for available updates for the software.",
            "3. Restart your computer.",
            "4. Reinstall the problematic software."
        ]
    },
}


class CustomerSupportAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.memory = WorkingMemory()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        self.system_prompt_template = """
        You are a helpful customer support assistant. Your goal is to assist users in troubleshooting issues.
        Maintain a polite and supportive tone. Use the provided context and knowledge to formulate responses.

        Current Dialog History:
        {dialog_history}

        Relevant External Knowledge:
        {external_evidence}

        Attempted Solutions So Far:
        {attempted_solutions}

        Based on the user's query and the above context, provide a solution or ask clarifying questions.
        If a solution is suggested, phrase it as a step the user should 'try' or a 'solution'.
        """

    def _retrieve_knowledge(self, query: str) -> Optional[str]:
        relevant_knowledge = []
        for topic, info in PRODUCT_KNOWLEDGE.items():
            if topic in query.lower():
                relevant_knowledge.append(f"Topic: {topic}\nDiagnosis: {info['diagnosis']}\nSolution Steps: {'\n'.join(info['solution_steps'])}")
        return "\n\n".join(relevant_knowledge) if relevant_knowledge else None

    def _construct_prompt(self) -> List[Any]:
        context = self.memory.get_context()

        formatted_dialog_history = "\n".join(
            [f"{d['role'].capitalize()}: {d['content']}" for d in context.dialog_history]
        ) if context.dialog_history else "No prior dialog."

        formatted_attempted_solutions = "\n".join(
            [f"- {s}" for s in context.attempted_solutions]
        ) if context.attempted_solutions else "None yet."

        system_message_content = self.system_prompt_template.format(
            dialog_history=formatted_dialog_history,
            external_evidence=context.external_evidence if context.external_evidence else "None.",
            attempted_solutions=formatted_attempted_solutions
        )

        messages = [SystemMessage(content=system_message_content)]
        # Add only the most recent user query as a HumanMessage to avoid redundancy with dialog_history in system prompt
        if context.user_query:
            messages.append(HumanMessage(content=context.user_query))

        return messages

    def _apply_policy(self, llm_response: str):
        # Simple policy: if response suggests a solution, add it to attempted_solutions
        if any(keyword in llm_response.lower() for keyword in ["try", "solution", "step"]) and "?" not in llm_response:
            self.memory.update_memory(attempted_solutions=llm_response)
            self.memory.update_memory(policy_decisions="Added LLM response to attempted solutions.")

    def handle_message(self, user_message: str) -> str:
        # 1. Update Working Memory with new user query
        self.memory.update_memory(user_query=user_message)
        self.memory.add_dialog_turn(role="user", content=user_message)

        # 2. Retrieve relevant knowledge
        external_evidence = self._retrieve_knowledge(user_message)
        if external_evidence:
            self.memory.update_memory(external_evidence=external_evidence)

        # 3. Construct detailed prompt for LLM
        prompt_messages = self._construct_prompt()

        # 4. Invoke LLM to get a response
        llm_response_obj = self.llm(prompt_messages)
        llm_response = llm_response_obj.content
        self.memory.update_memory(llm_candidate_responses=llm_response)

        # 5. Apply policy to update Working Memory
        self._apply_policy(llm_response)

        # 6. Update Working Memory with agent's response
        self.memory.add_dialog_turn(role="assistant", content=llm_response)

        return llm_response

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        print("You can get one from https://platform.openai.com/account/api-keys")
        return

    print("\n===== Multi-turn Customer Support Virtual Assistant =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("------------------------------------------------------")

    agent = CustomerSupportAgent()

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break

        response = agent.handle_message(user_input)
        print(f"Assistant: {response}")

        # Optional: Print current memory state for debugging
        # print("\n--- Current Memory State ---")
        # print(agent.memory.get_context().model_dump_json(indent=2))
        # print("--------------------------")


if __name__ == "__main__":
    main()