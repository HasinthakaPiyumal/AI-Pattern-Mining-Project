from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# 1. LLM Integration Layer
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. Prompt Management
PLANNING_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert AI assistant tasked with breaking down complex customer support queries into a detailed, step-by-step plan. "
        "The plan should outline the necessary actions to resolve the customer's issue. "
        "Output the plan as a numbered list of concise steps. Do not execute the plan, just create it."
    ),
    HumanMessagePromptTemplate.from_template("Customer Query: {query}")
])

EXECUTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an intelligent customer support chatbot. Your goal is to execute a specific step from a larger plan. "
        "Provide a helpful response to the customer based on the current step and the overall plan. "
        "If you believe you have gathered sufficient information or provided a solution for the current step, "
        "append '[STEP_COMPLETED]' to your response to indicate you are ready to move to the next logical step or conclude. "
        "If you need more information to complete the current step, do not append '[STEP_COMPLETED]'."
    ),
    HumanMessagePromptTemplate.from_template(
        "Full Plan:\n{plan}\n\nCurrent Step ({step_number}/{total_steps}): {current_step_description}\n\n" 
        "Conversation History:\n{history}\n\nCustomer Input: {user_input}"
    )
])

# 3. Core Logic - Plan Generation
class PlanningChain(LLMChain):
    def __init__(self, llm, prompt):
        super().__init__(llm=llm, prompt=prompt, output_key="plan")

    def parse_plan_output(self, llm_output: str) -> list[str]:
        steps = []
        for line in llm_output.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() and '.' in line):
                # Extract the step description, removing the numbering
                step_description = ' '.join(line.split(' ')[1:]).strip()
                if step_description: # Ensure it's not just a number
                    steps.append(step_description)
        return steps

# 4. Core Logic - Step Execution
class ExecutionChain(LLMChain):
    def __init__(self, llm, prompt):
        super().__init__(llm=llm, prompt=prompt, output_key="response")

# 5. Chatbot Orchestration
class PlanSolveChatbot:
    def __init__(self, llm_instance, planning_chain_instance, execution_chain_instance):
        self.llm = llm_instance
        self.planning_chain = planning_chain_instance
        self.execution_chain = execution_chain_instance
        self.plan = []
        self.current_step_index = 0
        self.conversation_history = []
        self.is_active = False

    def _format_history(self):
        formatted_history = []
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"Customer: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Chatbot: {msg.content}")
        return "\n".join(formatted_history)

    def start_new_conversation(self, query: str) -> str:
        self.conversation_history = [HumanMessage(content=query)]
        try:
            plan_raw = self.planning_chain.invoke({"query": query})["plan"]
            self.plan = self.planning_chain.parse_plan_output(plan_raw)
            if not self.plan:
                return "I apologize, I couldn't generate a clear plan for your request. Could you please rephrase it?"
            self.current_step_index = 0
            self.is_active = True
            print(f"\n--- Generated Plan ---")
            for i, step in enumerate(self.plan):
                print(f"{i+1}. {step}")
            print(f"----------------------\n")
            
            # Execute the first step immediately after planning
            return self._execute_current_step(user_input="") # No new user input yet for the first step execution

        except Exception as e:
            self.is_active = False
            return f"An error occurred while starting the conversation: {e}"

    def _execute_current_step(self, user_input: str) -> str:
        if not self.is_active or self.is_conversation_complete():
            return "The conversation is already complete or not active."

        current_step_description = self.plan[self.current_step_index]
        full_plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(self.plan)])
        
        # Add user_input to history if it's not empty, it will be added by process_input before calling this.
        # self.conversation_history.append(HumanMessage(content=user_input))

        execution_response_dict = self.execution_chain.invoke({
            "plan": full_plan_text,
            "current_step_description": current_step_description,
            "step_number": self.current_step_index + 1,
            "total_steps": len(self.plan),
            "history": self._format_history(),
            "user_input": user_input  # The actual user input for this turn
        })
        
        chatbot_response = execution_response_dict["response"]
        
        step_completed_flag = "[STEP_COMPLETED]"
        if step_completed_flag in chatbot_response:
            chatbot_response = chatbot_response.replace(step_completed_flag, "").strip()
            self.current_step_index += 1
        
        self.conversation_history.append(AIMessage(content=chatbot_response))
        return chatbot_response

    def process_input(self, user_input: str) -> str:
        if not self.is_active:
            return "Please start a new conversation first."
        if self.is_conversation_complete():
            return "This conversation is complete. Please start a new one if you have another query."

        self.conversation_history.append(HumanMessage(content=user_input))
        response = self._execute_current_step(user_input)
        return response

    def is_conversation_complete(self) -> bool:
        return self.current_step_index >= len(self.plan)

# 6. User Interface (Console-based for demonstration)
def main():
    planning_chain = PlanningChain(llm=llm, prompt=PLANNING_PROMPT_TEMPLATE)
    execution_chain = ExecutionChain(llm=llm, prompt=EXECUTION_PROMPT_TEMPLATE)
    chatbot = PlanSolveChatbot(llm_instance=llm, planning_chain_instance=planning_chain, execution_chain_instance=execution_chain)

    print("Welcome to the Plan-and-Solve Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        initial_query = input("\nHow can I help you today? (Start a new conversation): ")
        if initial_query.lower() == 'exit':
            break
        
        chatbot_response = chatbot.start_new_conversation(initial_query)
        print(f"Chatbot: {chatbot_response}")

        while chatbot.is_active and not chatbot.is_conversation_complete():
            user_input = input(f"({chatbot.current_step_index + 1}/{len(chatbot.plan)}) Customer: ")
            if user_input.lower() == 'exit':
                break
            
            chatbot_response = chatbot.process_input(user_input)
            print(f"Chatbot: {chatbot_response}")
        
        if chatbot.is_conversation_complete():
            print("Chatbot: Thank you for contacting support. Your issue has been resolved. Do you have any other questions?")

    print("Goodbye!")

if __name__ == "__main__":
    main()
