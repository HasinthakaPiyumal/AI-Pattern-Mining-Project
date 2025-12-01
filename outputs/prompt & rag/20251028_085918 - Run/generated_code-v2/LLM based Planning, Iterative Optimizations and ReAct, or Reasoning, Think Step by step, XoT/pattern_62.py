class Chatbot:
    def __init__(self, llm_model=None):
        self.llm_model = llm_model

    def _get_user_input(self, prompt="You: "):
        return input(prompt)

    def _construct_plan_and_solve_prompt(self, user_query: str) -> str:
        system_instruction = "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step by step."
        full_prompt = f"{system_instruction}\n\nProblem: {user_query}\n\nPlan and Solution:"
        return full_prompt

    def _call_llm(self, prompt: str) -> str:
        if self.llm_model:
            pass
        
        if "troubleshoot internet connection" in prompt.lower():
            return (
                "Understanding the problem: The user needs to troubleshoot their internet connection.\n\n"
                "Plan:\n"
                "1. Check physical connections.\n"
                "2. Restart router/modem.\n"
                "3. Check network adapter status.\n"
                "4. Contact ISP if problem persists.\n\n"
                "Solution: Please start by checking if all cables are securely connected to your modem and router. Then, try restarting both devices by unplugging them for 30 seconds and plugging them back in. If the issue continues, please describe what lights are on your modem and router."
            )
        elif "reset password" in prompt.lower():
            return (
                "Understanding the problem: The user wants to reset their password.\n\n"
                "Plan:\n"
                "1. Direct user to password reset page.\n"
                "2. Explain the reset process (email verification).\n"
                "3. Offer further assistance if issues arise.\n\n"
                "Solution: To reset your password, please visit our password reset page at [link to reset page]. Enter your registered email address, and we will send you a link to set a new password. Make sure to check your spam folder if you don't receive the email within a few minutes."
            )
        else:
            return (
                "Understanding the problem: I need to understand your request fully.\n\n"
                "Plan:\n"
                "1. Clarify the user's need.\n"
                "2. Formulate a step-by-step solution based on clarification.\n\n"
                "Solution: Could you please provide more details about your problem? This will help me devise a more specific plan to assist you."
            )

    def run(self):
        print("Welcome to the AI Customer Support Chatbot!")
        print("Type 'exit' to end the conversation.")

        while True:
            user_query = self._get_user_input()
            if user_query.lower() == 'exit':
                print("Thank you for using our support. Goodbye!")
                break

            prompt = self._construct_plan_and_solve_prompt(user_query)
            llm_response = self._call_llm(prompt)
            print(f"\nAI: {llm_response}\n")

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.run()