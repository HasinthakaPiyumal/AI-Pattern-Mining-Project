import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


class RuleBasedPolicy:
    def __init__(self, rules_filepath="rules.json"):
        self.rules = self._load_rules(rules_filepath)

    def _load_rules(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return {
            "shipping status": "Please provide your order number to check the shipping status.",
            "return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be unused and in original packaging.",
            "contact support": "You can contact our support team at support@example.com or call us at 1-800-123-4567."
        }

    def get_response(self, query):
        query_lower = query.lower()
        for rule_keyword, response in self.rules.items():
            if rule_keyword in query_lower:
                return response
        return None


class LLMAgent:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name)
        self.policy_context = "You are a helpful customer support agent for an e-commerce company. Provide concise and polite answers."

    def update_policy_context(self, new_context):
        self.policy_context = new_context

    def generate_response(self, query):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.policy_context),
            HumanMessage(content=query)
        ])
        chain = prompt | self.llm
        response = chain.invoke({"query": query})
        return response.content


class UserSimulator:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name)

    def simulate_query(self):
        prompt_template = "Generate a realistic customer support query for an e-commerce company. Keep it short and to the point."
        response = self.llm.invoke(prompt_template)
        return response.content

    def simulate_feedback(self, agent_response, original_query):
        prompt_template = f"Given the customer query: '{original_query}' and the agent's response: '{agent_response}', generate a customer's feedback. Indicate if the customer is 'satisfied', 'needs more info', or 'dissatisfied'. Example: 'satisfied: Thank you, that was helpful.'"
        response = self.llm.invoke(prompt_template)
        feedback_content = response.content.lower()
        if "satisfied" in feedback_content:
            return "satisfied", feedback_content
        elif "needs more info" in feedback_content:
            return "needs more info", feedback_content
        else:
            return "dissatisfied", feedback_content


class StagedPolicyLearner:
    def __init__(self, openai_api_key):
        self.rule_based_policy = RuleBasedPolicy()
        self.llm_agent = LLMAgent(openai_api_key)
        self.user_simulator = UserSimulator(openai_api_key)
        self.current_policy_stage = "bootstrapping"
        self.simulated_interactions_log = []

    def stage_bootstrapping(self):
        self.current_policy_stage = "bootstrapping"
        print("\n--- Stage 1: Bootstrapping from Rule-Based Policy ---")
        print("Agent is now using predefined rules.")

    def _evaluate_and_update_policy_from_sim(self):
        successful_interactions = [f"Customer asked: '{q}', Agent responded: '{a}'" 
                                   for q, a, f_type, _ in self.simulated_interactions_log 
                                   if f_type == "satisfied"]
        
        if successful_interactions:
            new_context = "You are a helpful customer support agent for an e-commerce company. Focus on providing clear and direct answers. Learn from these examples of successful interactions:\n" + "\n".join(successful_interactions)
            self.llm_agent.update_policy_context(new_context)
            print("Agent policy context updated based on successful simulated interactions.")
        else:
            print("No successful simulated interactions to learn from yet.")

    def stage_simulation_learning(self, num_interactions=10):
        self.current_policy_stage = "simulation_learning"
        print(f"\n--- Stage 2: Learning with User Simulators ({num_interactions} interactions) ---")
        for i in range(num_interactions):
            query = self.user_simulator.simulate_query()
            print(f"Simulator Query [{i+1}/{num_interactions}]: {query}")
            
            agent_response = self.llm_agent.generate_response(query) 
            
            # If in bootstrapping, try rule-based first
            if self.current_policy_stage == "bootstrapping":
                rule_response = self.rule_based_policy.get_response(query)
                if rule_response:
                    agent_response = rule_response

            print(f"Agent Response: {agent_response}")
            
            feedback_type, feedback_detail = self.user_simulator.simulate_feedback(agent_response, query)
            self.simulated_interactions_log.append((query, agent_response, feedback_type, feedback_detail))
            print(f"Simulator Feedback: [{feedback_type}] {feedback_detail}")
            print("---------------------")
        
        self._evaluate_and_update_policy_from_sim()

    def stage_human_refinement(self, human_feedback_data):
        self.current_policy_stage = "human_refinement"
        print("\n--- Stage 3: Refinement with Human Users ---")
        print("Integrating human feedback for policy refinement.")

        refined_examples = []
        for item in human_feedback_data:
            query = item["query"]
            human_approved_response = item["response"]
            refined_examples.append(f"Customer asked: '{query}', Human-approved response: '{human_approved_response}'")
        
        if refined_examples:
            current_context = self.llm_agent.policy_context
            new_context = current_context + "\n\nFurther refine your responses based on these human-approved examples:\n" + "\n".join(refined_examples)
            self.llm_agent.update_policy_context(new_context)
            print("Agent policy context updated with human feedback.")
        else:
            print("No human feedback provided for refinement.")

    def interact(self, query):
        # Prioritize rule-based if applicable in bootstrapping or initial stages
        if self.current_policy_stage == "bootstrapping":
            rule_response = self.rule_based_policy.get_response(query)
            if rule_response:
                return rule_response
        
        # Fallback to LLM agent with current learned policy
        return self.llm_agent.generate_response(query)

    def save_policy_context(self, filename="llm_agent_policy.txt"):
        with open(filename, "w") as f:
            f.write(self.llm_agent.policy_context)
        print(f"Agent policy context saved to {filename}")

    def load_policy_context(self, filename="llm_agent_policy.txt"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.llm_agent.update_policy_context(f.read())
            print(f"Agent policy context loaded from {filename}")
        else:
            print(f"Policy context file {filename} not found.")


if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable or replace 'os.getenv' directly
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    learner = StagedPolicyLearner(openai_api_key)

    # Stage 1: Bootstrapping
    learner.stage_bootstrapping()
    print("\nTest interaction during Bootstrapping Stage:")
    print(f"Customer: What is your return policy?")
    print(f"Agent: {learner.interact('What is your return policy?')}")
    print(f"Customer: I need help with my order.")
    print(f"Agent: {learner.interact('I need help with my order.')}")

    # Stage 2: Learning with User Simulators
    learner.stage_simulation_learning(num_interactions=5)
    learner.save_policy_context()

    print("\nTest interaction after Simulation Learning Stage:")
    print(f"Customer: What is the status of my recent order?")
    print(f"Agent: {learner.interact('What is the status of my recent order?')}")
    print(f"Customer: How do I reset my password?")
    print(f"Agent: {learner.interact('How do I reset my password?')}")

    # Stage 3: Refinement with Human Users
    human_data = [
        {"query": "My package hasn't arrived yet.", "response": "Could you please provide your order number so I can check the tracking details for you?"},
        {"query": "Can I change my delivery address?", "response": "Unfortunately, we cannot change the delivery address once an order has been shipped. Please contact the shipping carrier directly."}
    ]
    learner.stage_human_refinement(human_data)
    learner.save_policy_context("llm_agent_refined_policy.txt")

    print("\nTest interaction after Human Refinement Stage:")
    print(f"Customer: My package hasn't arrived yet.")
    print(f"Agent: {learner.interact('My package hasn\'t arrived yet.')}")
    print(f"Customer: How can I track my order?")
    print(f"Agent: {learner.interact('How can I track my order?')}")

    # Demonstrate loading a saved policy
    print("\nLoading a previously saved policy context:")
    new_learner = StagedPolicyLearner(openai_api_key)
    new_learner.load_policy_context("llm_agent_refined_policy.txt")
    print(f"Customer: Can I return a faulty item?")
    print(f"Agent: {new_learner.interact('Can I return a faulty item?')}")
