from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


class Persona(BaseModel):
    name: str
    description: str
    evaluation_criteria: List[str]
    llm: Any  # This would be an actual LLM instance, e.g., from langchain_openai

    def generate_query(self, scenario: str, conversation_history: List[Dict]) -> str:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are {self.name}. {self.description}. Your goal is to interact with a customer support chatbot based on the following scenario and evaluate its performance against these criteria: {', '.join(self.evaluation_criteria)}. Here is the conversation history:\n{history_str}"),
                ("user", f"Given the scenario: {scenario}, generate your next query or statement for the chatbot.")
            ]
        )
        chain = prompt_template | self.llm
        response = chain.invoke({"scenario": scenario, "history_str": history_str})
        return response.content

    def evaluate_response(self, chatbot_response: str, scenario: str, conversation_history: List[Dict]) -> Dict:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are {self.name}. {self.description}. Evaluate the following chatbot response based on the scenario: {scenario} and your evaluation criteria: {', '.join(self.evaluation_criteria)}. Provide a score from 1-5 for each criterion and a brief explanation. Here is the conversation history and the chatbot's last response:\n{history_str}\nChatbot: {chatbot_response}"),
                ("user", "Provide your evaluation in a JSON format with keys as criterion names and values as a dictionary with 'score' (1-5) and 'explanation'. Also include an overall_sentiment (positive, neutral, negative) and overall_comment.")
            ]
        )
        chain = prompt_template | self.llm
        response = chain.invoke({"scenario": scenario, "history_str": history_str, "chatbot_response": chatbot_response})
        # In a real scenario, you'd parse this JSON response more robustly
        try:
            return eval(response.content) # Using eval for simplicity, use json.loads in production
        except Exception:
            return {"error": "Could not parse evaluation", "raw_response": response.content}


class MockCustomerSupportChatbot:
    def __init__(self, name="MockChatbot"):
        self.name = name

    def get_response(self, user_query: str) -> str:
        if "frustrated" in user_query.lower() or "angry" in user_query.lower():
            return "I understand you're feeling frustrated. Please tell me more about the issue so I can help."
        elif "technical issue" in user_query.lower() or "bug" in user_query.lower():
            return "Can you please provide the error message or steps to reproduce the technical issue?"
        elif "price" in user_query.lower() or "buy" in user_query.lower():
            return "Our pricing details are available on our website. Would you like me to direct you there?"
        elif "hello" in user_query.lower() or "hi" in user_query.lower():
            return "Hello! How can I assist you today?"
        else:
            return "Thank you for your query. I'm connecting you with an agent who can provide more detailed assistance."


class ConversationOrchestrator:
    def __init__(self, personas: List[Persona], target_chatbot: MockCustomerSupportChatbot, max_turns: int = 5):
        self.personas = personas
        self.target_chatbot = target_chatbot
        self.max_turns = max_turns

    def run_scenario(self, scenario: str) -> Dict:
        all_evaluations = {}

        for persona in self.personas:
            print(f"\n--- Starting conversation with {persona.name} persona ---")
            conversation_history = []
            persona_evaluations = []

            for turn in range(self.max_turns):
                print(f"\nTurn {turn + 1}:")

                # Persona generates query
                user_query = persona.generate_query(scenario, conversation_history)
                print(f"{persona.name}: {user_query}")
                conversation_history.append({"role": persona.name, "content": user_query})

                # Target chatbot responds
                chatbot_response = self.target_chatbot.get_response(user_query)
                print(f"{self.target_chatbot.name}: {chatbot_response}")
                conversation_history.append({"role": self.target_chatbot.name, "content": chatbot_response})

                # Persona evaluates response
                evaluation = persona.evaluate_response(chatbot_response, scenario, conversation_history)
                persona_evaluations.append(evaluation)
                print(f"  {persona.name} Evaluation for this turn: {evaluation.get('overall_sentiment', 'N/A')}")

                # Decide if conversation should continue (simplified: always continue for max_turns)

            all_evaluations[persona.name] = {
                "conversation_history": conversation_history,
                "evaluations_per_turn": persona_evaluations
            }
            print(f"--- Finished conversation with {persona.name} persona ---")
        return all_evaluations


class EvaluationModule:
    def __init__(self, all_evaluations: Dict):
        self.all_evaluations = all_evaluations

    def aggregate_results(self) -> Dict:
        aggregated_scores = {}
        overall_comments = []

        for persona_name, data in self.all_evaluations.items():
            for evaluation_turn in data["evaluations_per_turn"]:
                if "error" in evaluation_turn:
                    overall_comments.append(f"Error in {persona_name} evaluation: {evaluation_turn['raw_response']}")
                    continue

                for criterion, details in evaluation_turn.items():
                    if criterion in ["overall_sentiment", "overall_comment"]:
                        if criterion == "overall_comment":
                            overall_comments.append(f"[{persona_name}]: {details}")
                        continue

                    if criterion not in aggregated_scores:
                        aggregated_scores[criterion] = {"total_score": 0, "count": 0, "persona_scores": {}}

                    score = details.get("score")
                    if score is not None and isinstance(score, (int, float)):
                        aggregated_scores[criterion]["total_score"] += score
                        aggregated_scores[criterion]["count"] += 1
                        if persona_name not in aggregated_scores[criterion]["persona_scores"]:
                            aggregated_scores[criterion]["persona_scores"][persona_name] = {"total_score": 0, "count": 0}
                        aggregated_scores[criterion]["persona_scores"][persona_name]["total_score"] += score
                        aggregated_scores[criterion]["persona_scores"][persona_name]["count"] += 1

        final_report = {"overall_average_scores": {}, "persona_average_scores": {}, "overall_comments": overall_comments}

        for criterion, data in aggregated_scores.items():
            if data["count"] > 0:
                final_report["overall_average_scores"][criterion] = data["total_score"] / data["count"]
            
            final_report["persona_average_scores"][criterion] = {}
            for persona_name, persona_data in data["persona_scores"].items():
                if persona_data["count"] > 0:
                    final_report["persona_average_scores"][criterion][persona_name] = persona_data["total_score"] / persona_data["count"]

        return final_report


class ReportingModule:
    def generate_report(self, aggregated_results: Dict):
        print("\n=== Chatbot Evaluation Report ===")

        print("\n--- Overall Average Scores per Criterion ---")
        for criterion, score in aggregated_results["overall_average_scores"].items():
            print(f"- {criterion}: {score:.2f}/5")

        print("\n--- Persona-Specific Average Scores per Criterion ---")
        for criterion, persona_scores in aggregated_results["persona_average_scores"].items():
            print(f"\nCriterion: {criterion}")
            for persona_name, score in persona_scores.items():
                print(f"  - {persona_name}: {score:.2f}/5")

        print("\n--- Overall Comments and Feedback ---")
        for comment in aggregated_results["overall_comments"]:
            print(f"- {comment}")



# Main execution flow (requires an actual LLM setup, e.g., OpenAI API key)
if __name__ == "__main__":
    # Mock LLM for demonstration purposes. 
    # In a real application, you would initialize an actual LLM like ChatOpenAI
    class MockLLM:
        def invoke(self, prompt_template_output):
            # This is a highly simplified mock. Real LLM would parse the prompt and generate a coherent response.
            # For evaluation, it's particularly tricky to mock without actual intelligence.
            if "generate your next query" in prompt_template_output.messages[-1].content:
                if "frustrated" in prompt_template_output.messages[-2].content.lower():
                    return AIMessage(content="I am still very frustrated! Why isn't this working?")
                elif "technical expert" in prompt_template_output.messages[-2].content.lower():
                    return AIMessage(content="What are the exact error logs I should be looking at?")
                elif "sales lead" in prompt_template_output.messages[-2].content.lower():
                    return AIMessage(content="Can you tell me more about the premium features?")
                elif "new user" in prompt_template_output.messages[-2].content.lower():
                    return AIMessage(content="How do I get started with this service?")
                else:
                    return AIMessage(content="Hello, I have a question.")
            elif "Provide your evaluation" in prompt_template_output.messages[-1].content:
                # Simplified mock evaluation. A real LLM would provide a structured JSON.
                if "I understand you're feeling frustrated" in prompt_template_output.messages[-2].content:
                    return AIMessage(content="{'Empathy': {'score': 4, 'explanation': 'Acknowledged frustration.'}, 'Problem Resolution': {'score': 2, 'explanation': 'Did not resolve yet.'}, 'Clarity': {'score': 4, 'explanation': 'Response was clear.'}, 'overall_sentiment': 'neutral', 'overall_comment': 'Good initial empathy.'}")
                elif "error message or steps" in prompt_template_output.messages[-2].content:
                    return AIMessage(content="{'Accuracy': {'score': 5, 'explanation': 'Asked for relevant info.'}, 'Problem Resolution': {'score': 3, 'explanation': 'Moving towards resolution.'}, 'Clarity': {'score': 4, 'explanation': 'Clear question.'}, 'overall_sentiment': 'positive', 'overall_comment': 'Good technical query.'}")
                elif "pricing details" in prompt_template_output.messages[-2].content:
                    return AIMessage(content="{'Accuracy': {'score': 3, 'explanation': 'Directed to website, not direct answer.'}, 'Sales Effectiveness': {'score': 2, 'explanation': 'Missed opportunity to upsell.'}, 'Clarity': {'score': 4, 'explanation': 'Clear suggestion.'}, 'overall_sentiment': 'negative', 'overall_comment': 'Could have been more proactive in sales.'}")
                else:
                    return AIMessage(content="{'Relevance': {'score': 3, 'explanation': 'Generic response.'}, 'Clarity': {'score': 4, 'explanation': 'Clear response.'}, 'overall_sentiment': 'neutral', 'overall_comment': 'A bit too generic.'}")
            return AIMessage(content="This is a mock LLM response.")


    # Initialize Mock LLM
    mock_llm = MockLLM()

    # 1. Define Personas
    personas = [
        Persona(
            name="Frustrated Customer",
            description="An impatient customer facing an issue, seeking quick resolution and empathy.",
            evaluation_criteria=["Empathy", "Problem Resolution", "Speed of Resolution", "Clarity"],
            llm=mock_llm
        ),
        Persona(
            name="Technical Expert",
            description="A knowledgeable user who understands technical terms and expects precise, accurate solutions.",
            evaluation_criteria=["Accuracy", "Technical Depth", "Problem Resolution", "Clarity"],
            llm=mock_llm
        ),
        Persona(
            name="Sales Lead",
            description="A potential customer interested in product features, pricing, and potential upgrades. Looking for sales opportunities.",
            evaluation_criteria=["Sales Effectiveness", "Feature Explanation", "Pricing Clarity", "Responsiveness"],
            llm=mock_llm
        ),
        Persona(
            name="New User",
            description="A beginner user who needs simple explanations and guidance to get started with the service.",
            evaluation_criteria=["Clarity", "Guidance", "Ease of Understanding", "Friendliness"],
            llm=mock_llm
        )
    ]

    # 2. Initialize Target Chatbot
    target_chatbot = MockCustomerSupportChatbot()

    # 3. Initialize Conversation Orchestrator
    orchestrator = ConversationOrchestrator(personas=personas, target_chatbot=target_chatbot, max_turns=2)

    # Define a scenario
    scenario = "The user is experiencing frequent disconnections with their internet service and is very annoyed."

    # Run the evaluation scenario
    all_evaluations = orchestrator.run_scenario(scenario)

    # 4. Process and Aggregate Evaluations
    evaluation_module = EvaluationModule(all_evaluations)
    aggregated_results = evaluation_module.aggregate_results()

    # 5. Generate Report
    reporting_module = ReportingModule()
    reporting_module.generate_report(aggregated_results)