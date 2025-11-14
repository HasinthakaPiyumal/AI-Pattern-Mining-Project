
import os
import json
from typing import List, Dict

# Assuming OPENAI_API_KEY is set as an environment variable
# from dotenv import load_dotenv
# load_dotenv()

# Placeholder for Langchain components. In a real scenario, you'd install and import them.
# For demonstration, we'll mock the LLM interaction.
class ChatOpenAI:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        print(f"Initialized ChatOpenAI with model: {model_name}")

    def invoke(self, messages: List[Dict]) -> Dict:
        # Mock LLM response for demonstration purposes
        # In a real application, this would call OpenAI API
        # For simplicity, we'll generate a consistent mock response structure.
        
        system_message = messages[0]['content']
        human_message = messages[1]['content']
        
        # Extract persona from system message to tailor mock response
        persona_name = "Unknown"
        if "Grammar Guru" in system_message:
            persona_name = "Grammar Guru"
        elif "Content Connoisseur" in system_message:
            persona_name = "Content Connoisseur"
        elif "Structure Specialist" in system_message:
            persona_name = "Structure Specialist"
        elif "Critical Thinker" in system_message:
            persona_name = "Critical Thinker"
            
        print(f"\n--- Mock LLM Call for {persona_name} ---")
        print(f"System: {system_message[:100]}...")
        print(f"Human: {human_message[:100]}...")
        
        # Simple mock logic for scores and feedback
        score = 75
        feedback = f"As the {persona_name}, I evaluated the essay. "
        strengths = [f"Good {persona_name.lower().replace(' ', '_')} aspects"] 
        weaknesses = [f"Areas for {persona_name.lower().replace(' ', '_')} improvement"] 

        if "refine" in system_message.lower() or "peer feedback" in human_message.lower():
            score += 5 # Simulate refinement improving score
            feedback += "I've considered peer feedback and refined my assessment. "
            
        feedback += "Overall, it's a solid piece."

        mock_response = {
            "score": score,
            "feedback": feedback,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
        return {"content": json.dumps(mock_response)}

class ChatPromptTemplate:
    def __init__(self, messages: List[Dict]):
        self.messages = messages

    def format_messages(self, **kwargs) -> List[Dict]:
        formatted_messages = []
        for msg in self.messages:
            content = msg['content'].format(**kwargs)
            formatted_messages.append({"role": msg['role'], "content": content})
        return formatted_messages


# --- llm_agents.py ---

class LLMAgent:
    def __init__(self, persona_name: str, persona_description: str, llm_model_name: str = "gpt-4"):
        self.persona_name = persona_name
        self.persona_description = persona_description
        # In a real application, ensure OPENAI_API_KEY is set in your environment
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0.5)

    def _construct_prompt(self, essay_text: str, context: str = "") -> ChatPromptTemplate:
        system_message = (
            f"You are {self.persona_name}, a highly skilled {self.persona_description}. "
            "Your task is to evaluate an essay and provide constructive feedback. "
            "Output your evaluation in a JSON format with 'score' (0-100), 'feedback', 'strengths', and 'weaknesses'."
        )
        if context:
            system_message += f" Consider the following context for refining your evaluation: {context}"

        human_message = f"Please evaluate the following essay:\n\n{{essay_text}}"

        return ChatPromptTemplate(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "human", "content": human_message}
            ]
        )

    def evaluate(self, essay_text: str) -> Dict:
        print(f"\n[{self.persona_name}] Conducting initial evaluation...")
        prompt = self._construct_prompt(essay_text)
        messages = prompt.format_messages(essay_text=essay_text)
        
        try:
            llm_response = self.llm.invoke(messages)
            content = llm_response.get("content", "{}")
            evaluation = json.loads(content)
            return evaluation
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.persona_name}: {e}")
            return {"score": 0, "feedback": "JSON decoding error.", "strengths": [], "weaknesses": []}
        except Exception as e:
            print(f"Error during LLM evaluation for {self.persona_name}: {e}")
            return {"score": 0, "feedback": "LLM interaction error.", "strengths": [], "weaknesses": []}

    def refine_evaluation(self, essay_text: str, peer_feedback: Dict) -> Dict:
        print(f"\n[{self.persona_name}] Refining evaluation with peer feedback...")
        context = (
            "Consider the following peer feedback from other evaluators to refine your current assessment. "
            "Adjust your score, feedback, strengths, and weaknesses if necessary to provide a more comprehensive view. "
            f"Peer feedback: {json.dumps(peer_feedback)}"
        )
        prompt = self._construct_prompt(essay_text, context=context)
        messages = prompt.format_messages(essay_text=essay_text)

        try:
            llm_response = self.llm.invoke(messages)
            content = llm_response.get("content", "{}")
            evaluation = json.loads(content)
            return evaluation
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON during refinement for {self.persona_name}: {e}")
            return {"score": 0, "feedback": "JSON decoding error during refinement.", "strengths": [], "weaknesses": []}
        except Exception as e:
            print(f"Error during LLM refinement for {self.persona_name}: {e}")
            return {"score": 0, "feedback": "LLM interaction error during refinement.", "strengths": [], "weaknesses": []}


# --- evaluator.py ---

class EssayEvaluator:
    def __init__(self, agent_personas: List[Dict]):
        self.agents: List[LLMAgent] = []
        for persona in agent_personas:
            self.agents.append(LLMAgent(persona_name=persona["name"], persona_description=persona["description"]))

    def _conduct_individual_evaluations(self, essay_text: str) -> Dict[str, Dict]:
        print("\n--- Conducting individual evaluations ---")
        individual_evaluations = {}
        for agent in self.agents:
            evaluation = agent.evaluate(essay_text)
            individual_evaluations[agent.persona_name] = evaluation
        return individual_evaluations

    def _conduct_debate(self, essay_text: str, initial_evaluations: Dict[str, Dict], num_rounds: int = 3) -> Dict[str, Dict]:
        print(f"\n--- Starting multi-agent debate ({num_rounds} rounds) ---")
        current_evaluations = initial_evaluations

        for round_num in range(num_rounds):
            print(f"\n--- Debate Round {round_num + 1} ---")
            refined_evaluations_this_round = {}
            
            for agent_name, agent_eval in current_evaluations.items():
                # Prepare peer feedback for the current agent
                peer_feedback = {name: eval for name, eval in current_evaluations.items() if name != agent_name}
                
                # Find the actual agent object to call refine_evaluation
                current_agent = next((agent for agent in self.agents if agent.persona_name == agent_name), None)
                if current_agent:
                    refined_eval = current_agent.refine_evaluation(essay_text, peer_feedback)
                    refined_evaluations_this_round[agent_name] = refined_eval
                else:
                    print(f"Warning: Agent {agent_name} not found for refinement.")
                    refined_evaluations_this_round[agent_name] = agent_eval # Keep original if agent not found

            current_evaluations = refined_evaluations_this_round
        
        print("--- Debate concluded ---")
        return current_evaluations

    def generate_report(self, final_evaluations: Dict[str, Dict]) -> Dict:
        print("\n--- Generating comprehensive report ---")
        overall_score = 0
        all_feedback = []
        all_strengths = set()
        all_weaknesses = set()

        for agent_name, evaluation in final_evaluations.items():
            score = evaluation.get("score", 0)
            feedback = evaluation.get("feedback", "No feedback provided.")
            strengths = evaluation.get("strengths", [])
            weaknesses = evaluation.get("weaknesses", [])
            
            overall_score += score
            all_feedback.append(f"[{agent_name}]: {feedback}")
            all_strengths.update(strengths)
            all_weaknesses.update(weaknesses)

        num_agents = len(final_evaluations)
        if num_agents > 0:
            overall_score = round(overall_score / num_agents, 2)
        else:
            overall_score = 0

        report = {
            "overall_grade": overall_score,
            "summary_feedback": "\n".join(all_feedback),
            "overall_strengths": list(all_strengths),
            "overall_weaknesses": list(all_weaknesses),
            "detailed_agent_evaluations": final_evaluations
        }
        return report

    def evaluate_essay(self, essay_text: str) -> Dict:
        individual_evals = self._conduct_individual_evaluations(essay_text)
        final_evals = self._conduct_debate(essay_text, individual_evals)
        report = self.generate_report(final_evals)
        return report


# --- main.py ---

if __name__ == "__main__":
    # Define agent personas
    agent_personas = [
        {"name": "Grammar Guru", "description": "an expert in grammar, spelling, and punctuation. Focuses on linguistic correctness."
        },
        {"name": "Content Connoisseur", "description": "a master of subject matter, logic, and argumentation. Evaluates depth and relevance."
        },
        {"name": "Structure Specialist", "description": "an architect of coherence and organization. Assesses essay flow, paragraphing, and overall structure."
        },
        {"name": "Critical Thinker", "description": "a sharp analyst of reasoning, originality, and insight. Examines the quality of thought and critical engagement."
        }
    ]

    # Instantiate the EssayEvaluator
    evaluator = EssayEvaluator(agent_personas)

    # Sample essay text
    sample_essay = """
    The Role of Artificial Intelligence in Modern Education

    Artificial intelligence (AI) is rapidly transforming various sectors, and education is no exception. This essay will explore the potential benefits and challenges of integrating AI into modern educational practices. By automating repetitive tasks, AI can free up educators to focus more on personalized student interaction. Furthermore, AI-powered tools can provide adaptive learning experiences, tailoring content and pace to individual student needs, which can significantly enhance learning outcomes.

    One of the primary advantages of AI in education is its ability to personalize learning. AI algorithms can analyze student performance data, identify areas of weakness, and recommend tailored resources or exercises. This level of customization is difficult to achieve in traditional classroom settings, where a single curriculum must cater to diverse learning styles and abilities. Moreover, AI can offer immediate feedback, a crucial element for effective learning, allowing students to correct mistakes in real-time and reinforce their understanding.

    However, the integration of AI also presents challenges. Concerns about data privacy and the ethical implications of AI algorithms are paramount. Who owns the data collected from students, and how will it be protected? There is also the risk of over-reliance on technology, potentially diminishing human interaction and the development of critical social skills. Furthermore, the initial cost of implementing AI infrastructure can be prohibitive for many educational institutions, creating a digital divide.

    In conclusion, while AI holds immense promise for revolutionizing education by offering personalized learning and efficient administrative support, it is imperative to address the ethical, privacy, and economic challenges. A balanced approach that leverages AI's strengths while mitigating its weaknesses will be key to successfully integrating this technology into the classrooms of the future.
    """

    print("\n--- EduCritique: Multi-Perspective Essay Grader ---")
    print("Evaluating sample essay...")
    
    # Evaluate the essay
    final_report = evaluator.evaluate_essay(sample_essay)

    # Print the comprehensive report
    print("\n======================================")
    print("\nFinal EduCritique Evaluation Report:")
    print("======================================")
    print(f"Overall Grade: {final_report['overall_grade']}/100")
    print("\n--- Summary Feedback ---")
    print(final_report['summary_feedback'])
    print("\n--- Overall Strengths ---")
    for strength in final_report['overall_strengths']:
        print(f"- {strength}")
    print("\n--- Overall Weaknesses ---")
    for weakness in final_report['overall_weaknesses']:
        print(f"- {weakness}")
    print("\n--- Detailed Agent Evaluations ---")
    for agent_name, eval_details in final_report['detailed_agent_evaluations'].items():
        print(f"\nAgent: {agent_name}")
        print(f"  Score: {eval_details.get('score', 'N/A')}")
        print(f"  Feedback: {eval_details.get('feedback', 'N/A')}")
        print(f"  Strengths: {', '.join(eval_details.get('strengths', []))}")
        print(f"  Weaknesses: {', '.join(eval_details.get('weaknesses', []))}")
