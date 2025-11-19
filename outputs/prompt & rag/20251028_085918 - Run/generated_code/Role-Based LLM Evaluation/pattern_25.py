# main.py

import json

# --- Mock LLM Class (Replace with actual LLM integration like OpenAI, LangChain, etc.) ---
class MockLLM:
    def __init__(self, model_name="mock-gpt-4"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Simulates an LLM response based on the prompt.
        In a real application, this would interact with an actual LLM API.
        """
        print(f"\n--- Mock LLM Generating for {self.model_name} ---")
        print(f"Prompt: {prompt[:200]}...") # Print first 200 chars of prompt

        # Simple keyword-based simulation for demonstration
        if "Skeptical Customer" in prompt:
            return json.dumps({
                "persona": "Skeptical Customer",
                "evaluation": "This product description sounds too good to be true. It lacks specific details on durability and potential drawbacks. Is the 'premium quality' truly reflected in the materials? I'd worry about hidden costs or a short lifespan.",
                "score": 3/5 # Example score
            })
        elif "Marketing Expert" in prompt:
            return json.dumps({
                "persona": "Marketing Expert",
                "evaluation": "The headline is catchy and the benefits are highlighted well. However, it could benefit from stronger calls to action and more emotional language to resonate with the target audience. SEO keywords seem present but could be more naturally integrated.",
                "score": 4/5
            })
        elif "Technical Reviewer" in prompt:
            return json.dumps({
                "persona": "Technical Reviewer",
                "evaluation": "Specifications are somewhat vague. 'High-performance' needs concrete metrics. What are the dimensions, power consumption, and exact materials? The description is clear but lacks the precise technical data an informed buyer would seek.",
                "score": 3.5/5
            })
        else:
            return json.dumps({
                "persona": "General Evaluator",
                "evaluation": "Could not determine specific persona. General feedback: The description is well-written but could be more engaging.",
                "score": 3/5
            })

# --- Persona Definitions ---
PERSONAS = {
    "Skeptical Customer": {
        "role": (
            "You are a highly critical and skeptical customer who is always looking for flaws, hidden costs, "
            "and potential misrepresentations in product descriptions. Your goal is to identify reasons not to buy, "
            "question claims, and assess the true value for money. Focus on durability, honesty, and potential buyer's remorse."
        ),
        "output_format_instruction": (
            "Your evaluation MUST be a JSON object with the following keys: "
            "'persona' (string), 'evaluation' (string describing your assessment), 'score' (float from 1.0 to 5.0)."
        )
    },
    "Marketing Expert": {
        "role": (
            "You are a seasoned marketing expert with extensive experience in e-commerce copywriting and SEO. "
            "Evaluate the product description for its persuasiveness, alignment with brand messaging, target audience appeal, "
            "clarity of benefits, and optimization for search engines. Look for strong calls to action and emotional resonance."
        ),
        "output_format_instruction": (
            "Your evaluation MUST be a JSON object with the following keys: "
            "'persona' (string), 'evaluation' (string describing your assessment), 'score' (float from 1.0 to 5.0)."
        )
    },
    "Technical Reviewer": {
        "role": (
            "You are a meticulous technical reviewer with a strong engineering background. "
            "Assess the product description for accuracy of specifications, completeness of technical details, "
            "clarity of features, and any potential ambiguities. Look for precise measurements, materials, performance metrics, "
            "and ensure all claims are technically plausible and well-supported."
        ),
        "output_format_instruction": (
            "Your evaluation MUST be a JSON object with the following keys: "
            "'persona' (string), 'evaluation' (string describing your assessment), 'score' (float from 1.0 to 5.0)."
        )
    },
}

# --- PersonaAgent Class ---
class PersonaAgent:
    def __init__(self, name: str, persona_definition: dict, llm: MockLLM):
        self.name = name
        self.persona_definition = persona_definition
        self.llm = llm
        self._construct_system_prompt()

    def _construct_system_prompt(self):
        self.system_prompt = (
            f"As a {self.name}, {self.persona_definition['role']}\n\n"
            f"Your output must follow this format: {self.persona_definition['output_format_instruction']}"
        )

    def evaluate(self, product_description: str) -> dict:
        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"Please evaluate the following product description:\n"
            f"---\n{product_description}\n---"
        )
        try:
            raw_response = self.llm.generate(full_prompt)
            return json.loads(raw_response) # Assuming LLM returns valid JSON
        except json.JSONDecodeError:
            print(f"Warning: {self.name} received invalid JSON from LLM: {raw_response}")
            return {
                "persona": self.name,
                "evaluation": f"Error: Invalid JSON response from LLM. Raw: {raw_response}",
                "score": 1.0
            }
        except Exception as e:
            print(f"Error during {self.name} evaluation: {e}")
            return {
                "persona": self.name,
                "evaluation": f"Error during evaluation: {e}",
                "score": 1.0
            }

# --- ProductDescriptionEvaluator Class ---
class ProductDescriptionEvaluator:
    def __init__(self, product_description: str, llm: MockLLM):
        self.product_description = product_description
        self.llm = llm
        self.agents = self._create_agents()

    def _create_agents(self) -> list[PersonaAgent]:
        agents = []
        for name, definition in PERSONAS.items():
            agents.append(PersonaAgent(name, definition, self.llm))
        return agents

    def run_evaluation(self) -> dict:
        individual_evaluations = []
        for agent in self.agents:
            print(f"\nRunning evaluation for: {agent.name}")
            evaluation = agent.evaluate(self.product_description)
            individual_evaluations.append(evaluation)
        
        synthesized_report = self._synthesize_results(individual_evaluations)
        
        return {
            "product_description": self.product_description,
            "individual_evaluations": individual_evaluations,
            "synthesized_report": synthesized_report
        }

    def _synthesize_results(self, evaluations: list[dict]) -> dict:
        total_score = 0.0
        feedback_summary = []
        
        for eval_data in evaluations:
            total_score += eval_data.get("score", 0.0)
            feedback_summary.append(f"- {eval_data['persona']}: {eval_data['evaluation']}")
            
        average_score = total_score / len(evaluations) if evaluations else 0.0
        
        overall_summary = (
            f"Overall Consensus:\n"
            f"The product description received an average score of {average_score:.2f} out of 5 from diverse perspectives.\n"
            f"Key insights from each persona are as follows:\n"
            f"{chr(10).join(feedback_summary)}\n\n"
            f"Actionable Recommendations: \n"
            f"Based on the collective feedback, consider adding more specific technical details (e.g., dimensions, materials). "
            f"Address potential skeptical customer concerns by elaborating on durability and warranty. "
            f"Enhance marketing appeal with stronger calls to action and perhaps user testimonials."
        )
        
        return {
            "average_overall_score": average_score,
            "detailed_feedback_summary": overall_summary
        }

# --- Main Execution --- 
if __name__ == "__main__":
    example_product_description = """
    **ZenithFlow Pro Wireless Earbuds**

    Experience unparalleled audio freedom with ZenithFlow Pro. These high-performance wireless earbuds deliver crystal-clear sound and deep bass, perfect for music lovers and professionals alike. Featuring advanced noise-cancellation technology, you can immerse yourself in your audio without distractions. The ergonomic design ensures a comfortable fit for hours, and the intuitive touch controls provide seamless management of your music and calls. With a long-lasting battery life of up to 30 hours (with charging case), your soundtrack never stops. Compatible with all Bluetooth-enabled devices. Get yours today and elevate your listening experience!
    """

    # Initialize the mock LLM
    mock_llm_instance = MockLLM()

    # Initialize the evaluator with the product description and LLM
    evaluator = ProductDescriptionEvaluator(example_product_description, mock_llm_instance)

    # Run the evaluation
    results = evaluator.run_evaluation()

    # Print the results
    print("\n=======================================================")
    print("           PRODUCT DESCRIPTION EVALUATION REPORT         ")
    print("=======================================================")
    print("\nProduct Description:")
    print(example_product_description)
    print("\n-------------------------------------------------------")
    print("Individual Persona Evaluations:")
    for eval_data in results["individual_evaluations"]:
        print(f"\n  Persona: {eval_data['persona']}")
        print(f"  Score: {eval_data['score']}/5")
        print(f"  Feedback: {eval_data['evaluation']}")
    
    print("\n-------------------------------------------------------")
    print("Synthesized Report:")
    print(f"  Average Overall Score: {results['synthesized_report']['average_overall_score']:.2f}/5")
    print("  Detailed Summary:\n")
    print(results["synthesized_report"]["detailed_feedback_summary"])
    print("\n=======================================================")

