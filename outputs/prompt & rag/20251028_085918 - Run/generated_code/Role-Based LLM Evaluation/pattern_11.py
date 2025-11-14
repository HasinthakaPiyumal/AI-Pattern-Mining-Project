class LLMAgent:
    """Represents an LLM agent with a specific evaluative persona."""
    def __init__(self, name: str, persona_prompt: str):
        self.name = name
        self.persona_prompt = persona_prompt

    def generate_feedback(self, essay_text: str) -> str:
        """Mocks the LLM's feedback generation based on its persona.

        In a real application, this would involve an API call to an LLM
        with the persona_prompt and essay_text.
        """
        print(f"\n--- {self.name} (simulating LLM processing) ---")
        # Mocking feedback based on persona
        if "Grammar and Style Editor" in self.name:
            return f"As a {self.name}, I've reviewed your essay for linguistic correctness and stylistic choices. Pay attention to sentence structure variety and ensure consistent tense usage. For example, in the phrase '{essay_text[:50]}...', consider rephrasing for conciseness."
        elif "Content and Argumentation Analyst" in self.name:
            return f"From a {self.name}'s perspective, your essay presents an interesting argument. However, strengthen your thesis statement and ensure each paragraph clearly supports it with specific evidence. The section about '{essay_text[50:100]}...' could benefit from more robust examples."
        elif "Clarity and Coherence Reviewer" in self.name:
            return f"As a {self.name}, I focused on the readability and flow. Your essay is generally clear, but some transitions between paragraphs, particularly leading into the discussion of '{essay_text[100:150]}...', could be smoother to enhance overall coherence."
        elif "Creativity and Originality Judge" in self.name:
            return f"The {self.name} finds some intriguing ideas in your work. To elevate originality, try to introduce a more unique perspective or an unexpected connection, especially around the theme of '{essay_text[150:200]}...' rather than relying on commonly discussed points."
        else:
            return f"As a generic reviewer, I found your essay interesting. Specific feedback would require a defined persona."


class EssayCritic:
    """Orchestrates the multi-perspective LLM evaluation of an essay."""
    def __init__(self, agents: list[LLMAgent]):
        self.agents = agents

    def evaluate_essay(self, essay_text: str) -> dict:
        """Sends the essay to each agent and collects their feedback."""
        all_feedback = {}
        print("\nStarting multi-perspective essay evaluation...")
        for agent in self.agents:
            feedback = agent.generate_feedback(essay_text)
            all_feedback[agent.name] = feedback
        print("\nEvaluation complete.")
        return all_feedback

    def present_feedback(self, feedback: dict):
        """Presents the aggregated feedback in a readable format."""
        print("\n===== Aggregated Essay Feedback =====")
        for agent_name, agent_feedback in feedback.items():
            print(f"\n--- Feedback from {agent_name} ---")
            print(agent_feedback)
        print("\n=====================================")


def main():
    """Main function to run the AI Essay Critic application."""
    print("Welcome to the AI Essay Critic!")
    print("Please paste your essay below. Type 'END' on a new line when you are done.\n")

    essay_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        essay_lines.append(line)
    
    essay_text = "\n".join(essay_lines).strip()

    if not essay_text:
        print("No essay provided. Exiting.")
        return

    # 1. Initialize LLM Agents with specific personas
    grammar_editor = LLMAgent(
        name="Grammar and Style Editor",
        persona_prompt="You are an expert grammar and style editor. Your task is to review essays for linguistic correctness, clarity, sentence structure, vocabulary usage, and overall stylistic elegance."
    )
    content_analyst = LLMAgent(
        name="Content and Argumentation Analyst",
        persona_prompt="You are a critical content and argumentation analyst. Your task is to evaluate the strength of the thesis, the logical coherence of arguments, the relevance and depth of evidence, and the overall persuasive power of the essay."
    )
    clarity_reviewer = LLMAgent(
        name="Clarity and Coherence Reviewer",
        persona_prompt="You are a clarity and coherence reviewer. Your task is to assess the essay's readability, logical flow between ideas and paragraphs, structural organization, and how easily a reader can follow the main points."
    )
    creativity_judge = LLMAgent(
        name="Creativity and Originality Judge",
        persona_prompt="You are a creativity and originality judge. Your task is to evaluate the uniqueness of ideas, innovative perspectives, imaginative language, and the overall original contribution of the essay to its topic."
    )

    agents = [grammar_editor, content_analyst, clarity_reviewer, creativity_judge]

    # 3. Initialize the Essay Critic with the agents
    critic = EssayCritic(agents)

    # 4. Perform Multi-Perspective Evaluation
    feedback = critic.evaluate_essay(essay_text)

    # 5. Present the feedback
    critic.present_feedback(feedback)

if __name__ == "__main__":
    main()