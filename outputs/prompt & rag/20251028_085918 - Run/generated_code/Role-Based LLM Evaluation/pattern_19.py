from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict

class LLMAgent:
    def __init__(self, name: str, role_description: str, model_name: str = "gpt-4"):
        self.name = name
        self.role_description = role_description
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7) # Consider using environment variables for API keys
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are a {self.role_description}. Your task is to evaluate a news article based on your specific expertise."),
            ("human", "Evaluate the following news article:\n\n{article_content}\n\nProvide a concise evaluation focusing on your perspective, including a score (1-10) and a brief justification.")
        ])

    def evaluate(self, article_content: str) -> Dict[str, str]:
        chain = self.prompt_template | self.llm
        response = chain.invoke({"article_content": article_content})
        return {"agent": self.name, "evaluation": response.content}

def create_agents(model_name: str = "gpt-4") -> List[LLMAgent]:
    agents = [
        LLMAgent("Fact-Checker", "meticulous fact-checker, identifying factual inaccuracies, logical fallacies, and unsubstantiated claims.", model_name),
        LLMAgent("Bias Analyst", "unbiased analyst detecting political, ideological, or commercial biases, identifying loaded language or framing.", model_name),
        LLMAgent("Journalistic Ethics LLM", "guardian of journalistic principles, assessing adherence to ethical standards like fairness, objectivity, and source transparency.", model_name),
        LLMAgent("Readability/Clarity LLM", "expert in clear communication, evaluating the article's readability, jargon use, and overall clarity for a general audience.", model_name),
        LLMAgent("Skeptic LLM", "critical thinker who questions underlying assumptions, looks for sensationalism, and assesses the strength of evidence provided.", model_name),
    ]
    return agents
