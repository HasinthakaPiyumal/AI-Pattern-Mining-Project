from typing import List, Dict

class MockTokenizer:
    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([chr(t) for t in tokens])

class MockLLM:
    def generate(self, prompt: str) -> str:
        if "summary in Spanish:" in prompt:
            return "Este es un resumen del contenido dado en español.\n" + prompt.split("Source (en): ")[-1].split("\nTarget (es): ")[0].replace("The sun is a star.", "El sol es una estrella.").replace("Water is H2O.", "El agua es H2O.")
        elif "quiz in French:" in prompt:
            return "Voici un quiz sur le contenu donné en français.\nQuestion: Quelle est la capitale de la France?\nRéponse: Paris."
        elif "explanation in German:" in prompt:
            return "Dies ist eine Erklärung des gegebenen Inhalts auf Deutsch.\nKonzept: Relativitätstheorie"
        else:
            return f"Generated content based on the prompt: {prompt}"

class MultilingualEducationalContentGenerator:
    def __init__(self, llm_model, tokenizer):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self._example_store = {
            ("en", "es", "summary"):
                [
                    {"source": "The sun is a star.", "target": "El sol es una estrella."},
                    {"source": "Water is H2O.", "target": "El agua es H2O."}
                ],
            ("en", "fr", "quiz"):
                [
                    {"source": "Question: What is the capital of France?", "target": "Question: Quelle est la capitale de la France?"},
                    {"source": "Answer: Paris.", "target": "Réponse: Paris."}
                ],
            ("en", "de", "explanation"):
                [
                    {"source": "Concept: Theory of Relativity", "target": "Konzept: Relativitätstheorie"},
                    {"source": "Explanation: E=mc^2", "target": "Erklärung: E=mc^2"}
                ],
        }

    def _prepare_in_context_examples(self, source_language: str, target_language: str, content_type: str) -> List[Dict[str, str]]:
        key = (source_language, target_language, content_type)
        return self._example_store.get(key, [])

    def generate_educational_content(self, source_content: str, source_language: str, target_language: str, content_type: str = "summary") -> str:
        in_context_examples = self._prepare_in_context_examples(source_language, target_language, content_type)

        prompt_parts = []
        prompt_parts.append(f"Given the following examples where source content is a {content_type} in the source language and its equivalent in the target language:\n\n")

        for example in in_context_examples:
            prompt_parts.append(f"Source ({source_language}): {example['source']}\nTarget ({target_language}): {example['target']}\n\n")

        prompt_parts.append(f"Now, for the following source content, generate a {content_type} in {target_language}:\n")
        prompt_parts.append(f"Source ({source_language}): {source_content}\nTarget ({target_language}): ")

        full_prompt = "".join(prompt_parts)
        generated_text = self.llm_model.generate(full_prompt)
        return generated_text

if __name__ == "__main__":
    mock_llm = MockLLM()
    mock_tokenizer = MockTokenizer()

    generator = MultilingualEducationalContentGenerator(mock_llm, mock_tokenizer)

    # Example 1: Generate a summary in Spanish
    english_article = "The Earth is the third planet from the Sun and the only astronomical object known to harbor life. It is the densest planet in the Solar System and the largest of the four terrestrial planets."
    spanish_summary = generator.generate_educational_content(
        source_content=english_article,
        source_language="en",
        target_language="es",
        content_type="summary"
    )
    print("Generated Spanish Summary:")
    print(spanish_summary)
    print("\n" + "-" * 30 + "\n")

    # Example 2: Generate a quiz in French
    french_quiz_content = "World War II was a global war that lasted from 1939 to 1945."
    french_quiz = generator.generate_educational_content(
        source_content=french_quiz_content,
        source_language="en",
        target_language="fr",
        content_type="quiz"
    )
    print("Generated French Quiz:")
    print(french_quiz)
    print("\n" + "-" * 30 + "\n")

    # Example 3: Generate an explanation in German
    german_explanation_content = "Gravity is a fundamental interaction which causes mutual attraction between all things that have mass or energy."
    german_explanation = generator.generate_educational_content(
        source_content=german_explanation_content,
        source_language="en",
        target_language="de",
        content_type="explanation"
    )
    print("Generated German Explanation:")
    print(german_explanation)
    print("\n" + "-" * 30 + "\n")

    # Example 4: Content type with no specific examples (will use generic LLM response)
    korean_summary_content = "Artificial intelligence is intelligence demonstrated by machines."
    korean_summary = generator.generate_educational_content(
        source_content=korean_summary_content,
        source_language="en",
        target_language="ko",
        content_type="summary"
    )
    print("Generated Korean Summary (without specific examples in store):")
    print(korean_summary)
    print("\n" + "-" * 30 + "\n")