from ner_module import NERModule
from frequency_analyzer import FrequencyAnalyzer

class MockLLM:
    """A mock LLM to simulate response generation."""
    def generate_response(self, query: str, context: str = None) -> str:
        if context:
            return f"[LLM with Retrieval Context] Based on the context provided ('{context}') and my knowledge, here's the answer to: '{query}'."
        else:
            return f"[LLM-only] Based on my internal knowledge, here's the answer to: '{query}'."

class MockRetrievalModule:
    """A mock retrieval module to simulate fetching documents."""
    def retrieve_documents(self, query: str) -> str:
        # In a real system, this would query a vector database or search engine
        if "CRISPR" in query or "sickle cell anemia" in query:
            return "Recent studies show gene editing via CRISPR holds promise for sickle cell anemia by correcting genetic mutations."
        elif "Alzheimer's" in query:
            return "New research explores amyloid-beta plaque reduction and tau protein aggregation as key therapeutic targets for Alzheimer's disease."
        else:
            return f"Retrieved general medical information relevant to: '{query}'."

class MedQueryAssistant:
    def __init__(
        self,
        ner_module: NERModule,
        frequency_analyzer: FrequencyAnalyzer,
        llm: MockLLM,
        retrieval_module: MockRetrievalModule,
        retrieval_threshold: float = 0.5
    ):
        self.ner_module = ner_module
        self.frequency_analyzer = frequency_analyzer
        self.llm = llm
        self.retrieval_module = retrieval_module
        self.retrieval_threshold = retrieval_threshold
        print(f"MedQueryAssistant initialized with retrieval threshold: {self.retrieval_threshold}")

    def answer_query(self, query: str) -> str:
        print(f"\nProcessing Query: '{query}'")
        # 1. Entity Extraction
        entities = self.ner_module.extract_entities(query)
        print(f"  Extracted Entities: {entities}")

        # 2. Query Complexity Scoring
        complexity_score = self.frequency_analyzer.calculate_query_complexity_score(entities)
        print(f"  Query Complexity Score: {complexity_score:.2f}")

        # 3. Dynamic Retrieval Decision
        context = None
        if complexity_score >= self.retrieval_threshold:
            print("  Decision: Complexity score is high, activating retrieval module.")
            context = self.retrieval_module.retrieve_documents(query)
            print(f"  Retrieved Context: '{context}'")
        else:
            print("  Decision: Complexity score is low, relying on LLM internal knowledge (no retrieval).")
        
        # 4. LLM Response Generation
        response = self.llm.generate_response(query, context)
        return response

if __name__ == "__main__":
    # Initialize components
    ner = NERModule(model_name="en_core_web_sm") # Use a specialized medical model in production
    analyzer = FrequencyAnalyzer()
    llm = MockLLM()
    retrieval = MockRetrievalModule()

    # Initialize the MedQuery Assistant with a threshold
    assistant = MedQueryAssistant(
        ner_module=ner,
        frequency_analyzer=analyzer,
        llm=llm,
        retrieval_module=retrieval,
        retrieval_threshold=0.5 # Example threshold
    )

    # Test Queries
    query1 = "What is the recommended dosage for paracetamol for adults?" # Expected low complexity
    print(f"Assistant Response: {assistant.answer_query(query1)}")

    query2 = "Discuss the latest advancements in gene therapy for Duchenne muscular dystrophy." # Expected high complexity
    print(f"Assistant Response: {assistant.answer_query(query2)}")

    query3 = "What are the side effects of ibuprofen?" # Expected low complexity
    print(f"Assistant Response: {assistant.answer_query(query3)}")

    query4 = "Explain the role of amyloid plaques in Alzheimer's disease progression." # Expected high complexity
    print(f"Assistant Response: {assistant.answer_query(query4)}")

    query5 = "What is the capital of France?" # A general knowledge query to see behavior with no medical entities, defaults to low complexity
    print(f"Assistant Response: {assistant.answer_query(query5)}")