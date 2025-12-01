from typing import List, Dict, Any
import time

from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class MockLLM(BaseLLM):
    """A mock LLM for demonstration purposes."""
    def _call(self, prompt: str, stop: List[str] = None, **kwargs: Any) -> str:
        # Simulate LLM processing time
        time.sleep(0.1)
        if "critique:" in prompt.lower() or "evaluate:" in prompt.lower():
            if "irrelevant" in prompt.lower() or "not comprehensive" in prompt.lower():
                return "CRITIQUE: The retrieved documents seem partially irrelevant, and the initial answer lacks detail. Suggest re-retrieval for more specific information and then refine the response. CONFIDENCE: LOW"
            elif "excellent" in prompt.lower() or "perfect" in prompt.lower():
                return "CRITIQUE: Retrieved documents are highly relevant and the initial answer is comprehensive and accurate. No further action needed. CONFIDENCE: HIGH"
            else:
                return "CRITIQUE: The retrieved documents are generally relevant, but the answer could be more concise. Suggest refinement. CONFIDENCE: MEDIUM"
        elif "refine:" in prompt.lower():
            return f"REFINED ANSWER based on review: This is an improved version of the answer based on the critique. Original intent: {prompt.split('Original Answer:')[-1].strip()}"
        else:
            # Basic response generation
            if "price" in prompt.lower():
                return "The price for the product is generally in the range of $100-$500, depending on the model and features. Please specify the exact product you are interested in for a precise quote."
            elif "shipping" in prompt.lower():
                return "Shipping usually takes 3-5 business days for standard delivery. Expedited options are available at checkout for an additional fee."
            elif "return policy" in prompt.lower():
                return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition with all packaging."
            elif "features of product A" in prompt.lower():
                return "Product A features a high-resolution display, long-lasting battery, and an advanced camera system. It's ideal for professional use."
            else:
                return "I'm sorry, I couldn't find a direct answer to that specific question. Could you please rephrase or provide more details?"

    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _stream(self, prompt: str, stop: List[str] = None, **kwargs: Any) -> Any:
        yield GenerationChunk(text=self._call(prompt, stop, **kwargs))

class SelfRAGChatbot:
    def __init__(self, knowledge_base_data: List[Dict[str, str]]):
        self.llm = MockLLM()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        self.collection_name = "ecommerce_knowledge_base"
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        try:
            self.vector_store = self.chroma_client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
        except:
            self.vector_store = self.chroma_client.create_collection(name=self.collection_name, embedding_function=self.embedding_function)
            self._index_knowledge_base(knowledge_base_data)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        self.initial_generation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an e-commerce customer support assistant. Answer the user's question concisely and accurately based on the provided context. If the context does not contain the answer, state that you don't have enough information."),
                ("human", "Context: {context}\nQuestion: {question}"),
            ]
        )

        self.critique_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a self-reflection agent for a RAG system. Evaluate the relevance of the retrieved documents and the quality of the generated answer for the given question. State if the documents are insufficient or irrelevant, or if the answer needs refinement or re-retrieval. Provide a confidence score (HIGH, MEDIUM, LOW). Format your response as: CRITIQUE: [Your assessment]. CONFIDENCE: [Confidence Score]."),
                ("human", "Original Question: {question}\nRetrieved Documents: {documents}\nInitial Answer: {answer}"),
            ]
        )

        self.refinement_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an e-commerce customer support assistant. Refine the following answer based on the critique and any new information if available. Aim for clarity, accuracy, and completeness."),
                ("human", "Original Question: {question}\nCritique: {critique}\nOriginal Answer: {answer}\nNew Context (if any): {new_context}"),
            ]
        )

    def _index_knowledge_base(self, data: List[Dict[str, str]]):
        documents = []
        metadatas = []
        ids = []
        for i, item in enumerate(data):
            documents.append(item["content"])
            metadatas.append({"source": item.get("source", "knowledge_base"), "id": str(i)})
            ids.append(f"doc_{i}")
        self.vector_store.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Indexed {len(documents)} documents into ChromaDB.")

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    def _should_re_retrieve(self, critique_result: str) -> bool:
        return "re-retrieval" in critique_result.lower() or "irrelevant" in critique_result.lower()

    def _process_query(self, query: str) -> Dict[str, Any]:
        print(f"\nUser Query: {query}")
        # 1. Initial Retrieval
        retrieved_docs = self.retriever.invoke(query)
        formatted_docs = self._format_docs(retrieved_docs)
        print(f"\n--- Retrieved Documents (Initial) ---\n{formatted_docs[:200]}...")

        # 2. Initial Generation
        initial_answer_chain = self.initial_generation_prompt | self.llm | StrOutputParser()
        initial_answer = initial_answer_chain.invoke({"context": formatted_docs, "question": query})
        print(f"\n--- Initial Answer ---\n{initial_answer}")

        # 3. Self-Critique
        critique_chain = self.critique_prompt | self.llm | StrOutputParser()
        critique_result = critique_chain.invoke({"question": query, "documents": formatted_docs, "answer": initial_answer})
        print(f"\n--- Self-Critique ---\n{critique_result}")

        final_answer = initial_answer
        action_taken = "initial_generation"
        confidence = critique_result.split("CONFIDENCE:")[-1].strip()

        # 4. Adaptive Retrieval/Refinement Loop
        if self._should_re_retrieve(critique_result):
            print("\n--- Critique suggests re-retrieval. Attempting deeper search... ---")
            # Simulate re-retrieval (e.g., with different search parameters or a more targeted query)
            # For this mock, we'll just pretend to get 'better' docs or more docs
            retrieved_docs_2 = self.retriever.invoke(f"more details on {query}")
            formatted_docs_2 = self._format_docs(retrieved_docs_2)
            print(f"\n--- Retrieved Documents (Re-retrieved) ---\n{formatted_docs_2[:200]}...")

            # Re-generate with new docs
            re_generated_answer_chain = self.initial_generation_prompt | self.llm | StrOutputParser()
            re_generated_answer = re_generated_answer_chain.invoke({"context": formatted_docs_2, "question": query})
            final_answer = re_generated_answer
            action_taken = "re_retrieval_and_regeneration"
            print(f"\n--- Re-generated Answer ---\n{final_answer}")

            # Re-critique the re-generated answer (optional, but good for robust SelfRAG)
            re_critique_result = critique_chain.invoke({"question": query, "documents": formatted_docs_2, "answer": re_generated_answer})
            print(f"\n--- Re-Critique ---\n{re_critique_result}")
            confidence = re_critique_result.split("CONFIDENCE:")[-1].strip()

        elif "refine" in critique_result.lower() and confidence != "HIGH":
            print("\n--- Critique suggests refinement. Refining answer... ---")
            refinement_chain = self.refinement_prompt | self.llm | StrOutputParser()
            refined_answer = refinement_chain.invoke({"question": query, "critique": critique_result, "answer": initial_answer, "new_context": formatted_docs})
            final_answer = refined_answer
            action_taken = "refinement"
            print(f"\n--- Refined Answer ---\n{final_answer}")

        return {"final_answer": final_answer, "confidence": confidence, "action_taken": action_taken}

# Mock E-commerce Knowledge Base Data
KNOWLEDGE_BASE_DATA = [
    {"content": "The LuxeSmart Watch features a vibrant AMOLED display, 7-day battery life, heart rate monitoring, and GPS tracking. It's water-resistant up to 50 meters. Price: $299.", "source": "product_page_LSW001"},
    {"content": "Our shipping policy offers standard shipping (3-5 business days) for $5, and express shipping (1-2 business days) for $15. Free standard shipping on orders over $100.", "source": "shipping_policy"},
    {"content": "Returns are accepted within 30 days of purchase for unused items in original packaging. Refunds are processed within 7-10 business days. Custom items are final sale.", "source": "return_policy"},
    {"content": "The NeoSound Earbuds boast active noise cancellation, 24-hour battery with charging case, and touch controls. Connects via Bluetooth 5.2. Price: $149.", "source": "product_page_NSE002"},
    {"content": "To reset your account password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your registered email address.", "source": "faq_account_access"},
    {"content": "The UltraGamer Laptop comes with an RTX 4080 GPU, Intel i9 processor, 32GB RAM, and a 144Hz QHD display. Ideal for high-performance gaming. Price: $2499.", "source": "product_page_UGL003"},
    {"content": "Orders are typically processed within 24 hours. You will receive a tracking number via email once your order has shipped.", "source": "order_processing"},
    {"content": "We offer a 1-year limited warranty on all electronics, covering manufacturing defects. Accidental damage is not covered.", "source": "warranty_policy"},
    {"content": "For technical support, please visit our support portal or call our hotline at 1-800-555-TECH. Available M-F, 9 AM - 5 PM EST.", "source": "customer_support"},
    {"content": "The price of the LuxeSmart Watch is $299. It includes a charger and two strap options.", "source": "product_page_LSW001_price"},
    {"content": "Our current promotion offers 15% off all NeoSound Earbuds for a limited time.", "source": "promotions"},
]

if __name__ == "__main__":
    chatbot = SelfRAGChatbot(KNOWLEDGE_BASE_DATA)

    queries = [
        "What is the price of the LuxeSmart Watch?",
        "How long does shipping usually take?",
        "What is your return policy for an item I bought last week?",
        "Tell me about the features of the NeoSound Earbuds.",
        "I forgot my password, how can I reset it?",
        "What are the key specifications of the UltraGamer Laptop?",
        "Do you have any current discounts on earbuds?",
        "What is your warranty policy?",
        "How do I contact technical support?",
        "I need information about Product X. Can you help?"
    ]

    for i, query in enumerate(queries):
        print(f"\n============================== QUERY {i+1} ==============================")
        result = chatbot._process_query(query)
        print(f"\nFinal Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Action Taken: {result['action_taken']}")
        print("======================================================================")
        time.sleep(1) # Pause for readability
