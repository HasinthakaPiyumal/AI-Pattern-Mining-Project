import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uuid

# --- Configuration --- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARIZATION_MODEL_NAME = "t5-small"
MAX_LLM_CONTEXT_TOKENS = 512  # Example max context window for a small LLM

# --- Placeholder for LLM Integration --- #
# In a real application, you would replace this with an actual LLM client (e.g., OpenAI, Hugging Face TGI)
class LLMClient:
    def __init__(self, api_key=None, endpoint=None):
        # Initialize your LLM client here
        self.api_key = api_key
        self.endpoint = endpoint

    def generate_response(self, prompt: str) -> str:
        # Simulate an LLM response
        print(f"\n--- LLM Input (first 200 chars): ---\n{prompt[:200]}...\n--------------------------")
        if "order status" in prompt.lower():
            return "I can help with your order status. Please provide your order number."
        elif "return policy" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging."
        elif "shipping" in prompt.lower():
            return "Standard shipping usually takes 3-5 business days. Expedited options are available at checkout."
        else:
            return "Thank you for contacting us. How can I assist you further?"

# --- Main Orchestrator Class --- #
class ConversationOrchestrator:
    def __init__(self, llm_api_key=None, llm_endpoint=None):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client()
        self.conversation_history_collection = self.chroma_client.get_or_create_collection(name="customer_conversations")
        self.customer_profiles_collection = self.chroma_client.get_or_create_collection(name="customer_profiles")

        self.summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_NAME)
        self.llm_client = LLMClient(api_key=llm_api_key, endpoint=llm_endpoint)

    def _get_embedding(self, text: str):
        return self.embedding_model.encode(text).tolist()

    def _retrieve_context(self, query: str, customer_id: str, k_history=5, k_profiles=3) -> (list, list):
        query_embedding = self._get_embedding(query)

        # Retrieve relevant conversation history for the customer
        history_results = self.conversation_history_collection.query(
            query_embeddings=[query_embedding],
            n_results=k_history,
            where={"customer_id": customer_id} if customer_id else None,
            include=['documents']
        )
        retrieved_history = [doc for docs in history_results.get('documents', []) for doc in docs]

        # Retrieve relevant customer profile facts
        profile_results = self.customer_profiles_collection.query(
            query_embeddings=[query_embedding],
            n_results=k_profiles,
            where={"customer_id": customer_id} if customer_id else None,
            include=['documents']
        )
        retrieved_profiles = [doc for docs in profile_results.get('documents', []) for doc in docs]

        return retrieved_history, retrieved_profiles

    def _summarize_context(self, text: str, max_length=150, min_length=30) -> str:
        if not text.strip():
            return ""
        try:
            # Summarize the text if it's too long
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}. Returning original text.")
            return text # Fallback to original text if summarization fails

    def _prepare_llm_input(self, query: str, history: list, profiles: list) -> str:
        context_parts = []
        if history:
            context_parts.append("Previous conversation snippets:\n" + "\n".join(history))
        if profiles:
            context_parts.append("Customer profile facts:\n" + "\n".join(profiles))
        
        combined_context = "\n\n".join(context_parts)

        # Basic token length check (using char count as a rough proxy for simplicity)
        # For production, use a proper tokenizer for the LLM being used
        current_context_len = len(combined_context.split())
        if current_context_len > MAX_LLM_CONTEXT_TOKENS * 0.7: # Summarize if over ~70% of max tokens
            print(f"Context too long ({current_context_len} words), attempting summarization...")
            combined_context = self._summarize_context(combined_context, max_length=int(MAX_LLM_CONTEXT_TOKENS * 0.5))
        
        prompt = f"Given the following context, please answer the customer's query:\n\n{combined_context}\n\nCustomer Query: {query}\nAgent:"
        return prompt

    def _update_memory(self, customer_id: str, query: str, agent_response: str):
        conversation_entry = f"Customer: {query}\nAgent: {agent_response}"
        embedding = self._get_embedding(conversation_entry)
        self.conversation_history_collection.add(
            documents=[conversation_entry],
            embeddings=[embedding],
            metadatas=[{"customer_id": customer_id}],
            ids=[str(uuid.uuid4())]
        )
        
        # Placeholder for customer profile fact extraction and update
        # In a real system, you'd use another LLM call or regex to extract facts
        # e.g., if agent_response mentions a new address or preference.
        # extracted_fact_embedding = self._get_embedding("new fact")
        # self.customer_profiles_collection.add(...)

    def handle_query(self, customer_id: str, query: str) -> str:
        # 1. Retrieve relevant context (history and profile)
        retrieved_history, retrieved_profiles = self._retrieve_context(query, customer_id)
        
        print(f"\nRetrieved History: {retrieved_history}")
        print(f"Retrieved Profiles: {retrieved_profiles}")

        # 2. Prepare LLM input, including summarization if necessary
        llm_prompt = self._prepare_llm_input(query, retrieved_history, retrieved_profiles)

        # 3. Call LLM
        agent_response = self.llm_client.generate_response(llm_prompt)

        # 4. Update memory with the current interaction
        self._update_memory(customer_id, query, agent_response)

        return agent_response

# --- Example Usage --- #
if __name__ == "__main__":
    orchestrator = ConversationOrchestrator()

    customer_id_1 = "user_123"
    customer_id_2 = "user_456"

    # Simulate a conversation for customer 1
    print("\n--- Customer 1 Conversation ---")
    response = orchestrator.handle_query(customer_id_1, "I want to know the status of my order.")
    print(f"Agent: {response}")

    response = orchestrator.handle_query(customer_id_1, "My order number is #XYZ789. Also, what's your return policy?")
    print(f"Agent: {response}")
    
    response = orchestrator.handle_query(customer_id_1, "So I can return within 30 days? What about shipping for a new item?")
    print(f"Agent: {response}")
    
    # Simulate a conversation for customer 2
    print("\n--- Customer 2 Conversation ---")
    response = orchestrator.handle_query(customer_id_2, "Hi, I need help with a product. It's not working.")
    print(f"Agent: {response}")

    response = orchestrator.handle_query(customer_id_2, "What are the steps to troubleshoot?")
    print(f"Agent: {response}")

    # Demonstrate retrieving long context and potential summarization
    # Add more interactions for customer 1 to make history long
    print("\n--- Customer 1 Long Conversation Example ---")
    for i in range(10):
        orchestrator.handle_query(customer_id_1, f"Follow-up query {i}. Any updates on my previous issues?")
    
    # Now a query that might trigger summarization
    response = orchestrator.handle_query(customer_id_1, "Considering all our past conversations, can you give me a summary of my account status and recent interactions?")
    print(f"Agent: {response}")

    # Cleanup ChromaDB collections for demonstration purposes
    # In a real app, you might not want to delete these immediately
    orchestrator.chroma_client.delete_collection(name="customer_conversations")
    orchestrator.chroma_client.delete_collection(name="customer_profiles")