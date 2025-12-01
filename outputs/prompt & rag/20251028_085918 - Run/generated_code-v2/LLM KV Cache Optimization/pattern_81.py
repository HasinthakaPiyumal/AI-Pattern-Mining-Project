import os
import uuid
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

# For LLM (conceptual interaction, as vLLM is a separate service)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# For RAG
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Conceptual LLM Service (Simulating vLLM interaction for KV Cache) ---
class LLMService:
    """
    Conceptual LLM service that simulates interaction with an optimized LLM backend
    (like vLLM) capable of KV cache reuse.

    In a real-world vLLM setup, 'kv_cache_id' would be an internal vLLM mechanism
    to identify and reuse cached prefix states. Here, we simplify by just
    handling the prompt construction based on whether a prefix is "cached".
    """
    def __init__(self, model_name: str = "distilbert/distilgpt2"): # Using a small model for local demo
        print(f"Loading LLM: {model_name}. This may take a moment.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu" # Use "cuda" if GPU is available
        )
        # We'll simulate KV cache by processing only the new parts
        # This is a high-level conceptual simulation, not actual tensor reuse.
        self._conceptual_kv_store = {} # Stores processed output for a given prefix

    def _generate_conceptual_prefix_output(self, prefix_text: str) -> str:
        """
        Simulates processing a prefix to get an 'internal state' or
        a consolidated response that can be built upon.
        In a real vLLM, this would be the actual KV tensors.
        Here, we generate a short summary or acknowledgment.
        """
        if prefix_text not in self._conceptual_kv_store:
            # Simulate initial LLM processing for the prefix
            print(f"LLM processing new prefix: '{prefix_text[:50]}...'")
            # For simplicity, let's just use the prefix itself as a conceptual "output state"
            # Or generate a very short, generic response if prefix is long
            if len(prefix_text) > 200:
                self._conceptual_kv_store[prefix_text] = self.pipeline(
                    prefix_text + "\n(Context processed.)",
                    max_new_tokens=10,
                    num_return_sequences=1,
                    truncation=True
                )[0]['generated_text']
            else:
                 self._conceptual_kv_store[prefix_text] = prefix_text # Simple storage
        return self._conceptual_kv_store[prefix_text]

    def generate(self, prompt: str, kv_cache_id: Optional[str] = None, max_new_tokens: int = 100) -> str:
        """
        Generates text. If kv_cache_id is provided, it implies a prefix was reused.
        """
        if kv_cache_id:
            # In a real vLLM, the vLLM server would handle loading the state
            # Here, we reconstruct the prompt to reflect a "continuation"
            cached_output = self._conceptual_kv_store.get(kv_cache_id, "")
            # print(f"Using conceptual KV cache for prefix: '{kv_cache_id[:50]}...'")
            full_prompt_with_continuation = prompt # The prompt should already be the suffix if cache is used
        else:
            full_prompt_with_continuation = prompt

        print(f"LLM Input (conceptual): {full_prompt_with_continuation[:200]}...")

        response = self.pipeline(
            full_prompt_with_continuation,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True
        )[0]['generated_text']

        # The pipeline output might include the input prompt, so we extract only the new part
        generated_text = response[len(full_prompt_with_continuation):].strip()
        return generated_text if generated_text else response.strip()


# --- 2. KV Cache Manager (Application-level logic) ---
class KVCacheManager:
    """
    Manages application-level mapping of prefixes to conceptual KV cache IDs.
    This would interact with the LLM service to request or provide cache IDs.
    """
    def __init__(self, llm_service: LLMService):
        self._cache: Dict[str, str] = {}  # {prefix_text: kv_cache_id}
        self._llm_service = llm_service
        self._next_kv_id = 0 # Simple incrementing ID for conceptual cache

    def _generate_kv_cache_id(self, prefix_text: str) -> str:
        """Generates a unique ID for a prefix."""
        # In a real vLLM, this might be returned by the vLLM service
        # after it processes the prefix and caches its KV states.
        kv_id = f"kv_cache_{uuid.uuid4().hex}"
        self._llm_service._conceptual_kv_store[kv_id] = self._llm_service._generate_conceptual_prefix_output(prefix_text)
        return kv_id

    def get_longest_cached_prefix(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Finds the longest prefix of `text` that is already in the cache.
        Returns (cached_prefix_text, kv_cache_id)
        """
        longest_prefix = ""
        cached_kv_id = None
        for prefix_text, kv_id in self._cache.items():
            if text.startswith(prefix_text) and len(prefix_text) > len(longest_prefix):
                longest_prefix = prefix_text
                cached_kv_id = kv_id
        return (longest_prefix, cached_kv_id) if longest_prefix else (None, None)

    def store_prefix(self, prefix_text: str) -> str:
        """
        Stores a new prefix in the cache and returns its KV cache ID.
        If already exists, just returns existing ID.
        """
        if prefix_text in self._cache:
            return self._cache[prefix_text]
        kv_id = self._generate_kv_cache_id(prefix_text)
        self._cache[prefix_text] = kv_id
        print(f"Cached new prefix (id: {kv_id}) of length {len(prefix_text)}: '{prefix_text[:50]}...'")
        return kv_id

# --- 3. RAG System Components ---
class KnowledgeBase:
    def __init__(self, docs_path: str = "knowledge_base.txt"):
        self.docs_path = docs_path
        self.vectorstore = None
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        print(f"Loading knowledge base from {self.docs_path}...")
        # Create a dummy knowledge base file if it doesn't exist
        if not os.path.exists(self.docs_path):
            with open(self.docs_path, "w") as f:
                f.write("""
Product Manual for Acme Widget 2000:
The Acme Widget 2000 is a versatile device designed for home and office use.
It features a long-lasting battery life of up to 10 hours and quick charging capabilities.
Troubleshooting: If the device does not turn on, ensure it is fully charged. If issues persist, refer to page 5.
Warranty: All Acme products come with a 1-year limited warranty. For claims, visit our website.
Customer Support: Contact us at support@acme.com or call 1-800-ACME.

Product Manual for Alpha Gadget X:
The Alpha Gadget X is our premium offering, featuring advanced AI capabilities.
It requires a stable internet connection for optimal performance.
Software updates are released monthly and can be downloaded via the settings menu.
For technical support, please consult the online FAQ or our dedicated forum.
Returns: Items can be returned within 30 days of purchase, provided they are in their original packaging.

General Company Policies:
Return Policy: You can return most items within 30 days of purchase.
Shipping: Standard shipping takes 3-5 business days. Expedited shipping options are available.
Privacy Policy: We respect your privacy. See our full policy on our website.
""")
        loader = TextLoader(self.docs_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(chunks, self.embedding_function, persist_directory="./chroma_db")
        print(f"Knowledge base loaded with {len(chunks)} chunks.")

    def retrieve_documents(self, query: str, k: int = 2) -> List[str]:
        """Retrieves relevant documents based on the query."""
        if not self.vectorstore:
            raise ValueError("Knowledge base not loaded.")
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# --- 4. Intelligent Customer Support Agent ---
class CustomerSupportAgent:
    def __init__(self, llm_service: LLMService, knowledge_base: KnowledgeBase):
        self.llm_service = llm_service
        self.knowledge_base = knowledge_base
        self.kv_cache_manager = KVCacheManager(llm_service)
        self.conversation_history: List[str] = []

    def _build_prompt(self, query: str, context: Optional[str] = None, previous_conversation: Optional[str] = None) -> str:
        """Constructs the prompt for the LLM."""
        prompt_parts = []
        if context:
            prompt_parts.append(f"Knowledge Base Context:\n{context}\n")
        if previous_conversation:
            prompt_parts.append(f"Previous Conversation:\n{previous_conversation}\n")
        prompt_parts.append(f"Customer Query: {query}\nAgent:")
        return "\n".join(prompt_parts)

    def answer_question(self, user_query: str) -> str:
        """
        Answers a customer query, utilizing RAG and KV cache reuse.
        """
        # 1. Prepare current conversation context
        current_conversation_text = "\n".join(self.conversation_history + [f"Customer: {user_query}"])

        # 2. Check for KV Cache Reuse on conversation history
        cached_prefix, kv_cache_id = self.kv_cache_manager.get_longest_cached_prefix(current_conversation_text)

        llm_input_suffix = user_query # Default: process full query if no cache or new turn
        rag_context = None

        if cached_prefix and kv_cache_id:
            print(f"\n--- KV Cache HIT! Reusing prefix for '{cached_prefix[:50]}...' (ID: {kv_cache_id}) ---")
            # The LLM conceptually starts from this cached state.
            # We provide only the *new* part of the conversation as the prompt to the LLM.
            # In a real vLLM, the vLLM client would pass 'prefix_id' and the suffix.
            # Here, we pass the user_query as the 'suffix' part of the prompt
            llm_input_suffix = user_query
            # For the RAG part, we might still want to retrieve docs based on the *full* context
            # or just the latest query, depending on strategy. Let's use the full current text.
            rag_query_for_retrieval = current_conversation_text
        else:
            print("\n--- KV Cache MISS or new conversation. Processing full context. ---")
            # Store the current full conversation as a potential prefix for future turns
            self.kv_cache_manager.store_prefix(current_conversation_text)
            rag_query_for_retrieval = current_conversation_text # For RAG retrieval
            kv_cache_id = None # No existing cache ID to pass

        # 3. RAG: Retrieve relevant documents
        retrieved_docs = self.knowledge_base.retrieve_documents(rag_query_for_retrieval)
        rag_context = "\n".join(retrieved_docs)

        # 4. Build LLM prompt
        # If we hit cache, the LLM conceptually processes 'llm_input_suffix'
        # built upon the cached state.
        # The prompt for LLM generation should still include RAG context
        # and possibly a truncated version of the previous conversation if needed by LLM.

        # For demonstration, if KV cache is used, we simplify the prompt by focusing on the query.
        # Otherwise, we use the full history and context.
        if cached_prefix:
            # When KV cache is hit, the 'cached_prefix' already contains the history.
            # We just need to add the RAG context and the current query.
            # The LLM will then continue generation based on its internal cached state
            # and the *new* textual input.
            prompt = self._build_prompt(
                query=user_query,
                context=rag_context,
                previous_conversation="" # History is conceptually 'in' the KV cache
            )
        else:
            prompt = self._build_prompt(
                query=user_query,
                context=rag_context,
                previous_conversation="\n".join(self.conversation_history)
            )

        # 5. Generate response using LLM service
        response = self.llm_service.generate(prompt, kv_cache_id=kv_cache_id)

        # 6. Update conversation history and cache
        self.conversation_history.append(f"Customer: {user_query}")
        self.conversation_history.append(f"Agent: {response}")

        # After generating a response, the *new complete conversation* up to this point
        # could become a new prefix to cache.
        full_current_context = "\n".join(self.conversation_history)
        self.kv_cache_manager.store_prefix(full_current_context)


        return response

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize components
    # Using a small, fast model for demonstration
    # In a real application, you'd use a powerful LLM (e.g., Llama-2, Mistral)
    # and a vLLM server running on GPU.
    llm_service = LLMService(model_name="distilbert/distilgpt2")
    knowledge_base = KnowledgeBase()
    agent = CustomerSupportAgent(llm_service, knowledge_base)

    print("\n--- Starting Customer Support Chatbot (type 'quit' to exit) ---")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        agent_response = agent.answer_question(user_input)
        print(f"Agent: {agent_response}")