import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Placeholder for LLM response
def mock_llm_response(prompt: str) -> str:
    if "how do I reset my password" in prompt.lower():
        return "To reset your password, please visit our website, click 'Forgot Password' and follow the instructions. We also sent you an email with a reset link based on your previous interaction."
    elif "what are my preferred products" in prompt.lower():
        return "Based on your past interactions, you've shown interest in our premium subscription and smart home devices. Is there anything specific you'd like to know about them?"
    elif "thank you" in prompt.lower() or "bye" in prompt.lower():
        return "You're welcome! Feel free to reach out if you have any more questions. Have a great day!"
    else:
        return f"Hello! How can I assist you with '{prompt}' today? I remember some of our past conversations."

class MemoryAugmentedChatbot:
    def __init__(self, chroma_path: str = "./chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.customer_history_collection = self.chroma_client.get_or_create_collection(name="customer_history")
        self.customer_preferences_collection = self.chroma_client.get_or_create_collection(name="customer_preferences")
        self.learned_solutions_collection = self.chroma_client.get_or_create_collection(name="learned_solutions")

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    def add_to_memory(self, collection_name: str, text: str, metadata: Dict[str, Any], id: str):
        collection = getattr(self, f"{collection_name}_collection")
        collection.add(
            embeddings=[self._get_embedding(text)],
            documents=[text],
            metadatas=[metadata],
            ids=[id]
        )

    def retrieve_from_memory(self, collection_name: str, query: str, n_results: int = 2) -> List[Dict[str, Any]]:
        collection = getattr(self, f"{collection_name}_collection")
        results = collection.query(
            query_embeddings=[self._get_embedding(query)],
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        return results.get('documents', [[]])[0] if results.get('documents') else []

    def augment_prompt(self, current_query: str, retrieved_history: List[str], retrieved_preferences: List[str], retrieved_solutions: List[str]) -> str:
        context_parts = []
        if retrieved_history:
            context_parts.append("\nPast Interactions:\n" + "\n".join(retrieved_history))
        if retrieved_preferences:
            context_parts.append("\nCustomer Preferences:\n" + "\n".join(retrieved_preferences))
        if retrieved_solutions:
            context_parts.append("\nRelevant Solutions:\n" + "\n".join(retrieved_solutions))

        if context_parts:
            context_string = "\n".join(context_parts)
            return f"Given the following context:\n{context_string}\n\nUser query: {current_query}\nBot:"
        else:
            return f"User query: {current_query}\nBot:"

    def get_response(self, user_query: str) -> str:
        # Retrieve relevant memory
        history = self.retrieve_from_memory("customer_history", user_query)
        preferences = self.retrieve_from_memory("customer_preferences", user_query)
        solutions = self.retrieve_from_memory("learned_solutions", user_query)

        # Augment the prompt for the LLM
        augmented_prompt = self.augment_prompt(user_query, history, preferences, solutions)

        # Get response from the mock LLM
        llm_response = mock_llm_response(augmented_prompt)

        # Store current interaction in history
        self.add_to_memory("customer_history", user_query, {"type": "user_query"}, f"user_query_{len(self.customer_history_collection.get()['ids'])}")
        self.add_to_memory("customer_history", llm_response, {"type": "bot_response"}, f"bot_response_{len(self.customer_history_collection.get()['ids'])}")

        return llm_response

# Streamlit UI
st.set_page_config(page_title="Memory-Augmented Chatbot")
st.title("Memory-Augmented Customer Support Chatbot")

# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = MemoryAugmentedChatbot()
    # Add some initial data to memory for demonstration
    st.session_state.chatbot.add_to_memory("customer_preferences", "Prefers email support", {"type": "contact_preference"}, "pref_email_1")
    st.session_state.chatbot.add_to_memory("customer_preferences", "Interested in smart home devices and premium subscription", {"type": "product_interest"}, "pref_product_1")
    st.session_state.chatbot.add_to_memory("learned_solutions", "Steps to reset password: Go to website, click 'Forgot Password', follow email instructions.", {"topic": "password_reset"}, "sol_pass_1")
    st.session_state.chatbot.add_to_memory("learned_solutions", "Troubleshooting for slow internet: Restart router, check cables, contact ISP.", {"topic": "internet_troubleshoot"}, "sol_internet_1")
    st.session_state.chatbot.add_to_memory("customer_history", "Initial query about product features.", {"type": "initial_query"}, "hist_init_1")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.chatbot.get_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
