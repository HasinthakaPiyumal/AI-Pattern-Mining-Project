
import streamlit as st
import os
from typing import List, Dict, Any

# Mocking external libraries for a self-contained example
# In a real application, these would be installed and configured

# --- Mock LangChain Components ---
class MockChatMessage:
    def __init__(self, content: str, role: str):
        self.content = content
        self.role = role

    def __str__(self):
        return f"{self.role}: {self.content}"

class MockHumanMessage(MockChatMessage):
    def __init__(self, content: str):
        super().__init__(content, "human")

class MockAIMessage(MockChatMessage):
    def __init__(self, content: str):
        super().__init__(content, "ai")

class MockConversationBufferWindowMemory:
    def __init__(self, k: int = 5):
        self.k = k
        self.buffer = []

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]):
        self.buffer.append(MockHumanMessage(inputs["human_input"]))
        self.buffer.append(MockAIMessage(outputs["ai_response"]))
        self.buffer = self.buffer[-self.k*2:] # Keep k human and k ai messages

    def load_memory_variables(self, input_variables: List[str]) -> Dict[str, Any]:
        history_str = "\n".join([str(msg) for msg in self.buffer])
        return {"history": history_str}

    def clear(self):
        self.buffer = []


class MockChroma:
    def __init__(self):
        self.store = {}
        self.id_counter = 0

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, str]] = None):
        for i, doc in enumerate(documents):
            doc_id = str(self.id_counter)
            self.store[doc_id] = {"document": doc, "metadata": metadatas[i] if metadatas else {}}
            self.id_counter += 1
        print(f"Added {len(documents)} documents to mock Chroma.")

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        # Very basic mock similarity search - returns all documents if query is empty
        # In a real scenario, this would involve embeddings and vector similarity
        results = []
        for doc_id, data in self.store.items():
            if query.lower() in data["document"].lower():
                results.append({"document": data["document"], "metadata": data["metadata"]})
        return results[:k]

# Mocking an Embedding Model (Sentence-Transformers)
class MockSentenceTransformer:
    def encode(self, texts: List[str], *args, **kwargs):
        # Returns dummy embeddings for demonstration
        return [[0.1] * 384 for _ in texts]

# Mocking an OpenAI-like LLM
class MockOpenAILLM:
    def __init__(self, model_name: str = "gpt-3.5-turbo"): # Model name is just for logging
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        # Simulate LLM response based on keywords
        prompt_lower = prompt.lower()
        if "order status" in prompt_lower or "track my order" in prompt_lower:
            return "To check your order status, please provide your order number. You can usually find it in your confirmation email." # Mock order status logic
        elif "product recommendation" in prompt_lower or "recommend a product" in prompt_lower:
            return "I can help with product recommendations! What kind of product are you looking for? For example, are you interested in electronics, apparel, or home goods?" # Mock product recommendation logic
        elif "return policy" in prompt_lower or "how do i return" in prompt_lower:
            return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our returns page for more details." # Mock return policy logic
        elif "shipping" in prompt_lower:
            return "We offer various shipping options. Standard shipping usually takes 3-5 business days. Express options are also available at checkout." # Mock shipping info
        elif "troubleshoot" in prompt_lower:
            return "I understand you're having trouble. Could you please describe the issue in more detail? For example, what product is it, and what problem are you encountering?" # Mock troubleshooting
        elif "customer information" in prompt_lower and "update" in prompt_lower:
            return "I can help you update your customer information. Please confirm what details you'd like to change." # Mock user info update
        else:
            # Default response, potentially using retrieved context
            context_match = self._find_context_match(prompt)
            if context_match:
                return f"Based on our knowledge base: {context_match}. How else can I assist?"
            return "I'm sorry, I couldn't find a direct answer to that. Can you please rephrase or provide more details?"

    def _find_context_match(self, query: str) -> str:
        # Simulate finding a relevant piece of context from the mock KB
        mock_kb = [
            "Our customer service hours are Monday to Friday, 9 AM to 5 PM EST.",
            "You can reset your password by clicking 'Forgot Password' on the login page.",
            "We accept Visa, MasterCard, American Express, and PayPal.",
            "For warranty claims, please contact our support team with your proof of purchase."
        ]
        for item in mock_kb:
            if query.lower() in item.lower():
                return item
        return ""

# --- System Components --- 

class QueryProcessor:
    def __init__(self):
        # In a real system, this would use a fine-tuned model
        pass

    def classify_intent_and_complexity(self, query: str) -> Dict[str, str]:
        query_lower = query.lower()
        intent = "general_inquiry"
        complexity = "simple"

        if any(keyword in query_lower for keyword in ["order status", "track my order"]):
            intent = "order_status"
            complexity = "simple"
        elif any(keyword in query_lower for keyword in ["recommend", "suggest product"]):
            intent = "product_recommendation"
            complexity = "complex"
        elif any(keyword in query_lower for keyword in ["return", "warranty", "policy"]):
            intent = "returns_policy"
            complexity = "simple"
        elif any(keyword in query_lower for keyword in ["troubleshoot", "issue", "fix"]):
            intent = "technical_support"
            complexity = "complex"
        elif any(keyword in query_lower for keyword in ["update info", "change address"]):
            intent = "user_profile_management"
            complexity = "simple"
        
        # A more sophisticated model would classify complexity based on sentence structure, entities, etc.
        if len(query.split()) > 10 or any(q_word in query_lower for q_word in ["how to", "why is", "what if"]):
            complexity = "complex"

        return {"intent": intent, "complexity": complexity}

class KnowledgeBaseManager:
    def __init__(self):
        self.vector_db = MockChroma()
        self.embedding_model = MockSentenceTransformer()
        self._populate_knowledge_base()

    def _populate_knowledge_base(self):
        # Mock product data and FAQs
        docs = [
            "The Acme Laptop Pro features an Intel i7 processor, 16GB RAM, and a 512GB SSD. It has a 14-inch display.",
            "Our return policy states that items can be returned within 30 days of purchase if unopened and in original packaging.",
            "For customer support, please visit our help center or call us at 1-800-BUY-ACME.",
            "Shipping typically takes 3-5 business days for standard delivery within the continental US.",
            "The latest software update for the Acme Smartwatch improves battery life and adds new fitness tracking features.",
            "You can track your order using the tracking number provided in your shipping confirmation email.",
            "Payment options include Visa, MasterCard, American Express, and PayPal."
        ]
        metadatas = [
            {"type": "product_spec", "product_id": "LAP001"},
            {"type": "policy", "policy_name": "returns"},
            {"type": "contact_info"},
            {"type": "shipping"},
            {"type": "product_update", "product_id": "SMW002"},
            {"type": "order_info"},
            {"type": "payment_options"}
        ]
        self.vector_db.add_documents(docs, metadatas)

    def retrieve_knowledge(self, query: str, intent: str, complexity: str, k: int = 3) -> List[Dict[str, Any]]:
        # In a real RAG system, query would be embedded and used for vector search.
        # Here, we do a simple keyword match for demonstration.
        print(f"Retrieving knowledge for query: '{query}' with intent '{intent}' and complexity '{complexity}'")
        if complexity == "complex" or intent != "general_inquiry":
            # Simulate more targeted retrieval for complex or specific intents
            return self.vector_db.similarity_search(query, k=k)
        else:
            return self.vector_db.similarity_search(query, k=k)

class LLMAgent:
    def __init__(self):
        self.llm = MockOpenAILLM()
        self.short_term_memory = MockConversationBufferWindowMemory(k=5) # Last 5 turns
        self.kb_manager = KnowledgeBaseManager()
        self.query_processor = QueryProcessor()

    def generate_response(self, user_query: str) -> str:
        # 1. Query Processing & Intent Classification
        classification = self.query_processor.classify_intent_and_complexity(user_query)
        intent = classification["intent"]
        complexity = classification["complexity"]
        
        st.sidebar.markdown(f"**Detected Intent:** {intent}")
        st.sidebar.markdown(f"**Detected Complexity:** {complexity}")

        # 2. Retrieve from Long-Term Memory (Knowledge Base)
        # The retrieval strategy can be adapted based on intent/complexity
        retrieved_knowledge = self.kb_manager.retrieve_knowledge(user_query, intent, complexity)
        kb_context = "\n".join([doc["document"] for doc in retrieved_knowledge]) if retrieved_knowledge else "No specific knowledge found."
        
        st.sidebar.markdown(f"**Retrieved KB Context:**\n```\n{kb_context}\n```")

        # 3. Load Short-Term Memory (Conversational History)
        memory_variables = self.short_term_memory.load_memory_variables(["history"])
        conversation_history = memory_variables.get("history", "No prior conversation.")

        st.sidebar.markdown(f"**Conversation History:**\n```\n{conversation_history}\n```")

        # 4. Construct the LLM Prompt
        # The prompt dynamically includes history, retrieved knowledge, and current query
        prompt_template = f"""
You are an Adaptive E-commerce Customer Support Agent. 

Conversation History:
{conversation_history}

E-commerce Knowledge Base Context:
{kb_context}

Based on the conversation history and the provided knowledge base context, please answer the user's query thoughtfully and adaptively. If you don't know the answer, politely state that you cannot provide it. Keep your responses concise and helpful.

User Query: {user_query}
Agent Response:"""
        
        st.sidebar.markdown(f"**Full LLM Prompt:**\n```\n{prompt_template}\n```")

        # 5. LLM Core Module - Generate Response
        raw_response = self.llm.invoke(prompt_template)

        # 6. Response Generation & Post-processing Module
        final_response = self._post_process_response(raw_response, intent)

        # 7. Save to Short-Term Memory
        self.short_term_memory.save_context(
            inputs={"human_input": user_query},
            outputs={"ai_response": final_response}
        )

        return final_response

    def _post_process_response(self, raw_response: str, intent: str) -> str:
        # Simple post-processing and guardrails
        processed_response = raw_response.strip()

        # Example: Add specific calls to action based on intent
        if intent == "order_status":
            if "order number" in processed_response.lower():
                processed_response += " Please enter your order number below."
            else:
                processed_response += " You can also visit our order tracking page: [Track My Order](https://example.com/track)."
        elif intent == "product_recommendation":
            processed_response += " For personalized recommendations, you can browse our categories or let me know your preferences!"
        
        # Basic safety check
        if "offensive" in processed_response.lower() or "inappropriate" in processed_response.lower():
            return "I'm sorry, I cannot assist with that request. Please ask a question related to our e-commerce products or services."

        return processed_response


# --- Streamlit UI --- 

st.set_page_config(page_title="Adaptive E-commerce Support Agent", layout="centered")
st.title("🛒 Adaptive E-commerce Customer Support")
st.subheader("Your intelligent assistant for all shopping inquiries")

# Initialize LLMAgent if not already in session state
if "agent" not in st.session_state:
    st.session_state.agent = LLMAgent()
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.generate_response(prompt)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.title("Agent Insights")
st.sidebar.info("This panel shows internal workings of the Adaptive Agent.\n\n**Note:** This is a simplified, mocked implementation for demonstration purposes. A real system would use actual LLMs, vector databases, and sophisticated classification models.")

# Optional: Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.agent.short_term_memory.clear()
    st.experimental_rerun()

# Monitoring & Evaluation (Conceptual - for a real system)
st.sidebar.markdown("### Monitoring & Evaluation (Conceptual)")
st.sidebar.markdown("- **User Feedback:** Collect thumbs up/down on responses (not implemented here).")
st.sidebar.markdown("- **LLM Metrics:** Track token usage, latency (via LangSmith/WandB, not implemented here).")
st.sidebar.markdown("- **Evaluation Sets:** Periodically evaluate agent on new data (via TruLens/Evals, not implemented here).")

# Fine-tuning & Adaptation (Conceptual - for a real system)
st.sidebar.markdown("### Fine-tuning & Adaptation (Conceptual)")
st.sidebar.markdown("- **SFT/RLHF:** Periodically fine-tune LLM on new e-commerce data and user feedback using TRL/Accelerate (not implemented here).")
