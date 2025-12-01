import streamlit as st
import time
import threading
from collections import deque
import uuid

# --- 1.2. Knowledge Base (Vector Store & Document Storage) ---
# Using simple in-memory storage for demonstration. In a real app, use FAISS/Chroma/Pinecone.
class KnowledgeBase:
    def __init__(self):
        # In a real application, use sentence-transformers for embeddings
        # For this demo, we'll simulate document content and IDs
        self.documents = {
            "doc1": "The new iPhone 15 features a A17 Bionic chip and a 48MP main camera. Available in black, blue, green, yellow, and pink.",
            "doc2": "Your order #12345 has been shipped and is expected to arrive on October 26, 2023. Tracking link: example.com/track/12345",
            "doc3": "FAQs: How do I return an item? Visit our returns page and follow the instructions. You have 30 days from purchase.",
            "doc4": "The MacBook Air M2 has a Liquid Retina display and up to 18 hours of battery life. Starts at $1199.",
            "doc5": "Troubleshooting common issues: Restart your device, check internet connection, or contact support.",
            "doc6": "Pre-order the new Samsung Galaxy S24 starting January 17th. Comes with a powerful new processor and enhanced camera features.",
            "doc7": "Your recent purchase, item 'Wireless Earbuds', is eligible for a 1-year warranty. Register it on our website.",
            "doc8": "Shipping policies: Standard shipping takes 3-5 business days. Expedited options are available at checkout.",
        }
        self.document_embeddings = {}
        # Simulate embeddings for simplicity; in reality, use a model
        for doc_id, text in self.documents.items():
            self.document_embeddings[doc_id] = [hash(word) % 1000 for word in text.split()[:5]] # Simplified hash-based embedding

    def search(self, query_embedding, num_results):
        # Simulate vector search. In reality, this would be FAISS/Chroma search.
        time.sleep(0.05) # Simulate retrieval latency
        scores = {}
        query_hash = sum(query_embedding) # Simple query hash

        for doc_id, doc_emb in self.document_embeddings.items():
            # Simulate similarity based on a simple heuristic
            similarity = abs(query_hash - sum(doc_emb))
            scores[doc_id] = similarity
        
        # Sort by simulated similarity (lower score is better)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1])
        
        # Return actual document texts for the top results
        retrieved_texts = []
        retrieved_ids = []
        for doc_id, _ in sorted_docs[:num_results]:
            retrieved_texts.append(self.documents[doc_id])
            retrieved_ids.append(doc_id)
        return retrieved_texts, retrieved_ids

    def get_document_text(self, doc_id):
        return self.documents.get(doc_id, "")

# --- 1.3. LLM Service ---
class LLMService:
    def __init__(self):
        self.active_generations = {}
        self._lock = threading.Lock()
        self.current_generation_id_counter = 0

    def _simulate_llm_generation(self, gen_id, query, context, event_terminate, initial_gen_time=1.0, refinement_time=0.5):
        print(f"[LLM-{gen_id}] Starting generation with context: {context[:50]}...")
        full_context = f"Based on the following information: {' '.join(context)}\n\nUser query: {query}"
        
        initial_response = f"(Speculative {gen_id} Initial) {query.capitalize()}... "
        current_step_time = initial_gen_time

        while not event_terminate.is_set():
            time.sleep(current_step_time)
            with self._lock:
                if gen_id in self.active_generations:
                    if self.active_generations[gen_id]["terminated"]:
                        print(f"[LLM-{gen_id}] TERMINATED.")
                        break # Exit if terminated externally
                    
                    # Simulate progressive generation / refinement
                    if not self.active_generations[gen_id]["finalized"]:
                        # Check if context has been updated for refinement
                        if self.active_generations[gen_id]["context_updated"]:
                            new_context = self.active_generations[gen_id]["context"]
                            print(f"[LLM-{gen_id}] Refinining with new context: {new_context[:50]}...")
                            initial_response = f"(Speculative {gen_id} Refined) {query.capitalize()} based on updated info: {new_context[0][:30]}... "
                            self.active_generations[gen_id]["context_updated"] = False # Reset flag
                            current_step_time = refinement_time # Shorter time for refinement

                        initial_response += f"[Part {int(time.time() * 10) % 10}] "
                        self.active_generations[gen_id]["output"] = initial_response

        # Finalization if not terminated externally
        with self._lock:
            if gen_id in self.active_generations and not self.active_generations[gen_id]["terminated"]:
                self.active_generations[gen_id]["output"] = f"Final response for query: '{query}' based on context: {' '.join(context)}. " + initial_response.replace(f"(Speculative {gen_id}", "")
                self.active_generations[gen_id]["finalized"] = True
                print(f"[LLM-{gen_id}] FINALIZED.")

    def speculate_generate(self, query, context):
        with self._lock:
            self.current_generation_id_counter += 1
            gen_id = f"gen-{self.current_generation_id_counter}"
            event_terminate = threading.Event()
            self.active_generations[gen_id] = {
                "event_terminate": event_terminate,
                "output": f"Thinking about '{query}' with early context...",
                "thread": None,
                "terminated": False,
                "finalized": False,
                "context": context,
                "context_updated": False
            }
            thread = threading.Thread(target=self._simulate_llm_generation, args=(gen_id, query, context, event_terminate))
            self.active_generations[gen_id]["thread"] = thread
            thread.start()
            return gen_id

    def terminate_generation(self, gen_id):
        with self._lock:
            if gen_id in self.active_generations and not self.active_generations[gen_id]["finalized"]:
                print(f"[LLM-{gen_id}] Signalling termination...")
                self.active_generations[gen_id]["event_terminate"].set()
                self.active_generations[gen_id]["terminated"] = True

    def continue_generation(self, gen_id, new_context):
        with self._lock:
            if gen_id in self.active_generations and not self.active_generations[gen_id]["finalized"] and not self.active_generations[gen_id]["terminated"]:
                print(f"[LLM-{gen_id}] Updating context for continuation...")
                self.active_generations[gen_id]["context"] = new_context
                self.active_generations[gen_id]["context_updated"] = True
                # The _simulate_llm_generation loop will pick up the context_updated flag

    def get_generation_output(self, gen_id):
        with self._lock:
            return self.active_generations.get(gen_id, {}).get("output", "")
            
    def is_generation_finalized(self, gen_id):
        with self._lock:
            return self.active_generations.get(gen_id, {}).get("finalized", False)

    def wait_for_generation(self, gen_id):
        thread = None
        with self._lock:
            if gen_id in self.active_generations:
                thread = self.active_generations[gen_id]["thread"]
        if thread and thread.is_alive():
            thread.join()

# --- 1.4. Retrieval Service (Staged & Concurrent) ---
class RetrievalService:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def staged_search(self, query, num_stages=3, base_results=2, increase_per_stage=2):
        query_embedding = [hash(word) % 1000 for word in query.split()[:5]] # Simulate query embedding
        all_retrieved_docs_ids = set()
        
        for stage in range(num_stages):
            time.sleep(0.5) # Simulate retrieval processing time for each stage
            num_results = base_results + (stage * increase_per_stage)
            retrieved_texts, retrieved_ids = self.kb.search(query_embedding, num_results)
            print(f"[Retrieval] Stage {stage+1}: Retrieved {len(retrieved_ids)} documents. IDs: {retrieved_ids}")
            
            current_stage_ids = set(retrieved_ids)
            
            # Yield text for the current stage's unique documents (if any)
            # Or simply yield all documents found up to this stage
            yield retrieved_texts, current_stage_ids

            # If this is the last stage, ensure all_retrieved_docs_ids is complete
            if stage == num_stages - 1:
                all_retrieved_docs_ids.update(current_stage_ids)


# --- 1.5. Dynamic System Monitor ---
class SystemMonitor:
    def __init__(self, max_pending_requests=2):
        self.max_pending_requests = max_pending_requests
        self._pending_requests = 0
        self._lock = threading.Lock()

    def can_pipeline(self):
        with self._lock:
            return self._pending_requests < self.max_pending_requests

    def increment_pending(self):
        with self._lock:
            self._pending_requests += 1
            print(f"[Monitor] Increment pending: {self._pending_requests}")

    def decrement_pending(self):
        with self._lock:
            self._pending_requests = max(0, self._pending_requests - 1)
            print(f"[Monitor] Decrement pending: {self._pending_requests}")

# --- 2. Orchestration Logic (RAG Pipeline) ---
class QueryHandler:
    def __init__(self, knowledge_base, llm_service, system_monitor):
        self.kb = knowledge_base
        self.llm = llm_service
        self.monitor = system_monitor

    def _get_document_similarity_score(self, set1, set2):
        # Simple heuristic: Jaccard similarity of document IDs
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def handle_query(self, query):
        self.monitor.increment_pending()
        final_response = ""
        try:
            if self.monitor.can_pipeline():
                st.info("Pipelining active! Starting speculative generation...")
                retrieval_thread_event = threading.Event()
                llm_generation_id = None
                latest_retrieved_ids = set()
                
                # Use a shared list to communicate retrieved docs from thread to main
                retrieval_results_queue = []

                def run_staged_retrieval():
                    for texts, ids in self.kb_retrieval.staged_search(query):
                        retrieval_results_queue.append((texts, ids))
                        time.sleep(0.1) # Give main thread time to pick up
                    retrieval_thread_event.set() # Signal completion

                self.kb_retrieval = RetrievalService(self.kb) # Initialize here
                retrieval_thread = threading.Thread(target=run_staged_retrieval)
                retrieval_thread.start()

                prev_retrieved_ids = set()
                last_llm_update_time = time.time()

                while retrieval_thread.is_alive() or retrieval_results_queue:
                    if retrieval_results_queue:
                        current_texts, current_ids = retrieval_results_queue.pop(0)
                        latest_retrieved_ids = current_ids # Update latest

                        similarity = self._get_document_similarity_score(prev_retrieved_ids, current_ids)
                        
                        if llm_generation_id is None: # First stage, start speculative
                            print(f"[QueryHandler] First stage retrieval. Starting speculative LLM.")
                            llm_generation_id = self.llm.speculate_generate(query, current_texts)
                            prev_retrieved_ids = current_ids
                            st.session_state.chat_history.append(("AI (Speculative)", self.llm.get_generation_output(llm_generation_id)))
                            st.experimental_rerun()
                        elif similarity < 0.8 and (time.time() - last_llm_update_time > 1.0): # Threshold for new speculation or update
                            print(f"[QueryHandler] Significant change in docs (similarity {similarity:.2f}). Terminating old, starting new speculation.")
                            self.llm.terminate_generation(llm_generation_id)
                            llm_generation_id = self.llm.speculate_generate(query, current_texts)
                            prev_retrieved_ids = current_ids
                            last_llm_update_time = time.time()
                            st.session_state.chat_history.append(("AI (Speculative Updated)", self.llm.get_generation_output(llm_generation_id)))
                            st.experimental_rerun()
                        elif similarity >= 0.8 and llm_generation_id and (time.time() - last_llm_update_time > 1.0):
                            print(f"[QueryHandler] Docs similar (similarity {similarity:.2f}). Continuing LLM with refined context.")
                            self.llm.continue_generation(llm_generation_id, current_texts)
                            last_llm_update_time = time.time()
                            st.session_state.chat_history.append(("AI (Speculative Refined)", self.llm.get_generation_output(llm_generation_id)))
                            st.experimental_rerun()
                    
                    # Update speculative output in UI regularly
                    if llm_generation_id and not self.llm.is_generation_finalized(llm_generation_id):
                        current_spec_output = self.llm.get_generation_output(llm_generation_id)
                        if st.session_state.chat_history and st.session_state.chat_history[-1][0].startswith("AI (Speculative") and st.session_state.chat_history[-1][1] != current_spec_output:
                            st.session_state.chat_history[-1] = ("AI (Speculative) Current", current_spec_output) # Update in place
                            st.experimental_rerun()
                        elif not st.session_state.chat_history or not st.session_state.chat_history[-1][0].startswith("AI (Speculative"):
                            st.session_state.chat_history.append(("AI (Speculative)", current_spec_output))
                            st.experimental_rerun()

                    time.sleep(0.2) # Small delay to prevent busy-waiting
                
                # Ensure the last speculative generation finishes or get the final one
                if llm_generation_id: # and not self.llm.is_generation_finalized(llm_generation_id):
                    self.llm.wait_for_generation(llm_generation_id)
                    final_response = self.llm.get_generation_output(llm_generation_id)
                    st.success("Pipelining complete! Final response generated.")
                else:
                    st.warning("No speculative generation initiated. Falling back to sequential.")
                    # Fallback if no speculation occurred (e.g., very fast retrieval)
                    # Perform a final, full retrieval
                    final_docs_texts, _ = self.kb.search([hash(word) % 1000 for word in query.split()[:5]], 8)
                    time.sleep(2) # Simulate full LLM generation
                    final_response = f"(Sequential Fallback) Final answer for '{query}' based on: {final_docs_texts[0][:50]}..."


            else: # Sequential execution
                st.info("System busy. Executing sequentially.")
                print(f"[QueryHandler] Sequential execution for query: {query}")
                # Simulate full retrieval
                retrieval_service = RetrievalService(self.kb)
                all_docs_gen = retrieval_service.staged_search(query, num_stages=3, base_results=8, increase_per_stage=0)
                final_docs_texts, final_docs_ids = next(all_docs_gen) # Get the full set from the first (and only) stage
                print(f"[QueryHandler] Sequential: Retrieved {len(final_docs_ids)} documents. IDs: {final_docs_ids}")

                time.sleep(2) # Simulate full LLM generation after retrieval
                llm_response = f"Final answer for '{query}' based on: {final_docs_texts[0][:50]}..."
                final_response = llm_response

        finally:
            self.monitor.decrement_pending()
        
        return final_response

# --- Streamlit Frontend ---
st.set_page_config(page_title="RAG Speculative Pipelining Chatbot")
st.title("🛒 E-commerce Chatbot with Speculative RAG")

# Initialize services
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
if "llm" not in st.session_state:
    st.session_state.llm = LLMService()
if "monitor" not in st.session_state:
    st.session_state.monitor = SystemMonitor(max_pending_requests=1) # Set to 1 to easily trigger sequential
if "query_handler" not in st.session_state:
    st.session_state.query_handler = QueryHandler(st.session_state.kb, st.session_state.llm, st.session_state.monitor)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for role, message in st.session_state.chat_history:
    with st.chat_message(role.split(" ")[0].lower()):
        st.write(message)

user_query = st.chat_input("Ask a question about products, orders, or support...")

if user_query:
    st.session_state.chat_history.append(("You", user_query))
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.chat_message("ai"):
        with st.spinner("Processing your request..."):
            # The handle_query will update chat_history and trigger reruns if pipelining
            final_ai_response = st.session_state.query_handler.handle_query(user_query)

            # Ensure the final response is displayed clearly
            # Remove any lingering speculative messages if they exist before final display
            if st.session_state.chat_history and st.session_state.chat_history[-1][0].startswith("AI (Speculative"):
                st.session_state.chat_history.pop()
            st.session_state.chat_history.append(("AI", final_ai_response))
            st.write(final_ai_response)

st.sidebar.header("System Status")
st.sidebar.write(f"Pending LLM Requests: {st.session_state.monitor._pending_requests}")
st.sidebar.write(f"Pipelining Enabled: {st.session_state.monitor.can_pipeline()}")

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()