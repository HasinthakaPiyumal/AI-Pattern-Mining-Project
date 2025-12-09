import uuid

class CritiqueResult:
    def __init__(self, is_relevant=False, is_complete=False, is_accurate=False, is_coherent=False, feedback="", needs_further_retrieval=False, needs_refinement=False):
        self.is_relevant = is_relevant
        self.is_complete = is_complete
        self.is_accurate = is_accurate
        self.is_coherent = is_coherent
        self.feedback = feedback
        self.needs_further_retrieval = needs_further_retrieval
        self.needs_refinement = needs_refinement

    def __str__(self):
        return (f"Relevant: {self.is_relevant}, Complete: {self.is_complete}, "
                f"Accurate: {self.is_accurate}, Coherent: {self.is_coherent}, "
                f"Feedback: '{self.feedback}', Further Retrieval: {self.needs_further_retrieval}, "
                f"Refinement: {self.needs_refinement}")

class KnowledgeBase:
    def __init__(self):
        self.documents = {}

    def add_documents(self, docs_content):
        for content in docs_content:
            doc_id = str(uuid.uuid4())
            self.documents[doc_id] = {"id": doc_id, "content": content}
        print(f"KnowledgeBase: Added {len(docs_content)} documents.")

    def retrieve(self, query_text, k=3):
        query_words = set(query_text.lower().split())
        scored_documents = []

        for doc_id, doc_data in self.documents.items():
            content = doc_data["content"].lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                scored_documents.append((score, doc_id, doc_data["content"]))

        scored_documents.sort(key=lambda x: x[0], reverse=True)
        return [(doc[1], doc[2]) for doc in scored_documents[:k]]

class SelfRAG_LLM:
    def __init__(self, knowledge_base, llm_model=None):
        self.knowledge_base = knowledge_base
        self.llm_model = llm_model # Simulated LLM for generation
        self.retrieval_trigger_keywords = ["return policy", "shipping cost", "warranty", "troubleshoot", "compatible"]
        self.complex_query_indicators = ["how to", "explain", "compare"]

    def _predict_retrieval_token(self, customer_query, current_context=""):
        query_lower = customer_query.lower()
        needs_retrieval = any(keyword in query_lower for keyword in self.retrieval_trigger_keywords) or \
                          any(indicator in query_lower for indicator in self.complex_query_indicators)
        
        refined_query = customer_query
        if "return policy" in query_lower:
            refined_query = "detailed return policy information"
        elif "shipping cost" in query_lower:
            refined_query = "shipping costs to different regions"
        
        print(f"  LLM: Predicted needs_retrieval={needs_retrieval} for query: '{customer_query}'")
        return needs_retrieval, refined_query

    def _retrieve_and_embed(self, query_text):
        # In this simulation, we directly use query_text for keyword-based retrieval
        print(f"  LLM: Retrieving documents for query: '{query_text}'")
        retrieved_docs = self.knowledge_base.retrieve(query_text, k=2)
        return retrieved_docs

    def _critique_retrieval(self, original_query, retrieved_documents):
        query_lower = original_query.lower()
        feedback = []
        is_relevant = True
        is_complete = True
        needs_further_retrieval = False

        if not retrieved_documents:
            is_relevant = False
            is_complete = False
            feedback.append("No relevant documents found.")
            needs_further_retrieval = True
        else:
            retrieved_content = " ".join([doc[1].lower() for doc in retrieved_documents])
            if "return policy" in query_lower and "return conditions" not in retrieved_content:
                is_complete = False
                feedback.append("Missing specific return conditions.")
                needs_further_retrieval = True
            if "warranty" in query_lower and "warranty period" not in retrieved_content:
                is_complete = False
                feedback.append("Missing warranty period details.")

        critique = CritiqueResult(is_relevant=is_relevant, is_complete=is_complete, 
                                  feedback="; ".join(feedback), needs_further_retrieval=needs_further_retrieval)
        print(f"  LLM: Retrieval Critique: {critique}")
        return critique

    def _generate_answer(self, customer_query, retrieved_context=""):
        base_answer = f"I'm here to help with your request about '{customer_query}'."
        if retrieved_context:
            base_answer = f"Based on the information I found about '{customer_query}' and the following context: {retrieved_context}, I can tell you: "
            if "return policy" in customer_query.lower() and "30-day return window" in retrieved_context.lower():
                base_answer += "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition and packaging."
            elif "shipping cost" in customer_query.lower() and "standard shipping is $5" in retrieved_context.lower():
                base_answer += "Standard shipping within the continental U.S. costs $5. Expedited options are also available."
            elif "warranty" in customer_query.lower() and "1-year manufacturer's warranty" in retrieved_context.lower():
                 base_answer += "The product comes with a 1-year manufacturer's warranty covering defects in materials and workmanship."
            else:
                base_answer += "Please provide more details for me to give a precise answer based on the context."
        else:
            if "hello" in customer_query.lower():
                base_answer = "Hello! How can I assist you today?"
            elif "thank you" in customer_query.lower():
                base_answer = "You're welcome! Let me know if you need anything else."

        print(f"  LLM: Generated initial answer: '{base_answer}'")
        return base_answer

    def _critique_generation(self, original_query, generated_answer, retrieved_context):
        query_lower = original_query.lower()
        answer_lower = generated_answer.lower()
        feedback = []
        is_accurate = True
        is_coherent = True
        is_complete = True
        needs_refinement = False

        # Simulate accuracy check
        if "return policy" in query_lower and "30 days" not in answer_lower and "30-day return window" in retrieved_context.lower():
            is_accurate = False
            feedback.append("Answer did not mention the 30-day return window.")
            needs_refinement = True
        if "shipping cost" in query_lower and "$5" not in answer_lower and "standard shipping is $5" in retrieved_context.lower():
            is_accurate = False
            feedback.append("Answer did not mention the standard $5 shipping cost.")
            needs_refinement = True
        
        # Simulate completeness check
        if "how to return" in query_lower and "steps" not in answer_lower:
            is_complete = False
            feedback.append("Answer is missing steps for how to return.")
            needs_refinement = True

        critique = CritiqueResult(is_accurate=is_accurate, is_coherent=is_coherent, is_complete=is_complete,
                                  feedback="; ".join(feedback), needs_refinement=needs_refinement)
        print(f"  LLM: Generation Critique: {critique}")
        return critique

    def respond_to_customer(self, customer_query, max_reflection_steps=3):
        print(f"\nCustomer: {customer_query}")
        conversation_log = []
        current_answer = ""
        retrieved_context = ""

        # Step 1: Initial Generation (without explicit retrieval token prediction for the very first pass)
        needs_retrieval, refined_retrieval_query = self._predict_retrieval_token(customer_query)
        if not needs_retrieval:
            current_answer = self._generate_answer(customer_query)
            gen_critique = self._critique_generation(customer_query, current_answer, retrieved_context)
            conversation_log.append(f"Initial Answer (no retrieval): {current_answer}")
            if not gen_critique.needs_refinement:
                return current_answer, conversation_log

        # Self-Reflection Loop
        for step in range(max_reflection_steps):
            print(f"\n--- Reflection Step {step + 1} ---")
            if needs_retrieval:
                retrieved_docs = self._retrieve_and_embed(refined_retrieval_query)
                retrieval_critique = self._critique_retrieval(customer_query, retrieved_docs)
                
                if retrieval_critique.needs_further_retrieval and step < max_reflection_steps - 1:
                    print("  LLM: Retrieval needs further refinement. Adjusting query.")
                    refined_retrieval_query += " more details"
                    conversation_log.append(f"Step {step+1}: Retrieval needed further refinement. New query: '{refined_retrieval_query}'")
                    continue # Retry retrieval in next step
                
                if retrieved_docs:
                    retrieved_context = " ".join([doc[1] for doc in retrieved_docs])
                    conversation_log.append(f"Step {step+1}: Retrieved context: '{retrieved_context[:100]}...' ")
                else:
                    retrieved_context = ""
                    conversation_log.append(f"Step {step+1}: No documents retrieved.")
            
            current_answer = self._generate_answer(customer_query, retrieved_context)
            gen_critique = self._critique_generation(customer_query, current_answer, retrieved_context)
            conversation_log.append(f"Step {step+1}: Generated Answer: {current_answer}")

            if not gen_critique.needs_refinement:
                print("  LLM: Answer deemed satisfactory after reflection.")
                return current_answer, conversation_log
            else:
                print(f"  LLM: Answer needs refinement: {gen_critique.feedback}")
                # Simulate refinement strategy: e.g., ask clarifying question or retry generation
                if "missing steps" in gen_critique.feedback.lower() or "missing details" in gen_critique.feedback.lower():
                    current_answer = f"{current_answer} Could you please provide more specific details or elaborate on what steps you are looking for?"
                elif "not mention" in gen_critique.feedback.lower():
                    current_answer = f"{current_answer} Let me clarify: {gen_critique.feedback.replace('Answer did not mention', 'It seems I missed mentioning')}"
                conversation_log.append(f"Step {step+1}: Refined Answer: {current_answer}")

        print("  LLM: Reached max reflection steps. Returning best effort answer.")
        return current_answer, conversation_log

# --- Simulated Chatbot Frontend / Main Execution ---
if __name__ == "__main__":
    # 1. Setup Knowledge Base
    kb = KnowledgeBase()
    kb.add_documents([
        "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags and packaging. For electronics, a 15-day return window applies. Read full return conditions on our website.",
        "Standard shipping for orders within the continental U.S. is a flat rate of $5. Expedited shipping options (2-day, overnight) are available at an additional cost, calculated at checkout. International shipping rates vary.",
        "All our electronic products come with a 1-year manufacturer's warranty against defects in materials and workmanship. This warranty does not cover accidental damage or misuse. Extended warranties are available for purchase.",
        "To return an item, please visit our 'Returns' page on the website, fill out the return request form, and print the shipping label. Pack the item securely in its original packaging and drop it off at any USPS location.",
        "Troubleshooting guide for Product X: Ensure the device is charged. Check all cable connections. Refer to page 5 of the manual for common error codes. Contact support if issues persist."
    ])

    # 2. Initialize SelfRAG LLM
    selfrag_llm = SelfRAG_LLM(knowledge_base=kb)

    # 3. Simulate Customer Interactions
    queries = [
        "What is your return policy?",
        "How much is shipping to California?",
        "My Product X is not turning on, what should I do?",
        "Tell me about the warranty for the headphones.",
        "Hello",
        "How do I return a product?"
    ]

    for query in queries:
        final_answer, log = selfrag_llm.respond_to_customer(query)
        print(f"Final Chatbot Response: {final_answer}")
        print("--- Conversation Log ---")
        for entry in log:
            print(entry)
        print("===========================================")
