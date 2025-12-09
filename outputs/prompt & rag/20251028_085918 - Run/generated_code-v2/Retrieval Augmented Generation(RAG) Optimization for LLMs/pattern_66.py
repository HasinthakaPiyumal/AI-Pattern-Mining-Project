import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RAGModel:
    def __init__(self, generator_model_name="t5-small", document_store=None):
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        self.document_store = document_store if document_store is not None else []

    def retrieve_documents(self, query, top_k=2):
        # Simplified document retrieval: in a real scenario, this would involve
        # a vector database (e.g., FAISS, Chroma, Pinecone) and embeddings.
        # Here, we just return a subset of documents based on a very naive keyword match
        # or simply return the first few for demonstration.
        print(f"Retrieving documents for query: '{query}'")
        retrieved = []
        query_lower = query.lower()
        for i, doc in enumerate(self.document_store):
            if query_lower in doc.lower():
                retrieved.append(doc)
            if len(retrieved) >= top_k:
                break
        if not retrieved and self.document_store: # Fallback if no keyword match
            retrieved = self.document_store[:top_k]
        return retrieved

    def generate_response(self, query, context_document, max_length=50):
        input_text = f"question: {query} context: {context_document}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.generator.generate(**inputs, max_new_tokens=max_length, do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_log_likelihood(self, query, context_document, hypothesis):
        # This function simulates getting the log-likelihood of a hypothesis given query and context.
        # In a full RAG model, this would involve calculating the probability of each token
        # in the hypothesis. For this conceptual example, we'll assign a placeholder.
        input_text = f"question: {query} context: {context_document}"
        target_text = hypothesis

        input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
        labels = self.tokenizer(target_text, return_tensors="pt", truncation=True, max_length=512).input_ids

        # Shift input_ids for decoder attention
        decoder_input_ids = self.generator._shift_right(labels)

        with torch.no_grad():
            outputs = self.generator(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

        # Calculate log probabilities of the target tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)

        # Sum log probabilities for the entire hypothesis
        # Mask out padding tokens if any
        log_likelihood = target_log_probs.sum(dim=-1).item()

        return log_likelihood

class FastRAGSequenceDecoder:
    def __init__(self, rag_model, num_beams=5, max_new_tokens=50):
        self.rag_model = rag_model
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def _initial_beam_search(self, query, context_document):
        input_text = f"question: {query} context: {context_document}"
        inputs = self.rag_model.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Perform beam search and get sequences and their scores
        outputs = self.rag_model.generator.generate(
            **inputs,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            num_return_sequences=self.num_beams # Return all beams
        )
        
        hypotheses = []
        for seq in outputs.sequences:
            hypotheses.append(self.rag_model.tokenizer.decode(seq, skip_special_tokens=True))
            
        # In a real scenario, you'd also get scores associated with these beams.
        # For this approximation, we just need the set of generated hypotheses.
        return set(hypotheses)

    def decode(self, query):
        retrieved_docs = self.rag_model.retrieve_documents(query)
        if not retrieved_docs:
            return "I'm sorry, I couldn't find relevant information."

        candidate_hypotheses_per_doc = {}
        all_candidate_hypotheses = set()

        # Step 1: Generate candidate set Y from initial beam searches for each document
        print("Generating initial candidate hypotheses...")
        for doc_id, doc in enumerate(retrieved_docs):
            hypotheses = self._initial_beam_search(query, doc)
            candidate_hypotheses_per_doc[doc_id] = hypotheses
            all_candidate_hypotheses.update(hypotheses)
        
        if not all_candidate_hypotheses:
            return "I couldn't generate any valid hypotheses."

        # Step 2: Calculate approximate p(y|x, z) for all candidates and documents
        # and then combine to find the best overall hypothesis.
        
        # Store (log_p_y_given_x_z) for each (hypothesis, document)
        hypothesis_doc_log_probs = {}
        
        print("Calculating approximate log-likelihoods...")
        for hyp in all_candidate_hypotheses:
            for doc_id, doc in enumerate(retrieved_docs):
                if hyp in candidate_hypotheses_per_doc[doc_id]:
                    # Hypothesis was generated during beam search for this document
                    # Calculate actual log-likelihood (or a good approximation)
                    log_prob = self.rag_model.get_log_likelihood(query, doc, hyp)
                else:
                    # Approximation: p(y|x, z) = 0 if y was not generated from beam search for z
                    log_prob = -float('inf') # log(0) is -infinity
                hypothesis_doc_log_probs[(hyp, doc_id)] = log_prob
        
        # Step 3: Combine probabilities p(y|x) = sum_z p(y|x, z) * p(z|x)
        # For simplicity, we'll assume p(z|x) is uniform across retrieved docs for now,
        # or implicitly handled by averaging/summing log-likelihoods.
        # A more rigorous approach would estimate p(z|x) (e.g., from retriever scores).
        
        best_hypothesis = None
        max_total_log_prob = -float('inf')

        for hyp in all_candidate_hypotheses:
            total_log_prob_for_hyp = -float('inf') # Initialize to a very small number for sum of logs
            
            for doc_id, doc in enumerate(retrieved_docs):
                log_p_y_z = hypothesis_doc_log_probs.get((hyp, doc_id), -float('inf'))
                
                # Summing log probabilities (equivalent to multiplying probabilities)
                # log(A + B) is not logA + logB. We need to do log(sum(exp(log_p_y_z)))
                # For practical purposes and to avoid exp underflow, we can use logsumexp or approximate.
                # Here, we'll simplify and take the max log_p_y_z for a given y across all z as a heuristic for fast decoding, 
                # assuming the highest probability path dominates. 
                # A more correct approach would involve sum_z p(y|x,z)p(z|x) but this is simpler for demonstration.
                if log_p_y_z > -float('inf'): # Only consider if not 0 probability
                    if total_log_prob_for_hyp == -float('inf'):
                        total_log_prob_for_hyp = log_p_y_z
                    else:
                        total_log_prob_for_hyp = torch.logaddexp(torch.tensor(total_log_prob_for_hyp), torch.tensor(log_p_y_z)).item()
            
            if total_log_prob_for_hyp > max_total_log_prob:
                max_total_log_prob = total_log_prob_for_hyp
                best_hypothesis = hyp

        return best_hypothesis if best_hypothesis else "I'm sorry, I couldn't generate a good response."


# --- Chatbot Application --- 

if __name__ == "__main__":
    # Sample E-commerce document store
    ecommerce_docs = [
        "Our return policy allows returns within 30 days of purchase with the original receipt. Items must be unused and in their original packaging.",
        "Shipping usually takes 5-7 business days for standard delivery within the country. Expedited shipping options are available at checkout.",
        "You can track your order using the tracking number provided in your shipping confirmation email on our website's 'Order Tracking' page.",
        "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay for all online purchases.",
        "To reset your password, click on 'Forgot Password' on the login page and follow the instructions sent to your email.",
        "Our customer support is available Monday to Friday, 9 AM to 6 PM EST. You can reach us via live chat or email.",
        "Product XYZ is currently out of stock but is expected to be restocked next month.",
        "Promotional code SAVE15 gives 15% off on all orders over $50. This offer is valid until the end of the month."
    ]

    print("Initializing RAG model and Fast Decoder...")
    rag_model = RAGModel(document_store=ecommerce_docs)
    fast_decoder = FastRAGSequenceDecoder(rag_model)
    print("Chatbot ready! Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        print("Chatbot: Thinking...")
        response = fast_decoder.decode(user_query)
        print(f"Chatbot: {response}")

