import asyncio
import time
from fastapi import FastAPI
from typing import List, Dict, Any

app = FastAPI()

# --- Simulated Knowledge Retrieval Module ---
class KnowledgeRetriever:
    def __init__(self):
        self.documents = [
            "Doc 1: Information about product A features. Product A is known for its durability.",
            "Doc 2: Troubleshooting steps for common issues with product B. Refer to manual for error codes.",
            "Doc 3: Warranty details for all products. Standard warranty is 1 year, extended warranty available.",
            "Doc 4: Latest updates on product A's software. Version 2.1 improves performance.",
            "Doc 5: Customer service contact information. Available 24/7 via chat or phone.",
            "Doc 6: Technical specifications for product B. Includes dimensions and power requirements.",
            "Doc 7: How to reset product A to factory settings. A hard reset can fix many issues.",
            "Doc 8: Pricing plans for enterprise clients. Custom quotes available upon request.",
        ]

    async def retrieve_documents_staged(self, query: str, stage: int) -> List[str]:
        await asyncio.sleep(0.5)  # Simulate I/O latency for retrieval
        if stage == 1: # Initial broad retrieval
            if "product A" in query.lower():
                return [self.documents[0], self.documents[3], self.documents[7]]
            elif "product B" in query.lower():
                return [self.documents[1], self.documents[5], self.documents[7]]
            else:
                return [self.documents[2], self.documents[4], self.documents[7]]
        elif stage == 2: # More refined retrieval
            if "product A" in query.lower() and "troubleshoot" in query.lower():
                return [self.documents[0], self.documents[3], self.documents[7], self.documents[4]]
            elif "product A" in query.lower() and "reset" in query.lower():
                return [self.documents[0], self.documents[7], self.documents[4], self.documents[6]]
            elif "product B" in query.lower() and "troubleshoot" in query.lower():
                return [self.documents[1], self.documents[5], self.documents[7], self.documents[4]]
            else:
                return self.documents[2:5] # General info
        elif stage == 3: # Most refined retrieval
            if "product A" in query.lower() and "warranty" in query.lower():
                return [self.documents[0], self.documents[2], self.documents[4]]
            elif "product B" in query.lower() and "specs" in query.lower():
                return [self.documents[1], self.documents[5], self.documents[4]]
            else:
                return self.documents[0:6] # Broadest possible
        return []

# --- Simulated LLM Inference Module ---
class LLMSpeculativeGenerator:
    def __init__(self):
        self._current_task: asyncio.Task = None
        self._cancel_flag = asyncio.Event()

    async def generate_tokens(self, context: str, generation_id: str): # Added generation_id for tracking
        full_response = f"Based on the information: {context}. "
        tokens = full_response.split()
        generated_so_far = []
        for i, token in enumerate(tokens):
            if self._cancel_flag.is_set():
                self._cancel_flag.clear()
                print(f"\n--- Generation {generation_id} cancelled ---")
                return ""
            await asyncio.sleep(0.1)  # Simulate token generation latency
            generated_so_far.append(token)
            yield " ".join(generated_so_far) # Yield partial response
            if i == len(tokens) - 1:
                print(f"\n--- Generation {generation_id} completed ---")


    async def start_generation(self, context: str, generation_id: str):
        self._cancel_flag.clear() # Clear any previous cancellation
        self._current_task = asyncio.create_task(self._run_generation(context, generation_id))
        return self._current_task

    async def _run_generation(self, context: str, generation_id: str):
        generated_text = []
        async for token in self.generate_tokens(context, generation_id):
            generated_text.append(token)
        return generated_text[-1] if generated_text else ""

    def cancel_generation(self):
        if self._current_task and not self._current_task.done():
            self._cancel_flag.set()


# --- Chatbot Service and Orchestration ---
retriever = KnowledgeRetriever()
llm_generator = LLMSpeculativeGenerator()

LLM_LOAD_THRESHOLD = 1  # Simulate LLM load. 1 means always speculative for single user
llm_current_load = 0

@app.post("/chat")
async def chat_endpoint(query: Dict[str, str]):
    global llm_current_load
    user_query = query.get("query", "")
    print(f"\nReceived query: {user_query}")

    start_time = time.time()
    final_response = ""
    current_speculative_task: asyncio.Task = None
    last_used_docs = []
    current_generation_id = 0

    # Dynamic check for speculative pipelining
    if llm_current_load < LLM_LOAD_THRESHOLD:
        print("--- Speculative Pipelining ENABLED ---")
        llm_current_load += 1

        # Stage 1 Retrieval
        retrieval_stage_1_task = retriever.retrieve_documents_staged(user_query, 1)
        docs_stage_1 = await retrieval_stage_1_task
        last_used_docs = docs_stage_1
        print(f"Retrieval Stage 1 Docs: {len(docs_stage_1)} documents")

        # Start initial speculative generation
        current_generation_id += 1
        current_speculative_task = await llm_generator.start_generation(
            context=" ".join(docs_stage_1), generation_id=f"S1-{current_generation_id}"
        )

        # Concurrently perform subsequent retrieval stages
        retrieval_stage_2_task = asyncio.create_task(retriever.retrieve_documents_staged(user_query, 2))
        retrieval_stage_3_task = asyncio.create_task(retriever.retrieve_documents_staged(user_query, 3))

        for i, retrieval_task in enumerate([retrieval_stage_2_task, retrieval_stage_3_task]):
            docs_next_stage = await retrieval_task
            print(f"Retrieval Stage {i+2} Docs: {len(docs_next_stage)} documents")

            # Simple comparison for demonstration: check if document sets are exactly different
            # In a real system, this would involve embedding similarity or hash comparison.
            if set(docs_next_stage) != set(last_used_docs):
                print(f"Docs changed at Stage {i+2}. Cancelling previous speculation.")
                llm_generator.cancel_generation()
                await asyncio.sleep(0.01) # Give a tiny moment for cancellation to register
                
                last_used_docs = docs_next_stage
                current_generation_id += 1
                current_speculative_task = await llm_generator.start_generation(
                    context=" ".join(docs_next_stage), generation_id=f"S{i+2}-{current_generation_id}"
                )
            else:
                print(f"Docs similar at Stage {i+2}. Continuing current speculation.")

        # Await the final speculative generation or the one that wasn't cancelled
        if current_speculative_task:
            final_response = await current_speculative_task
        else:
            # Fallback if no task was ever started or all cancelled prematurely (edge case)
            final_response = "No speculative generation completed. Please try again."

        llm_current_load -= 1
    else:
        print("--- Speculative Pipelining DISABLED (LLM overloaded) ---")
        # Sequential RAG fallback
        docs = await retriever.retrieve_documents_staged(user_query, 3) # Go straight to full retrieval
        llm_current_load += 1
        current_generation_id += 1
        sequential_generation_task = await llm_generator.start_generation(
            context=" ".join(docs), generation_id=f"Sequential-{current_generation_id}"
        )
        final_response = await sequential_generation_task
        llm_current_load -= 1

    end_time = time.time()
    latency = end_time - start_time
    print(f"Response Latency: {latency:.2f} seconds")

    return {"query": user_query, "response": final_response, "latency": f"{latency:.2f}s"}

# To run this application:
# 1. Save the code as `main.py` (or any other name).
# 2. Install FastAPI and Uvicorn: `pip install fastapi "uvicorn[standard]"`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API at `http://127.0.0.1:8000/docs` for the Swagger UI.