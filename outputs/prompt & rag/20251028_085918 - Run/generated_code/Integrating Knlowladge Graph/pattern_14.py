"""
This script implements a simplified Healthcare Knowledge Assistant that integrates a Large Language Model (LLM)
with a mock Knowledge Graph (KG) to answer medical queries. It demonstrates the core concepts of the
proposed architecture, including LLM as an agent, knowledge augmentation (RAG), prompt engineering,
and simulated specialized functions like entity extraction and semantic parsing.

Note: This is a highly simplified, in-memory implementation for demonstration purposes.
Actual production systems would involve robust graph databases, advanced NLP models, and commercial LLM APIs.
"""

import collections
import datetime

# --- 1. Mock Knowledge Graph (KG) Management Module ---

class KnowledgeGraphManager:
    """Simulates a medical Knowledge Graph and its querying mechanisms."""

    def __init__(self):
        # A simple, in-memory KG represented as a list of (subject, predicate, object) triples
        self.kg_data = [
            ("Aspirin", "has_side_effect", "Nausea"),
            ("Aspirin", "treats", "Headache"),
            ("Aspirin", "treats", "Fever"),
            ("Nausea", "is_symptom_of", "Migraine"),
            ("Nausea", "is_symptom_of", "Stomach Flu"),
            ("Headache", "is_symptom_of", "Migraine"),
            ("Headache", "is_symptom_of", "Stress"),
            ("Migraine", "has_treatment", "Sumatriptan"),
            ("Sumatriptan", "has_side_effect", "Dizziness"),
            ("Patient A", "has_condition", "Migraine"),
            ("Patient A", "is_prescribed", "Sumatriptan"),
            ("Tylenol", "treats", "Headache"),
            ("Tylenol", "treats", "Fever"),
            ("Diabetes", "has_symptom", "Increased Thirst"),
            ("Diabetes", "has_treatment", "Metformin"),
            ("Metformin", "has_side_effect", "Diarrhea"),
            ("Hypertension", "has_treatment", "Lisinopril"),
            ("Lisinopril", "has_side_effect", "Cough"),
            ("Common Cold", "has_symptom", "Runny Nose"),
            ("Common Cold", "has_symptom", "Sore Throat"),
            ("Common Cold", "has_treatment", "Rest"),
            ("Common Cold", "has_treatment", "Fluids"),
        ]

    def query_kg(self, query_entities: list, query_type: str = "related") -> list:
        """Simulates querying the KG for facts related to given entities.
        `query_type` can be 'related', 'side_effects', 'treatments', 'symptoms'.
        """
        results = []
        query_entities_lower = [e.lower() for e in query_entities]

        for s, p, o in self.kg_data:
            s_lower, p_lower, o_lower = s.lower(), p.lower(), o.lower()

            if query_type == "side_effects":
                if p_lower == "has_side_effect" and any(ent in s_lower for ent in query_entities_lower):
                    results.append((s, p, o))
            elif query_type == "treatments":
                if p_lower == "has_treatment" and any(ent in s_lower for ent in query_entities_lower):
                    results.append((s, p, o))
                elif p_lower == "treats" and any(ent in o_lower for ent in query_entities_lower):
                    results.append((s, p, o))
            elif query_type == "symptoms":
                if p_lower == "has_symptom" and any(ent in s_lower for ent in query_entities_lower):
                    results.append((s, p, o))
                elif p_lower == "is_symptom_of" and any(ent in o_lower for ent in query_entities_lower):
                    results.append((s, p, o))
            else: # "related" or general query
                if any(ent in s_lower or ent in o_lower for ent in query_entities_lower):
                    results.append((s, p, o))
        return list(set(results)) # Remove duplicates


# --- 2. Mock LLM Integration ---

class MockLLM:
    """A mock LLM that provides predefined or simple rule-based responses."""

    def __init__(self, model_name="MockMedicalLLM"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """Generates a mock response based on the prompt content."""
        prompt_lower = prompt.lower()
        if "side effect of aspirin" in prompt_lower:
            return "Based on the knowledge graph, common side effects of Aspirin include Nausea."
        elif "treats migraine" in prompt_lower or "treatment for migraine" in prompt_lower:
            return "Migraine can be treated with Sumatriptan. Lifestyle adjustments like managing stress can also help."
        elif "symptoms of diabetes" in prompt_lower:
            return "The knowledge graph indicates that a symptom of Diabetes is Increased Thirst."
        elif "differential diagnosis for headache" in prompt_lower:
            return "A headache can be a symptom of Migraine or Stress. Further investigation is recommended."
        elif "drug x with condition y" in prompt_lower:
            return "Please specify the drug and condition for a more precise interaction check. However, in general, side effects and interactions are crucial to consider."
        elif "drug interactions" in prompt_lower:
            return "I need specific drugs to check for interactions. Please provide more details."
        elif "no relevant information" in prompt_lower:
            return "I currently don't have enough specific information in my knowledge graph to answer that. Could you rephrase or provide more context?"
        else:
            return f"As a medical assistant, I've processed your query. My current understanding is limited to the provided KG context. Based on your prompt, here's a general thought: {prompt[:100]}..."


# --- 3. LLM Orchestration & Reasoning Module ---

class LLMOrchestrator:
    """Manages the interaction between the LLM and the KG for structured reasoning."""

    def __init__(self, kg_manager: KnowledgeGraphManager, llm: MockLLM):
        self.kg_manager = kg_manager
        self.llm = llm
        self.monitoring_logger = MonitoringLogger()

    def extract_topic_entities(self, query: str) -> list:
        """Simulates topic entity extraction (e.g., using spaCy or a simple keyword matcher)."""
        entities = []
        medical_terms = [
            "Aspirin", "Nausea", "Headache", "Fever", "Migraine", "Sumatriptan",
            "Diabetes", "Increased Thirst", "Metformin", "Hypertension", "Lisinopril",
            "Cough", "Common Cold", "Runny Nose", "Sore Throat", "Stress"
        ]
        query_lower = query.lower()
        for term in medical_terms:
            if term.lower() in query_lower:
                entities.append(term)
        return list(set(entities))

    def semantic_parse_query(self, query: str, entities: list) -> dict:
        """Simulates converting natural language to a structured KG query (e.g., SPARQL/Cypher-like).
           Returns a dict representing the query intent and parameters.
        """
        query_lower = query.lower()
        parsed_query = {"type": "general", "entities": entities}

        if "side effect" in query_lower and entities:
            parsed_query["type"] = "side_effects"
        elif "treat" in query_lower or "therapy" in query_lower or "medication" in query_lower:
            parsed_query["type"] = "treatments"
        elif "symptom" in query_lower and entities:
            parsed_query["type"] = "symptoms"
        elif "differential diagnosis" in query_lower or "what causes" in query_lower:
            parsed_query["type"] = "symptoms" # To find conditions associated with symptoms

        return parsed_query

    def retrieve_kg_facts(self, parsed_query: dict) -> list:
        """Retrieves relevant facts from the KG based on the parsed query."""
        entities = parsed_query.get("entities", [])
        query_type = parsed_query.get("type", "related")
        if not entities and query_type != "general": # For broad queries without specific entities initially
             return [] # Or implement a broader KG search

        facts = self.kg_manager.query_kg(entities, query_type)
        return facts

    def apply_hybrid_pruning(self, kg_facts: list, user_query: str) -> list:
        """Simulates a hybrid pruning strategy to filter/rank KG facts.
           Here, it's a simple relevance filter based on keywords in the user query.
        """
        if not kg_facts: return []

        pruned_facts = []
        query_lower = user_query.lower()
        for s, p, o in kg_facts:
            if any(keyword in s.lower() or keyword in o.lower() for keyword in query_lower.split()):
                pruned_facts.append((s, p, o))

        # If pruning is too aggressive, revert to original facts (simple heuristic)
        if not pruned_facts and kg_facts:
            return kg_facts
        return pruned_facts

    def format_prompt(self, user_query: str, kg_facts: list, previous_thoughts: list = None) -> str:
        """Formats the user query and retrieved KG facts into a prompt for the LLM.
           Uses Triple-Based Path Representation.
        """
        context_facts = """Knowledge Graph Context:
"""
        if kg_facts:
            for s, p, o in kg_facts:
                context_facts += f"- [{s}] --{p.replace('_', ' ')}--> [{o}]\n"
        else:
            context_facts += "- No directly relevant facts found in the knowledge graph for initial retrieval.\n"

        if previous_thoughts:
            thought_process = """Previous Reasoning Steps:\n""" + "\n".join(previous_thoughts)
        else:
            thought_process = ""

        prompt = f"""{context_facts}
{thought_process}

Medical Professional's Query: {user_query}

Based on the provided Knowledge Graph context and your medical understanding, please provide an accurate and explainable answer. If information is insufficient, state that. Focus on patient safety and evidence-based reasoning.
Response:"""
        return prompt

    def perform_iterative_reasoning(self, user_query: str, max_iterations: int = 3, beam_width: int = 2) -> str:
        """Orchestrates the LLM as an agent, performing iterative reasoning over the KG.
           Simulates beam search by exploring a few promising paths.
        """
        self.monitoring_logger.log_interaction(f"User Query: {user_query}")

        current_thought_paths = collections.deque([([], user_query)]) # (list_of_thoughts, current_query_context)
        final_responses = []

        for i in range(max_iterations):
            next_thought_paths = collections.deque()
            self.monitoring_logger.log_interaction(f"--- Iteration {i+1} ---")

            if not current_thought_paths: # No more paths to explore
                break

            for current_thoughts, current_query_context in list(current_thought_paths):
                self.monitoring_logger.log_interaction(f"  Exploring path with context: {current_query_context}")

                # Step 1: Topic Entity Extraction
                entities = self.extract_topic_entities(current_query_context)
                self.monitoring_logger.log_interaction(f"    Extracted entities: {entities}")

                # Step 2: Semantic Parsing
                parsed_query = self.semantic_parse_query(current_query_context, entities)
                self.monitoring_logger.log_interaction(f"    Parsed query: {parsed_query}")

                # Step 3: Knowledge Retrieval (RAG)
                kg_facts = self.retrieve_kg_facts(parsed_query)
                self.monitoring_logger.log_interaction(f"    Retrieved {len(kg_facts)} KG facts.")

                # Step 4: Hybrid Pruning
                pruned_facts = self.apply_hybrid_pruning(kg_facts, user_query)
                self.monitoring_logger.log_interaction(f"    Pruned to {len(pruned_facts)} facts.")

                # Step 5: Prompt Engineering
                prompt = self.format_prompt(user_query, pruned_facts, current_thoughts)
                self.monitoring_logger.log_interaction(f"    Sending prompt to LLM (first 200 chars): {prompt[:200]}...")

                # Step 6: LLM Generation
                llm_response = self.llm.generate_response(prompt)
                self.monitoring_logger.log_interaction(f"    LLM Raw Response: {llm_response}")

                # Simulate LLM deciding to refine or conclude (simplified beam search)
                if "more context" in llm_response.lower() and i < max_iterations - 1:
                    new_thought = f"Thought {i+1}: LLM requested more context or refinement. Exploring deeper.\nLLM said: {llm_response}"
                    next_query_context = f"{user_query}. Given: {llm_response}" # Refine context for next iteration
                    new_thoughts_path = current_thoughts + [new_thought]
                    next_thought_paths.append((new_thoughts_path, next_query_context))
                    self.monitoring_logger.log_interaction(f"    LLM requested refinement. Adding new path for iteration {i+2}.")
                else:
                    # This path concludes or is a final answer candidate
                    final_responses.append({
                        "response": llm_response,
                        "kg_facts_used": pruned_facts,
                        "reasoning_steps": current_thoughts + [f"Thought {i+1}: Final answer generation based on available context and LLM response."]
                    })
                    self.monitoring_logger.log_interaction(f"    LLM concluded for this path.")

            # Select top 'beam_width' paths for the next iteration (simplified)
            # In a real beam search, you'd score paths. Here, we just take the first 'beam_width'
            current_thought_paths = collections.deque(list(next_thought_paths)[:beam_width])

        if final_responses:
            # For simplicity, return the first successful final response or aggregate them.
            # In a real system, you might rank or combine responses.
            best_response = final_responses[0]
            explanation = "\n".join(best_response["reasoning_steps"])
            explanation += "\n\nKG Facts Used:\n" + "\n".join([f"- [{s}] --{p.replace('_', ' ')}--> [{o}]" for s,p,o in best_response["kg_facts_used"]])
            return f"**Final Answer:** {best_response['response']}\n\n**Explanation:**\n{explanation}"
        else:
            return "The assistant could not find a conclusive answer based on the available knowledge graph and reasoning steps."


# --- 4. Monitoring & Logging Module ---

class MonitoringLogger:
    """Basic logger for monitoring interactions and reasoning paths."""

    def log_interaction(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LOGGER][{timestamp}] {message}")


# --- 5. Frontend Simulator (Main Application Logic) ---

def run_healthcare_assistant():
    """Simulates the frontend interaction with the Healthcare Knowledge Assistant."""
    print("\n--- Healthcare Knowledge Assistant (Simulated) ---")
    print("Type 'exit' or 'quit' to end the session.")

    kg_manager = KnowledgeGraphManager()
    llm = MockLLM()
    orchestrator = LLMOrchestrator(kg_manager, llm)

    while True:
        user_input = input("\nMedical Professional (You): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Healthcare Knowledge Assistant. Goodbye!")
            break

        response = orchestrator.perform_iterative_reasoning(user_input)
        print(f"\nAI Assistant: {response}")

if __name__ == "__main__":
    run_healthcare_assistant()