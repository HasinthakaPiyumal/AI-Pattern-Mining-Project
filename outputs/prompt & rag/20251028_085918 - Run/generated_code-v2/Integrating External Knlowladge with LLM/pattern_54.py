"""
clinical_research_assistant.py

This script implements an LLM-Guided Beam Search for Knowledge Graph (KG) Exploration,
simulating a Clinical Research Assistant. It helps identify complex multi-hop reasoning paths
between medical entities like diseases, genes, drugs, and symptoms within a medical KG.

The core idea is to use an LLM (simulated here) to intelligently guide a beam search
algorithm in exploring the KG. The LLM performs a two-step exploration:
1. Search: Formal queries are executed on the KG to retrieve all candidate neighboring
   relations and entities for the current set of top-N reasoning paths.
2. Prune: The LLM evaluates these candidates based on the input question and current
   path context, selecting the top-N most relevant relations or entities to extend
   the beam and form the next set of promising reasoning paths.
"""

import collections

class MedicalKnowledgeGraph:
    """
    A simplified in-memory representation of a Medical Knowledge Graph.
    Entities: diseases, genes, drugs, symptoms.
    Relationships are directed.
    """
    def __init__(self):
        # Graph represented as an adjacency list: entity -> [(relation, target_entity)]
        self.graph = collections.defaultdict(list)
        self._populate_sample_data()

    def _populate_sample_data(self):
        # Diseases
        self.add_entity("Disease:Diabetes Type 2")
        self.add_entity("Disease:Hypertension")
        self.add_entity("Disease:Alzheimer's")
        self.add_entity("Disease:Cystic Fibrosis")

        # Genes
        self.add_entity("Gene:APOE")
        self.add_entity("Gene:CFTR")
        self.add_entity("Gene:PPARG")
        self.add_entity("Gene:ACE")

        # Drugs
        self.add_entity("Drug:Metformin")
        self.add_entity("Drug:Lisinopril")
        self.add_entity("Drug:Donepezil")
        self.add_entity("Drug:Ivacaftor")

        # Symptoms
        self.add_entity("Symptom:High Blood Sugar")
        self.add_entity("Symptom:High Blood Pressure")
        self.add_entity("Symptom:Memory Loss")
        self.add_entity("Symptom:Fatigue")
        self.add_entity("Symptom:Chronic Cough")

        # Relationships
        self.add_relation("Disease:Diabetes Type 2", "causes", "Symptom:High Blood Sugar")
        self.add_relation("Symptom:High Blood Sugar", "associated_with", "Gene:PPARG")
        self.add_relation("Gene:PPARG", "targets", "Drug:Metformin")
        self.add_relation("Drug:Metformin", "treats", "Disease:Diabetes Type 2")

        self.add_relation("Disease:Hypertension", "causes", "Symptom:High Blood Pressure")
        self.add_relation("Symptom:High Blood Pressure", "associated_with", "Gene:ACE")
        self.add_relation("Gene:ACE", "targets", "Drug:Lisinopril")
        self.add_relation("Drug:Lisinopril", "treats", "Disease:Hypertension")

        self.add_relation("Disease:Alzheimer's", "causes", "Symptom:Memory Loss")
        self.add_relation("Symptom:Memory Loss", "associated_with", "Gene:APOE")
        self.add_relation("Gene:APOE", "targets", "Drug:Donepezil") # Simplified
        self.add_relation("Drug:Donepezil", "treats", "Disease:Alzheimer's")

        self.add_relation("Disease:Cystic Fibrosis", "causes", "Symptom:Chronic Cough")
        self.add_relation("Symptom:Chronic Cough", "associated_with", "Gene:CFTR")
        self.add_relation("Gene:CFTR", "targets", "Drug:Ivacaftor")
        self.add_relation("Drug:Ivacaftor", "treats", "Disease:Cystic Fibrosis")

        # Cross-connections / more complex paths
        self.add_relation("Disease:Diabetes Type 2", "comorbidity_with", "Disease:Hypertension")
        self.add_relation("Disease:Hypertension", "risk_factor_for", "Disease:Alzheimer's")
        self.add_relation("Gene:PPARG", "interacts_with", "Gene:ACE") # Example interaction

    def add_entity(self, entity_name):
        # Ensure the entity exists in the graph keys
        if entity_name not in self.graph:
            self.graph[entity_name] = []

    def add_relation(self, source, relation, target):
        self.add_entity(source) # Ensure source exists
        self.add_entity(target) # Ensure target exists
        self.graph[source].append((relation, target))

    def get_neighbors(self, entity):
        """
        Retrieves all direct neighbors (relations and target entities) for a given entity.
        Returns a list of tuples: [(relation, target_entity), ...]
        """
        return self.graph.get(entity, [])

class LLM_Agent:
    """
    A simulated LLM agent responsible for pruning candidate paths.
    In a real application, this would involve API calls to a large language model.
    Here, it uses simple heuristic-based relevance scoring.
    """
    def prune_candidates(self, question: str, current_path: list, candidates: list, top_n: int) -> list:
        """
        Evaluates candidate path extensions and selects the top_n most relevant ones.
        
        Args:
            question (str): The initial research question.
            current_path (list): The path constructed so far (list of (entity, relation) tuples).
            candidates (list): List of potential (relation, next_entity) tuples to extend the path.
            top_n (int): The number of top candidates to select.

        Returns:
            list: The top_n most relevant (relation, next_entity) tuples.
        """
        if not candidates:
            return []

        # Simple heuristic for demonstration: prioritize candidates that contain keywords
        # from the question or entities already in the path.
        scores = []
        q_keywords = set(word.lower() for word in question.split() if len(word) > 2)
        current_path_entities = set([item[0] for item in current_path] + [current_path[-1][1]] if current_path else [])

        for relation, next_entity in candidates:
            score = 0
            # Boost if entity/relation name contains question keywords
            if any(k in next_entity.lower() for k in q_keywords):
                score += 1.5
            if any(k in relation.lower() for k in q_keywords):
                score += 1.0
            
            # Boost if the new entity is semantically similar to existing path entities (conceptual)
            # For this simulation, we'll just check if type is relevant
            if "disease" in question.lower() and "disease" in next_entity.lower():
                score += 0.8
            elif "gene" in question.lower() and "gene" in next_entity.lower():
                score += 0.8
            elif "drug" in question.lower() and "drug" in next_entity.lower():
                score += 0.8
            elif "symptom" in question.lower() and "symptom" in next_entity.lower():
                score += 0.8

            # Deduct if the entity is already in the current path to prevent cycles (simple check)
            if next_entity in current_path_entities:
                score -= 2.0 # Significantly penalize cycles

            scores.append((score, (relation, next_entity)))
        
        # Sort by score in descending order and take the top_n
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_n]]

class LLM_Guided_BeamSearch:
    """
    Orchestrates the LLM-Guided Beam Search on the Knowledge Graph.
    """
    def __init__(self, kg: MedicalKnowledgeGraph, llm_agent: LLM_Agent):
        self.kg = kg
        self.llm_agent = llm_agent

    def search(self, start_entity: str, question: str, beam_width: int, max_hops: int) -> list:
        """
        Performs an LLM-guided beam search to find multi-hop reasoning paths.
        
        Args:
            start_entity (str): The starting entity for the search (e.g., "Disease:Diabetes Type 2").
            question (str): The research question guiding the search.
            beam_width (int): The number of top paths to keep at each step.
            max_hops (int): The maximum number of hops to explore.

        Returns:
            list: A list of discovered reasoning paths. Each path is a list of
                  (entity, relation) tuples, ending with the final entity.
        """
        if start_entity not in self.kg.graph:
            print(f"Error: Start entity '{start_entity}' not found in the Knowledge Graph.")
            return []

        # Beam stores paths: [((entity1, relation1), (entity2, relation2), ..., (last_entity, None))]
        # Initialize beam with the start entity. Paths are (source_entity, relation_to_target, target_entity)
        # For the start entity, we can represent it as (start_entity, None, start_entity)
        # to align with (entity, relation) tuples later, but simpler to start with just the entity
        # and build paths as [(start_entity, None)] then [(entity1, relation1), (entity2, relation2)]

        # A path is represented as a list of tuples: [('EntityA', 'relation_to_B'), ('EntityB', 'relation_to_C'), ..., ('EntityN', None)]
        # The last element 'None' signifies the end of the path at EntityN.
        # Let's adjust: a path is a list of (relation, entity) where the first item is (None, start_entity)
        # and subsequent are (relation_from_prev_entity, current_entity)
        
        # A path will be a list of entities and relations traversed:
        # [start_entity, (relation1, entity1), (relation2, entity2), ...]
        # This makes it easier to reconstruct the path.

        # Initial beam: Each element is a path, which is a list of (relation, entity) tuples.
        # For the start node, it's (None, start_entity)
        beam = [[(None, start_entity)]]
        all_discovered_paths = []

        print(f"\nStarting LLM-Guided Beam Search from '{start_entity}' for question: '{question}'")

        for hop in range(max_hops):
            print(f"\n--- Hop {hop + 1}/{max_hops} ---")
            new_candidates = [] # Stores potential new paths
            
            if not beam:
                print("Beam is empty, stopping search.")
                break

            for current_path in beam:
                last_entity_in_path = current_path[-1][1] # Get the entity at the end of the current path
                
                # Step 1: Search - Query KG for neighbors of the last entity
                neighbors = self.kg.get_neighbors(last_entity_in_path)
                
                print(f"  Exploring from: {' -> '.join([e for r,e in current_path])}")
                if not neighbors:
                    print(f"    No neighbors found for '{last_entity_in_path}'.")

                for relation, next_entity in neighbors:
                    new_candidates.append((current_path, relation, next_entity))
            
            if not new_candidates:
                print("No new candidates found to extend paths.")
                break

            # Step 2: Prune - LLM Agent evaluates and selects top candidates
            # We need to pass the *potential next steps* to the LLM agent for pruning
            # For the LLM Agent, we need current_path context and a list of (relation, next_entity) tuples
            
            # Transform candidates for LLM: (current_path, [(relation, next_entity), ...])
            # For simplicity, let's just pass all unique (relation, next_entity) from current iteration
            # and let LLM prune, then reconstruct paths.
            
            llm_input_candidates = []
            candidate_map = {}
            for path_prefix, rel, next_ent in new_candidates:
                candidate_tuple = (rel, next_ent)
                if candidate_tuple not in candidate_map:
                    candidate_map[candidate_tuple] = []
                candidate_map[candidate_tuple].append(path_prefix) # Store which path prefix leads to this candidate
                llm_input_candidates.append(candidate_tuple)
            
            # Remove duplicates for LLM processing efficiency
            unique_llm_input_candidates = list(set(llm_input_candidates))
            
            print(f"  LLM evaluating {len(unique_llm_input_candidates)} unique candidate extensions...")
            
            # Pass a representative current_path to LLM Agent. 
            # For a true LLM-as-Agent, it would consider each full current_path and its extensions.
            # For this simulation, we'll pass the *first* path from the beam as context for simplicity
            # if a candidate doesn't have its specific path context.

            # To ensure the LLM Agent gets context for each candidate, we'll iterate through beam and prune per path
            # Or, we can design the LLM agent to prune a global list of (current_path, relation, next_entity)
            # Let's go with the latter for a single LLM call per hop, which is more beam-search like.

            # LLM-Agent will return (relation, next_entity) tuples.
            # We need to map these back to original `(current_path, relation, next_entity)` for `new_beam` construction.
            
            # The `prune_candidates` expects `current_path` and `candidates` as `(relation, next_entity)` tuples.
            # To make it truly LLM-guided, the LLM needs to know the full context of *each* potential path.
            # For this simulation, we'll simplify and have the LLM prune the *unique* extensions, 
            # and then apply those to any path that can use them.
            # A more robust LLM-as-Agent would likely prune full candidate *paths*.

            # Let's refactor `new_candidates` to be `(full_path_so_far, new_relation, new_entity)`
            # And LLM_Agent prunes a list of these full candidate paths.
            
            candidate_full_paths = []
            for path_prefix, relation, next_entity in new_candidates:
                candidate_full_paths.append(path_prefix + [(relation, next_entity)])
            
            # LLM Agent needs to prune these full paths. Its input should reflect this.
            # Let's update `LLM_Agent.prune_candidates` to expect `list[list[tuple]]` for candidates.
            
            # NOTE: For simplicity, the current `LLM_Agent.prune_candidates` expects `(relation, next_entity)` candidates
            # and a single `current_path`. This is a simplification. 
            # A more accurate LLM-as-Agent would prune full candidate *paths*.
            # We'll use the simplified version for now, feeding the LLM unique (relation, next_entity) pairs
            # and a 'representative' path, then manually reconstructing.

            unique_next_steps = list(set([(rel, next_ent) for _, rel, next_ent in new_candidates]))
            
            # The current_path argument to LLM_Agent is a bit ambiguous for a global prune across beam.
            # For the purpose of this simulation, we can pass an empty path or the `start_entity` path as context,
            # or consider the 'question' as primary context.
            
            # Let's pass the question and *all* candidate next steps. The LLM then picks top-N next steps.
            selected_next_steps = self.llm_agent.prune_candidates(
                question=question,
                current_path=[], # Simplified: LLM only uses question and candidate (rel, entity)
                candidates=unique_next_steps,
                top_n=beam_width
            )

            print(f"  LLM selected {len(selected_next_steps)} extensions.")
            
            new_beam = []
            for path_prefix, relation, next_entity in new_candidates:
                if (relation, next_entity) in selected_next_steps:
                    extended_path = path_prefix + [(relation, next_entity)]
                    new_beam.append(extended_path)
                    
            # Sort new_beam to apply beam_width correctly if multiple paths lead to same good extension
            # For simplicity, if multiple current_paths extend with the same selected_next_step, they all form new paths.
            # We then take the top `beam_width` *distinct* paths (or based on some quality score if available).
            # Here, we'll just limit the overall size of the `new_beam` to `beam_width` if it exceeds.
            if len(new_beam) > beam_width:
                # This simple truncation might not be ideal if paths are of vastly different quality.
                # In a real system, LLM would assign scores, and we'd sort by those scores.
                # For now, we take a diverse set based on initial discovery order.
                new_beam = new_beam[:beam_width]
            
            beam = new_beam
            all_discovered_paths.extend(beam) # Collect all paths at each step

            if not beam:
                print("Beam became empty, stopping search.")
                break

        return self._format_paths(all_discovered_paths)

    def _format_paths(self, raw_paths: list) -> list:
        """
        Formats the raw path data into a more readable string representation.
        """
        formatted_paths = []
        for path in raw_paths:
            if not path: continue
            
            # Path starts with (None, start_entity)
            formatted_path_str = f"Path: {path[0][1]}"
            for i in range(1, len(path)):
                relation, entity = path[i]
                formatted_path_str += f" --[{relation}]--> {entity}"
            formatted_paths.append(formatted_path_str)
        return formatted_paths

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Knowledge Graph
    kg = MedicalKnowledgeGraph()
    print("Knowledge Graph initialized with sample data.")

    # 2. Initialize LLM Agent (simulated)
    llm_agent = LLM_Agent()
    print("Simulated LLM Agent initialized.")

    # 3. Initialize Beam Search Orchestrator
    beam_search_engine = LLM_Guided_BeamSearch(kg, llm_agent)
    print("LLM-Guided Beam Search engine initialized.")

    # Example 1: Find paths related to Diabetes and drugs
    question1 = "What drugs are related to Diabetes Type 2 through gene pathways?"
    start_entity1 = "Disease:Diabetes Type 2"
    beam_width1 = 3
    max_hops1 = 4
    
    results1 = beam_search_engine.search(start_entity1, question1, beam_width1, max_hops1)
    
    print("\n--- Search Results for Example 1 ---")
    if results1:
        for i, path in enumerate(results1):
            print(f"Path {i+1}: {path}")
    else:
        print("No paths found for this query.")

    print("\n" + "="*80 + "\n")

    # Example 2: Explore connections of a specific gene to diseases or symptoms
    question2 = "What diseases or symptoms are associated with the gene APOE?"
    start_entity2 = "Gene:APOE"
    beam_width2 = 2
    max_hops2 = 3

    results2 = beam_search_engine.search(start_entity2, question2, beam_width2, max_hops2)

    print("\n--- Search Results for Example 2 ---")
    if results2:
        for i, path in enumerate(results2):
            print(f"Path {i+1}: {path}")
    else:
        print("No paths found for this query.")

    print("\n" + "="*80 + "\n")

    # Example 3: Find drugs treating symptoms related to a disease
    question3 = "Find drugs that treat symptoms caused by Hypertension."
    start_entity3 = "Disease:Hypertension"
    beam_width3 = 2
    max_hops3 = 4

    results3 = beam_search_engine.search(start_entity3, question3, beam_width3, max_hops3)

    print("\n--- Search Results for Example 3 ---")
    if results3:
        for i, path in enumerate(results3):
            print(f"Path {i+1}: {path}")
    else:
        print("No paths found for this query.")
