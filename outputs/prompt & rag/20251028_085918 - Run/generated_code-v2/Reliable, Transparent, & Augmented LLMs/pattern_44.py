
import streamlit as st
from typing import List, Dict, Tuple

# --- Simulated Knowledge Graph (In-memory for demonstration) ---
class KnowledgeGraphManager:
    def __init__(self):
        self.facts: List[Tuple[str, str, str]] = [
            ("patient_A", "has_symptom", "fever"),
            ("patient_A", "has_symptom", "cough"),
            ("fever", "is_symptom_of", "flu"),
            ("cough", "is_symptom_of", "flu"),
            ("flu", "treatment", "rest_and_fluids"),
            ("flu", "can_cause", "fatigue"),
            ("patient_B", "has_symptom", "headache"),
            ("patient_B", "has_symptom", "stiff_neck"),
            ("headache", "is_symptom_of", "meningitis"),
            ("stiff_neck", "is_symptom_of", "meningitis"),
            ("meningitis", "treatment", "antibiotics"),
            ("antibiotics", "requires", "prescription"),
        ]

    def get_facts_related_to(self, entity: str) -> List[Tuple[str, str, str]]:
        """Retrieves facts from the KG related to a given entity."""
        relevant_facts = []
        for s, p, o in self.facts:
            if entity in (s, o):
                relevant_facts.append((s, p, o))
        return relevant_facts

    def update_fact(self, old_fact: Tuple[str, str, str], new_fact: Tuple[str, str, str]) -> bool:
        """Simulates updating a fact in the KG. In a real system, this would involve
        a more robust update mechanism and potentially expert review."""
        try:
            index = self.facts.index(old_fact)
            self.facts[index] = new_fact
            st.success(f"Fact updated successfully: {old_fact} -> {new_fact}")
            return True
        except ValueError:
            st.error(f"Fact {old_fact} not found in KG.")
            return False

    def add_fact(self, fact: Tuple[str, str, str]) -> None:
        """Simulates adding a new fact to the KG."""
        if fact not in self.facts:
            self.facts.append(fact)
            st.success(f"New fact added: {fact}")
        else:
            st.warning(f"Fact {fact} already exists.")


# --- Mock LLM for Diagnosis and Reasoning Path Generation ---
def mock_llm_diagnosis(query: str, kg_manager: KnowledgeGraphManager) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    A mock LLM function that generates a diagnosis and a simulated reasoning path.
    In a real scenario, this would involve a sophisticated LLM and KG interaction.
    """
    reasoning_path = []
    diagnosis = "Undetermined"

    query_lower = query.lower()

    if "fever and cough" in query_lower:
        diagnosis = "Possible Flu"
        reasoning_path.append(("patient_input", "has_symptom", "fever"))
        reasoning_path.append(("patient_input", "has_symptom", "cough"))
        reasoning_path.append(("fever", "is_symptom_of", "flu"))
        reasoning_path.append(("cough", "is_symptom_of", "flu"))
        reasoning_path.append(("flu", "treatment", "rest_and_fluids"))
    elif "headache and stiff neck" in query_lower:
        diagnosis = "Possible Meningitis"
        reasoning_path.append(("patient_input", "has_symptom", "headache"))
        reasoning_path.append(("patient_input", "has_symptom", "stiff_neck"))
        reasoning_path.append(("headache", "is_symptom_of", "meningitis"))
        reasoning_path.append(("stiff_neck", "is_symptom_of", "meningitis"))
        reasoning_path.append(("meningitis", "treatment", "antibiotics"))
    elif "tiredness" in query_lower and "flu" in query_lower:
        diagnosis = "Flu-related fatigue"
        reasoning_path.append(("patient_input", "has_symptom", "tiredness"))
        reasoning_path.append(("patient_input", "diagnosed_with", "flu"))
        reasoning_path.append(("flu", "can_cause", "fatigue"))
    else:
        diagnosis = "Further investigation needed. Unable to determine based on current knowledge."
        reasoning_path.append(("system", "status", "insufficient_data"))

    return diagnosis, reasoning_path


# --- Streamlit Application --- 
st.set_page_config(layout="wide", page_title="MedExplain AI: Clinical Decision Support")
st.title("MedExplain AI: Clinical Decision Support with Explainable Reasoning")
st.markdown("---")

# Initialize Knowledge Graph Manager (singleton pattern for Streamlit)
if "kg_manager" not in st.session_state:
    st.session_state.kg_manager = KnowledgeGraphManager()

kg_manager = st.session_state.kg_manager

st.header("Patient Query")
patient_query = st.text_area(
    "Enter patient symptoms or medical query (e.g., 'patient with fever and cough', 'headache and stiff neck'):",
    height=100,
)

if st.button("Get Diagnosis and Reasoning"):
    if patient_query:
        diagnosis, reasoning_path = mock_llm_diagnosis(patient_query, kg_manager)

        st.subheader("AI Diagnosis:")
        st.success(diagnosis)

        st.subheader("AI Reasoning Path (Knowledge Triples):")
        if reasoning_path:
            for i, triple in enumerate(reasoning_path):
                st.markdown(f"**Step {i+1}:** `{triple[0]}` → `{triple[1]}` → `{triple[2]}`")
        else:
            st.info("No specific reasoning path generated for this query.")

        st.session_state.current_reasoning_path = reasoning_path
    else:
        st.warning("Please enter a patient query.")

st.markdown("---")
st.header("Provide Feedback and Correct Knowledge Graph")
st.write("If you identify an incorrect or outdated fact in the reasoning path above, you can propose a correction.")

if "current_reasoning_path" in st.session_state and st.session_state.current_reasoning_path:
    selected_triple_str = st.selectbox(
        "Select the triple you wish to correct:",
        [' → '.join(triple) for triple in st.session_state.current_reasoning_path]
    )

    if selected_triple_str:
        # Convert selected string back to tuple
        selected_triple_parts = selected_triple_str.split(' → ')
        if len(selected_triple_parts) == 3:
            old_triple = tuple(selected_triple_parts)
            st.write(f"You selected: `{selected_triple_str}`")

            st.subheader("Propose Correction:")
            subject = st.text_input("Subject", old_triple[0], key="subject_input")
            predicate = st.text_input("Predicate", old_triple[1], key="predicate_input")
            object_val = st.text_input("Object", old_triple[2], key="object_input")

            new_triple = (subject, predicate, object_val)

            if st.button("Submit Correction"):
                if new_triple != old_triple:
                    kg_manager.update_fact(old_triple, new_triple)
                    st.info("Correction submitted for review. In a real system, this would go through an expert validation workflow.")
                else:
                    st.warning("The new fact is identical to the old one. No change submitted.")
            
            st.subheader("Add New Fact")
            st.write("You can also add an entirely new fact to the knowledge graph.")
            new_sub = st.text_input("New Subject", key="new_sub")
            new_pred = st.text_input("New Predicate", key="new_pred")
            new_obj = st.text_input("New Object", key="new_obj")

            if st.button("Add New Fact to KG"):
                if new_sub and new_pred and new_obj:
                    kg_manager.add_fact((new_sub, new_pred, new_obj))
                else:
                    st.warning("Please fill in all fields to add a new fact.")

else:
    st.info("Generate a diagnosis first to see reasoning paths and propose corrections.")

st.markdown("---")
st.subheader("Current State of Simulated Knowledge Graph (for debugging/demonstration)")
with st.expander("View all facts in KG"):
    for i, fact in enumerate(kg_manager.facts):
        st.markdown(f"{i+1}. `{fact[0]}` → `{fact[1]}` → `{fact[2]}`")
