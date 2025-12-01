"""Module for defining prompt templates for the ICDSS LLM agent."""

# Prompt for guiding the LLM to identify and score relevant relations
RELATION_PRUNE_PROMPT = """
Given the patient's symptoms: {symptoms}
And the current exploration path ending in entity: {current_entity}

Evaluate and score the relevance of the following candidate relations (on a scale of 1 to 5, where 5 is highly relevant) to finding a rare disease diagnosis.
Also, provide a brief justification for each score.

Candidate Relations:
{candidate_relations}

Format your response as a list of dictionaries, each with 'relation', 'score', and 'justification'.
Example: [
  {{"relation": "relation_1", "score": 4, "justification": "Reason for score 4"}},
  {{"relation": "relation_2", "score": 2, "justification": "Reason for score 2"}}
]
"""

# Prompt for directing the LLM to score the contribution of candidate entities
ENTITY_PRUNE_PROMPT = """
Considering the patient's profile:
Symptoms: {symptoms}
Medical History: {medical_history}

And the highly relevant relations identified:
{relevant_relations}

Evaluate and score the potential contribution of these candidate entities to forming a coherent rare disease diagnostic hypothesis (on a scale of 1 to 5, where 5 is highly contributory).
Also, provide a brief justification for each score.

Candidate Entities:
{candidate_entities}

Format your response as a list of dictionaries, each with 'entity', 'score', and 'justification'.
Example: [
  {{"entity": "entity_A", "score": 5, "justification": "Reason for score 5"}},
  {{"entity": "entity_B", "score": 3, "justification": "Reason for score 3"}}
]
"""

# Prompt for asking the LLM to evaluate the sufficiency of current reasoning paths
REASONING_PROMPT = """
Based on the explored knowledge paths and identified entities regarding patient ID: {patient_id}
Symptoms: {symptoms}

And the current set of potential diagnostic paths and evidence:
{current_reasoning_paths}

Do you have sufficient information to propose a confident rare disease diagnosis? (Respond with 'yes' or 'no').
If 'no', what further information or exploration is needed from the Knowledge Graph to strengthen the diagnostic hypothesis or rule out alternatives?

Format your response as a JSON object with 'sufficient' (boolean) and 'needed_info' (string, if not sufficient).
Example: {{
  "sufficient": true,
  "needed_info": ""
}}
OR
Example: {{
  "sufficient": false,
  "needed_info": "Need to explore gene mutations related to entity_X."
}}
"""

# Prompt for instructing the LLM to synthesize the final answer
GENERATE_PROMPT = """
Synthesize a comprehensive report for patient ID: {patient_id}
Symptoms: {symptoms}
Medical History: {medical_history}

Based on all explored information, including the following key reasoning paths and entities:
{final_reasoning_paths}

Provide a prioritized list of potential rare disease diagnoses. For each diagnosis, include:
1. The disease name.
2. A confidence score (0-100%).
3. Strong supporting evidence from the Knowledge Graph (e.g., connections to symptoms, genes, other diseases).
4. The reasoning steps that led to this diagnosis.
5. Suggestions for next steps for clinical validation (e.g., specific genetic tests, specialist consultations).

Format your response as a JSON object with a list of 'diagnoses'.
Example: {{
  "diagnoses": [
    {{
      "disease": "Disease_X",
      "confidence": 90,
      "evidence": "Linked to symptom_A via relation_R1, and gene_G1 via relation_R2.",
      "reasoning": "Step-by-step reasoning.",
      "next_steps": "Recommend genetic sequencing for G1."
    }}
  ]
}}
"""