def primary_llm_summarizer(paper_text: str) -> str:
    """Mocks a primary LLM that summarizes a medical research paper."""
    # In a real application, this would call an actual LLM API
    if "aspirin" in paper_text.lower():
        return "This paper discusses the effects of aspirin on cardiovascular health. It was observed to reduce inflammation and blood clotting at a dosage of 75mg daily. The target condition was prevention of heart attacks in high-risk patients. There were also mentions of its use for pain relief at higher doses."
    elif "metformin" in paper_text.lower():
        return "A study on metformin's role in type 2 diabetes management. Patients showed improved glycemic control. The standard dosage was 500mg twice daily. Some secondary effects on weight loss were also noted. This study focused on adult patients with newly diagnosed type 2 diabetes."
    else:
        return "This paper presents a general medical review. No specific drug findings are prominently discussed in a way that can be easily summarized for drug repurposing."

def secondary_llm_extractor(summary_text: str, extraction_prompt: str) -> dict:
    """Mocks a secondary LLM that extracts structured information from a summary.
    It uses a simple keyword-based approach to simulate extraction based on prompts.
    """
    extracted_data = {}

    # Simulate extraction for Drug Name
    if "*Drug Name*" in extraction_prompt:
        if "aspirin" in summary_text.lower():
            extracted_data["Drug Name"] = "Aspirin"
        elif "metformin" in summary_text.lower():
            extracted_data["Drug Name"] = "Metformin"

    # Simulate extraction for Observed Effect
    if "*Observed Effect*" in extraction_prompt:
        if "reduce inflammation" in summary_text.lower() and "aspirin" in summary_text.lower():
            extracted_data["Observed Effect"] = "Reduced inflammation and blood clotting"
        elif "improved glycemic control" in summary_text.lower() and "metformin" in summary_text.lower():
            extracted_data["Observed Effect"] = "Improved glycemic control"

    # Simulate extraction for Dosage
    if "*Dosage*" in extraction_prompt:
        if "75mg daily" in summary_text.lower():
            extracted_data["Dosage"] = "75mg daily"
        elif "500mg twice daily" in summary_text.lower():
            extracted_data["Dosage"] = "500mg twice daily"

    # Simulate extraction for Target Condition
    if "*Target Condition*" in extraction_prompt:
        if "prevention of heart attacks" in summary_text.lower():
            extracted_data["Target Condition"] = "Prevention of heart attacks in high-risk patients"
        elif "type 2 diabetes management" in summary_text.lower():
            extracted_data["Target Condition"] = "Type 2 diabetes management (newly diagnosed adult patients)"
            
    return extracted_data

# --- Example Usage ---
if __name__ == "__main__":
    # Mock medical research paper text
    paper_1 = """A comprehensive study on the efficacy of low-dose aspirin for primary prevention of cardiovascular events. Patients administered 75mg of aspirin daily showed a significant reduction in thrombotic events and inflammatory markers. Side effects were minimal. This research supports the use of aspirin in high-risk populations for preventing myocardial infarction."""

    paper_2 = """Recent clinical trials demonstrate the effectiveness of metformin in regulating blood glucose levels in adults with type 2 diabetes. A typical regimen involved 500mg administered orally twice per day. Beyond glycemic control, some participants experienced modest weight reduction, suggesting additional therapeutic benefits."""

    paper_3 = """An in-depth review of various neurological disorders and their current treatment paradigms. This paper extensively covers different therapeutic approaches but does not focus on specific drug trials or repurposing candidates."""

    print("\n--- Processing Paper 1 (Aspirin) ---")
    summary_1 = primary_llm_summarizer(paper_1)
    print(f"Primary LLM Summary:\n{summary_1}\n")

    # Prompt for secondary LLM extraction
    extraction_prompt_1 = (
        "From the following summary, identify the *Drug Name*. The drug name is:\n" +
        "Identify the *Observed Effect*. The effect is:\n" +
        "What was the *Dosage* mentioned? The dosage was:\n" +
        "What is the *Target Condition*? The target condition is:"
    )
    extracted_info_1 = secondary_llm_extractor(summary_1, extraction_prompt_1)
    print(f"Secondary LLM Extracted Information:\n{extracted_info_1}\n")

    print("\n--- Processing Paper 2 (Metformin) ---")
    summary_2 = primary_llm_summarizer(paper_2)
    print(f"Primary LLM Summary:\n{summary_2}\n")

    extraction_prompt_2 = (
        "Please extract the following information: *Drug Name*, *Observed Effect*, *Dosage*, and *Target Condition*. " +
        "Provide each item prefixed with 'Drug Name is:', 'Effect is:', 'Dosage is:', 'Target Condition is:'."
    )
    extracted_info_2 = secondary_llm_extractor(summary_2, extraction_prompt_2)
    print(f"Secondary LLM Extracted Information:\n{extracted_info_2}\n")

    print("\n--- Processing Paper 3 (General Review) ---")
    summary_3 = primary_llm_summarizer(paper_3)
    print(f"Primary LLM Summary:\n{summary_3}\n")

    extraction_prompt_3 = (
        "Extract *Drug Name*, *Observed Effect*, *Dosage*, and *Target Condition*. " +
        "If not found, state 'N/A'."
    )
    extracted_info_3 = secondary_llm_extractor(summary_3, extraction_prompt_3)
    print(f"Secondary LLM Extracted Information:\n{extracted_info_3}\n")