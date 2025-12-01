
import json
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# --- 1. Pydantic Models for Structured Data ---

class RubricCriterion(BaseModel):
    name: str = Field(..., description="Name of the criterion (e.g., 'Clarity', 'Argument Strength')")
    description: str = Field(..., description="Detailed description of what constitutes a good performance for this criterion.")
    max_score: int = Field(..., description="Maximum score achievable for this criterion.")
    evaluation_steps: List[str] = Field(..., description="Step-by-step instructions for evaluating this criterion.")

class Rubric(BaseModel):
    title: str = Field(..., description="Title of the rubric (e.g., 'Essay Grading Rubric for Persuasive Writing')")
    overall_guidelines: str = Field(..., description="General guidelines for the essay evaluation.")
    criteria: List[RubricCriterion] = Field(..., description="List of specific grading criteria.")

class CriterionFeedback(BaseModel):
    criterion_name: str
    score: int
    feedback: str

class GradingFeedback(BaseModel):
    overall_score: int
    max_possible_score: int
    overall_comments: str
    criterion_feedback: List[CriterionFeedback]

# --- 2. Mock LLM Class (for demonstration without actual API calls) ---

class MockLLM:
    """A mock LLM to simulate responses for rubric generation and essay grading."""
    def invoke(self, prompt: str) -> str:
        if "generate a detailed grading rubric" in prompt.lower():
            # Simulate rubric generation
            return json.dumps({
                "title": "Persuasive Essay Grading Rubric",
                "overall_guidelines": "Essays will be evaluated based on the clarity of the argument, evidence provided, organization, and adherence to academic writing standards. Focus on the logical flow and the strength of the thesis statement.",
                "criteria": [
                    {
                        "name": "Thesis Statement & Argument",
                        "description": "The clarity, originality, and arguable nature of the thesis statement, and the consistency of the argument throughout the essay.",
                        "max_score": 25,
                        "evaluation_steps": [
                            "1. Identify the thesis statement. Is it clear and concise?",
                            "2. Is the thesis arguable and not merely a statement of fact?",
                            "3. Does the essay consistently support and defend this thesis?",
                            "4. Are there any contradictory points or digressions?"
                        ]
                    },
                    {
                        "name": "Evidence & Support",
                        "description": "The quality, relevance, and integration of evidence to support claims.",
                        "max_score": 25,
                        "evaluation_steps": [
                            "1. Are claims supported by relevant evidence (e.g., examples, facts, quotes)?",
                            "2. Is the evidence adequately explained and connected to the argument?",
                            "3. Is there sufficient evidence to persuade the reader?",
                            "4. Is the evidence accurately cited (if applicable)?"
                        ]
                    },
                    {
                        "name": "Organization & Cohesion",
                        "description": "The logical structure of the essay, including paragraphing, transitions, and overall flow.",
                        "max_score": 20,
                        "evaluation_steps": [
                            "1. Is there a clear introduction, body, and conclusion?",
                            "2. Do paragraphs have clear topic sentences and logical development?",
                            "3. Are transitions between paragraphs and ideas smooth and effective?",
                            "4. Does the essay maintain a coherent overall structure?"
                        ]
                    },
                    {
                        "name": "Language & Style",
                        "description": "Clarity, precision, word choice, sentence structure, and adherence to academic tone.",
                        "max_score": 15,
                        "evaluation_steps": [
                            "1. Is the language clear, precise, and appropriate for an academic essay?",
                            "2. Is there variety in sentence structure?",
                            "3. Are there significant grammatical errors, spelling mistakes, or typos?",
                            "4. Is the tone consistently academic and objective?"
                        ]
                    },
                     {
                        "name": "Conclusion",
                        "description": "The effectiveness of the conclusion in summarizing main points and providing a sense of closure, without introducing new information.",
                        "max_score": 15,
                        "evaluation_steps": [
                            "1. Does the conclusion effectively summarize the main arguments?",
                            "2. Does it restate the thesis in a new way?",
                            "3. Does it offer a final thought or implication without introducing new ideas?",
                            "4. Is it concise and impactful?"
                        ]
                    }
                ]
            })
        elif "grade the following essay based on the provided rubric" in prompt.lower():
            # Simulate essay grading
            return json.dumps({
                "overall_score": 78,
                "max_possible_score": 100,
                "overall_comments": "The essay presents a clear argument and uses relevant evidence, but could improve in transitional phrases and deeper analysis of some evidence. The conclusion effectively summarizes the main points.",
                "criterion_feedback": [
                    {"criterion_name": "Thesis Statement & Argument", "score": 20, "feedback": "Strong, clear thesis statement consistently argued throughout. Occasionally, some points felt slightly tangential but quickly reconnected."}, 
                    {"criterion_name": "Evidence & Support", "score": 22, "feedback": "Good use of relevant evidence. Some explanations of how the evidence supports the claim could be more in-depth. "}, 
                    {"criterion_name": "Organization & Cohesion", "score": 15, "feedback": "Generally well-organized, but some transitions between paragraphs felt abrupt, disrupting the flow slightly."},
                    {"criterion_name": "Language & Style", "score": 10, "feedback": "Clear and concise language. A few minor grammatical errors were present, but did not hinder comprehension."},
                    {"criterion_name": "Conclusion", "score": 11, "feedback": "The conclusion effectively summarized the main points and restated the thesis. It provided a good sense of closure."}
                ]
            })
        else:
            return json.dumps({"error": "Unknown mock prompt"})

# --- 3. Rubric Generation Function ---

def generate_rubric(essay_prompt: str, example_essays: Optional[List[str]] = None) -> Rubric:
    """Generates a detailed grading rubric using an LLM based on the essay prompt."""
    llm = MockLLM()
    
    prompt_parts = [
        f"You are an expert educator. Your task is to generate a detailed grading rubric for an essay based on the following prompt.",
        f"The essay prompt is: '{essay_prompt}'"
    ]

    if example_essays:
        prompt_parts.append("Consider the following example essays (expert-graded) to derive criteria, if helpful:")
        for i, essay in enumerate(example_essays):
            prompt_parts.append(f"Example {i+1}:\n{essay}")

    prompt_parts.append("\n\nGenerate a chain-of-thought of detailed evaluation steps for each criterion. The output MUST be a JSON object conforming to the Rubric Pydantic model structure:")
    prompt_parts.append(Rubric.schema_json(indent=2))

    full_prompt = "\n".join(prompt_parts)
    print("\n--- Rubric Generation Prompt ---\n")
    print(full_prompt)
    print("\n--- Mock LLM Response (Rubric) ---\n")
    
    llm_response_str = llm.invoke(full_prompt)
    print(llm_response_str)
    
    rubric_data = json.loads(llm_response_str)
    return Rubric(**rubric_data)

# --- 4. Essay Grading Function ---

def grade_essay(student_essay: str, rubric: Rubric) -> GradingFeedback:
    """Grades a student essay using an LLM and the provided rubric."""
    llm = MockLLM()

    rubric_json = rubric.json(indent=2)

    prompt_parts = [
        f"You are an expert essay grader. Grade the following essay based on the provided rubric.",
        "Provide a score and specific feedback for each criterion, as well as overall comments and an overall score.",
        "The output MUST be a JSON object conforming to the GradingFeedback Pydantic model structure:",
        GradingFeedback.schema_json(indent=2),
        f"\n\n--- Rubric ---\n{rubric_json}",
        f"\n\n--- Student Essay ---\n{student_essay}"
    ]

    full_prompt = "\n".join(prompt_parts)
    print("\n--- Essay Grading Prompt ---\n")
    print(full_prompt)
    print("\n--- Mock LLM Response (Grading) ---\n")

    llm_response_str = llm.invoke(full_prompt)
    print(llm_response_str)
    
    feedback_data = json.loads(llm_response_str)
    return GradingFeedback(**feedback_data)

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    # Sample Data
    essay_prompt = "Write a persuasive essay arguing for or against the implementation of mandatory community service hours for high school students."
    
    student_essay = """
    Mandatory community service for high school students is a terrible idea. It forces students to do work they don't want to do, taking away valuable time they could spend on studies or extracurriculars they actually care about. Instead of fostering a sense of civic duty, it breeds resentment. Students should be encouraged, not forced, to volunteer. Volunteering should come from the heart, not from a school requirement. Furthermore, many students already have busy schedules with demanding coursework, part-time jobs, and sports. Adding another compulsory activity only increases their stress levels and reduces their time for rest and personal development. This type of policy often fails to consider the diverse backgrounds and responsibilities of students, penalizing those who might not have the luxury of free time. Therefore, mandatory community service would be counterproductive and detrimental to student well-being.
    """

    print("\n### Automated Essay Grading System ###")

    # Step 1: Generate Rubric
    print("\nGenerating grading rubric...")
    generated_rubric = generate_rubric(essay_prompt=essay_prompt)
    print("\nRubric Generated Successfully!")
    print("\n--- Generated Rubric ---\n")
    print(generated_rubric.json(indent=2))

    # Step 2: Grade Essay using the generated rubric
    print("\nGrading student essay using the generated rubric...")
    grading_feedback = grade_essay(student_essay=student_essay, rubric=generated_rubric)
    print("\nEssay Graded Successfully!")
    print("\n--- Grading Feedback ---\n")
    print(grading_feedback.json(indent=2))

    print("\nTotal Score: {}/{} ({:.2f}%) ".format(
        grading_feedback.overall_score,
        grading_feedback.max_possible_score,
        (grading_feedback.overall_score / grading_feedback.max_possible_score) * 100
    ))
    print("Overall Comments: {}".format(grading_feedback.overall_comments))

