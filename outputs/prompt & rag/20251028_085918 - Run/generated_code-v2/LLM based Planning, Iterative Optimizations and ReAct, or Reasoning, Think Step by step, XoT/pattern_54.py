from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    id: str
    text: str

class AnswerSubmission(BaseModel):
    question_id: str
    answer: str

class Feedback(BaseModel):
    incorrect_reasoning: str
    why_incorrect: str
    correct_reasoning: str
    why_correct: str

# Simulated in-memory data for STEM questions and common misconceptions
QUESTIONS_DB = {
    "q1": {
        "text": "What is the result of 5 + 3 * 2?",
        "correct_answer": "11",
        "misconception": {
            "student_error_example": "I calculated 5 + 3 = 8, then 8 * 2 = 16. So the answer is 16.",
            "incorrect_reasoning_explanation": "This reasoning incorrectly applies the order of operations, performing addition before multiplication.",
            "correct_reasoning_example": "According to the order of operations (PEMDAS/BODMAS), multiplication should be performed before addition. So, first calculate 3 * 2 = 6, then 5 + 6 = 11. The answer is 11.",
            "correct_reasoning_explanation": "The correct order of operations dictates that multiplication takes precedence over addition. Failing to follow this leads to an incorrect result."
        }
    },
    "q2": {
        "text": "If a car travels at 60 mph for 2 hours, how far has it traveled?",
        "correct_answer": "120 miles",
        "misconception": {
            "student_error_example": "I just added 60 + 2 = 62. So the car traveled 62 miles.",
            "incorrect_reasoning_explanation": "This reasoning incorrectly adds speed and time instead of multiplying them to find distance.",
            "correct_reasoning_example": "Distance is calculated by multiplying speed by time. So, 60 mph * 2 hours = 120 miles. The car traveled 120 miles.",
            "correct_reasoning_explanation": "The formula for distance is speed multiplied by time. Understanding this fundamental relationship is key to solving such problems."
        }
    }
}

class LLMService:
    def generate_contrastive_feedback(self, question_text: str, student_answer: str, misconception_data: dict) -> Feedback:
        # In a real application, this would involve calling a large language model
        # using langchain to construct a prompt like:
        # "Here is a question: {question_text}. A student answered {student_answer}."
        # "Here is a common incorrect reasoning: {misconception_data['student_error_example']}. Why is this incorrect?"
        # "Now, provide the correct reasoning for the question. Why is this correct?"
        # For this demonstration, we simulate the LLM's response based on pre-defined misconception data.

        # Simulate LLM output based on the provided misconception data
        # The LLM would be prompted to extract and format this information.
        
        # In a real scenario, the LLM's output would be parsed into the Feedback object.
        # Here, we directly construct the Feedback object for demonstration.
        
        return Feedback(
            incorrect_reasoning=misconception_data["student_error_example"],
            why_incorrect=misconception_data["incorrect_reasoning_explanation"],
            correct_reasoning=misconception_data["correct_reasoning_example"],
            why_correct=misconception_data["correct_reasoning_explanation"]
        )

llm_service = LLMService()

@app.get("/question/{question_id}", response_model=Question)
def get_question(question_id: str):
    question = QUESTIONS_DB.get(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return Question(id=question_id, text=question["text"])

@app.post("/answer")
def submit_answer(submission: AnswerSubmission):
    question_data = QUESTIONS_DB.get(submission.question_id)
    if not question_data:
        raise HTTPException(status_code=404, detail="Question not found")

    if submission.answer == question_data["correct_answer"]:
        return {"message": "Correct! Great job!"}
    else:
        misconception = question_data.get("misconception")
        if misconception:
            feedback = llm_service.generate_contrastive_feedback(
                question_text=question_data["text"],
                student_answer=submission.answer,
                misconception_data=misconception
            )
            return {"message": "Incorrect. Here's some feedback to help you understand:", "feedback": feedback}
        else:
            return {"message": "Incorrect. Try again!"}
