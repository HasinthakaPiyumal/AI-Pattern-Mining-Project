from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class PromptTunedReader:
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.prompt_template = """
Answer the following question based on the provided medical context. If the answer is not in the context, state "Answer not found in the provided context."

Context: {context}

Question: {question}

Answer: """

    def read(self, question: str, contexts: list[str]) -> str:
        combined_context = " ".join(contexts)
        
        # Limit context length to avoid tokenizer issues and computational burden
        max_context_length = 512 - self.tokenizer.num_special_tokens_to_add(pair=False) - len(self.prompt_template.split('{context}')[0]) - len(self.prompt_template.split('{question}')[0]) - len(self.prompt_template.split('{context}')[1].split('{question}')[0]) # Approximate
        
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."

        prompt = self.prompt_template.format(context=combined_context, question=question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # For QuestionAnswering models like SQuAD-trained BERT, we get start and end logits
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits) + 1

        input_ids = inputs["input_ids"].squeeze().tolist()
        answer_tokens = input_ids[answer_start_index:answer_end_index]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Basic heuristic to check if the answer is plausible or simply the prompt
        # This might need more sophisticated handling in a real application
        if not answer or "Answer not found" in answer or len(answer.split()) < 2: # Heuristic for no answer found
            return "Answer not found in the provided context."

        return answer

app = FastAPI()
reader = PromptTunedReader()

class MedicalQuestion(BaseModel):
    question: str
    contexts: list[str]

@app.post("/qa")
async def get_medical_answer(medical_question: MedicalQuestion):
    answer = reader.read(medical_question.question, medical_question.contexts)
    return {"question": medical_question.question, "answer": answer}