from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pandas as pd


class UnifiedInstructionFormatter:

    @staticmethod
    def format_product_qa(customer_question: str, product_description_and_faqs: str) -> str:
        return f"Instruction: \"Answer the following question about the product from the provided context. Question: {customer_question} Context: {product_description_and_faqs}\" Output:"

    @staticmethod
    def format_order_status(order_id: str, order_system_data: str) -> str:
        return f"Instruction: \"Provide the status for order {order_id} based on the following details. Details: {order_system_data}\" Output:"

    @staticmethod
    def format_troubleshooting_assistance(customer_problem: str, troubleshooting_knowledge_base_article: str) -> str:
        return f"Instruction: \"Assist the customer with their issue using the troubleshooting guide. Issue: {customer_problem} Guide: {troubleshooting_knowledge_base_article}\" Output:"

    @staticmethod
    def format_complaint_triage(customer_complaint_text: str, list_of_categories: list) -> str:
        categories_str = ", ".join(list_of_categories)
        return f"Instruction: \"Categorize the following customer complaint into one of these categories: {categories_str}. Complaint: {customer_complaint_text}\" Output:"


class CustomerSupportLLM:

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self, formatted_dataset: pd.DataFrame, num_epochs: int = 3):
        print(f"Placeholder for training LLM on {len(formatted_dataset)} samples for {num_epochs} epochs.")
        print("In a real scenario, this would involve tokenizing data, creating DataLoader, and training with a Hugging Face Trainer.")
        print("For this demonstration, we are using a pre-trained model directly for inference.")

    def inference(self, instruction_text: str, max_length: int = 150) -> str:
        inputs = self.tokenizer(instruction_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


app = FastAPI()
llm_model = CustomerSupportLLM()


class ProductQARequest(BaseModel):
    customer_question: str
    product_description_and_faqs: str


class OrderStatusRequest(BaseModel):
    order_id: str
    order_system_data: str


class TroubleshootingRequest(BaseModel):
    customer_problem: str
    troubleshooting_knowledge_base_article: str


class ComplaintTriageRequest(BaseModel):
    customer_complaint_text: str
    list_of_categories: list[str]


class PredictionRequest(BaseModel):
    task_type: str
    data: dict


@app.post("/predict")
async def predict(request: PredictionRequest):
    instruction_text = ""
    try:
        if request.task_type == "product_qa":
            req_data = ProductQARequest(**request.data)
            instruction_text = UnifiedInstructionFormatter.format_product_qa(
                req_data.customer_question,
                req_data.product_description_and_faqs
            )
        elif request.task_type == "order_status":
            req_data = OrderStatusRequest(**request.data)
            instruction_text = UnifiedInstructionFormatter.format_order_status(
                req_data.order_id,
                req_data.order_system_data
            )
        elif request.task_type == "troubleshooting":
            req_data = TroubleshootingRequest(**request.data)
            instruction_text = UnifiedInstructionFormatter.format_troubleshooting_assistance(
                req_data.customer_problem,
                req_data.troubleshooting_knowledge_base_article
            )
        elif request.task_type == "complaint_triage":
            req_data = ComplaintTriageRequest(**request.data)
            instruction_text = UnifiedInstructionFormatter.format_complaint_triage(
                req_data.customer_complaint_text,
                req_data.list_of_categories
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid task_type")

        response = llm_model.inference(instruction_text)
        return {"task_type": request.task_type, "instruction_input": instruction_text, "llm_response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("To run the FastAPI application, save this code as a Python file (e.g., app.py) and run:")
    print("uvicorn app:app --reload")
    print("\nExample usage (using curl):\n")
    print("Product QA:")
    print("curl -X POST -H \"Content-Type: application/json\" -d \'{\"task_type\": \"product_qa\", \"data\": {\"customer_question\": \"What are the dimensions of the laptop?\", \"product_description_and_faqs\": \"The XYZ Laptop features a 15.6-inch display, dimensions of 14.1 x 9.6 x 0.7 inches, and weighs 3.8 lbs.\"}}\' http://127.0.0.1:8000/predict\n")
    print("Order Status:")
    print("curl -X POST -H \"Content-Type: application/json\" -d \'{\"task_type\": \"order_status\", \"data\": {\"order_id\": \"12345\", \"order_system_data\": \"Order 12345 was shipped on 2023-10-26 and is expected to arrive by 2023-10-30.\"}}\' http://127.0.0.1:8000/predict\n")
    print("Complaint Triage:")
    print("curl -X POST -H \"Content-Type: application/json\" -d \'{\"task_type\": \"complaint_triage\", \"data\": {\"customer_complaint_text\": \"My new blender arrived broken, it doesn't even turn on.\", \"list_of_categories\": [\"damaged_item\", \"wrong_product\", \"billing_error\", \"technical_support\"]}}\' http://127.0.0.1:8000/predict")