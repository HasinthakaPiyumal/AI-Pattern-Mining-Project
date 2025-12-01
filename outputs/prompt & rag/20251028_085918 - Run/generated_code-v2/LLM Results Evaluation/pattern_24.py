import os
import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# Ensure OPENAI_API_KEY is set in your environment variables
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# 1. Evaluation Criteria Definition
class EvaluationCriteria(BaseModel):
    grammar: int = Field(..., description="Score for grammar (1-5, 5 being excellent)")
    relevance: int = Field(..., description="Score for relevance to the customer query (1-5, 5 being highly relevant)")
    helpfulness: int = Field(..., description="Score for helpfulness of the response (1-5, 5 being very helpful)")
    sentiment: int = Field(..., description="Score for the sentiment conveyed by the response (1-5, 5 being very positive)")

# 2. LLM Evaluation Output Schema
class LLMEvaluationOutput(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    turn_id: int = Field(..., description="Sequential identifier for the conversation turn")
    evaluation: EvaluationCriteria = Field(..., description="Multidimensional evaluation scores")
    overall_comment: str = Field(..., description="Overall qualitative comment on the chatbot's response")

# 3. Conversation Data Input (Simulated for Demo)
simulated_conversations = [
    {
        "conversation_id": "conv_001",
        "turn_id": 1,
        "customer_query": "I want to buy a new smartphone. What are your recommendations?",
        "chatbot_response": "We have a wide range of smartphones. Could you tell me your budget and preferred brand?"
    },
    {
        "conversation_id": "conv_001",
        "turn_id": 2,
        "customer_query": "My budget is around $500 and I prefer Samsung.",
        "chatbot_response": "Okay, for around $500, Samsung has the Galaxy A54 which is a great option. It has a good camera and long battery life."
    },
    {
        "conversation_id": "conv_002",
        "turn_id": 1,
        "customer_query": "How do I return an item? The shirt I ordered is too small.",
        "chatbot_response": "To return an item, please navigate to your 'Orders' page, select the item, and click on 'Return Item'. Follow the instructions there. You have 30 days from delivery."
    },
    {
        "conversation_id": "conv_003",
        "turn_id": 1,
        "customer_query": "What is the meaning of life?",
        "chatbot_response": "I am a customer support chatbot, and I can only assist with inquiries related to our products and services. For philosophical questions, I recommend consulting a philosophy expert."
    },
    {
        "conversation_id": "conv_004",
        "turn_id": 1,
        "customer_query": "Your website is not working correctly and I cannot complete my purchase.",
        "chatbot_response": "I apologize for the inconvenience. Could you please provide more details about the issue you're facing, such as the error message or browser you are using?"
    }
]

# 4. Evaluation Orchestrator (LangChain setup)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = PydanticOutputParser(pydantic_object=LLMEvaluationOutput)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant tasked with evaluating chatbot responses in customer support conversations for an e-commerce platform. Your evaluation should be multidimensional based on the provided criteria. Output your evaluation in JSON format conforming to the Pydantic schema provided.\n{format_instructions}"),
    ("human", "Evaluate the following chatbot interaction:\n\nConversation ID: {conversation_id}\nTurn ID: {turn_id}\nCustomer Query: {customer_query}\nChatbot Response: {chatbot_response}\n\nProvide scores (1-5, 5 being best) for grammar, relevance, helpfulness, and sentiment. Also, provide an overall qualitative comment.")
]).partial(format_instructions=parser.get_format_instructions())

evaluation_chain = prompt | llm | parser

# 5. Evaluation Loop and Reporting
all_evaluations = []

print("Starting chatbot response evaluation...")
for conv in simulated_conversations:
    print(f"\nEvaluating Conversation ID: {conv['conversation_id']}, Turn ID: {conv['turn_id']}")
    try:
        evaluation_result = evaluation_chain.invoke({
            "conversation_id": conv["conversation_id"],
            "turn_id": conv["turn_id"],
            "customer_query": conv["customer_query"],
            "chatbot_response": conv["chatbot_response"]
        })
        all_evaluations.append(evaluation_result.dict())
        print(f"  Evaluation complete: {evaluation_result.evaluation.dict()}")
    except ValidationError as e:
        print(f"  Pydantic Validation Error: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred during evaluation: {e}")

# 6. Reporting
if all_evaluations:
    df_evaluations = pd.DataFrame(all_evaluations)
    df_scores = pd.json_normalize(df_evaluations['evaluation'])
    df_final = pd.concat([df_evaluations.drop('evaluation', axis=1), df_scores], axis=1)
    print("\n--- Unified Multidimensional Evaluation Results ---")
    print(df_final.to_string())
    print("\nAverage Scores:")
    print(df_scores.mean().to_string())
else:
    print("No evaluations were successfully completed.")
