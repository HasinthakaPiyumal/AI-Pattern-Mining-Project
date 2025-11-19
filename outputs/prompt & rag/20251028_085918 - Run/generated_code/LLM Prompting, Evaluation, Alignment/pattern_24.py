from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import random

app = FastAPI()

# Initialize LLM
# Ensure OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# --- 1. Vector Database (Simulated for Few-Shot Examples) ---
few_shot_examples = {
    "order_status": [
        {"query": "Where is my order #12345?", "response": "Your order #12345 is currently in transit and expected to arrive by [Date]. You can track it here: [Tracking Link]"},
        {"query": "Can I get an update on order 67890?", "response": "Order #67890 has been shipped and is scheduled for delivery on [Date]. Track its journey here: [Tracking Link]"}
    ],
    "product_inquiry": [
        {"query": "Tell me about the new XYZ laptop.", "response": "The XYZ laptop features a [Processor], [RAM]GB RAM, and a [Storage]TB SSD. It's ideal for [use case]."},
        {"query": "What are the specs of item ABC?", "response": "Item ABC comes with a [feature 1], [feature 2], and is available in [colors]."}
    ],
    "return_request": [
        {"query": "How do I return a faulty product?", "response": "To initiate a return for a faulty product, please visit our Returns Center at [URL] and follow the instructions. You'll need your order number."}, 
        {"query": "I want to send back an item.", "response": "Our return policy allows returns within 30 days of purchase. Please visit [URL] to start your return process."}
    ],
    "complaint": [
        {"query": "My package was damaged upon arrival!", "response": "I apologize for the damaged package. Please provide your order number, and we'll arrange for a replacement or refund immediately. We'll also investigate the shipping issue."},
        {"query": "The customer service was terrible.", "response": "I'm very sorry to hear about your negative experience. Please tell me more about what happened so I can escalate this and ensure it doesn't happen again."}
    ]
}

def get_few_shot_examples(query_type: str, num_examples: int = 2) -> list:
    if query_type in few_shot_examples:
        return random.sample(few_shot_examples[query_type], min(num_examples, len(few_shot_examples[query_type])))
    return []

# --- 2. Query Classifier ---
query_classifier_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful assistant that classifies customer queries into one of these categories: order_status, product_inquiry, return_request, complaint, general_inquiry. Respond with only the category name."),
    HumanMessage(content="{query}")
])

query_classifier_chain = query_classifier_prompt | llm | StrOutputParser()

# --- 3. Sentiment Analyzer (Simplified) ---
def analyze_sentiment(text: str) -> str:
    text_lower = text.lower()
    if "damage" in text_lower or "faulty" in text_lower or "terrible" in text_lower or "unhappy" in text_lower:
        return "negative"
    elif "thank you" in text_lower or "great" in text_lower or "happy" in text_lower:
        return "positive"
    return "neutral"

# --- 4. Dynamic Prompt Generator ---
def create_dynamic_prompt(query: str, query_type: str, sentiment: str, examples: list) -> ChatPromptTemplate:
    system_message_content = (
        "You are an AI-powered customer support assistant for an e-commerce platform. "
        "Your goal is to provide accurate, helpful, and ethically aligned responses. "
        "Adhere to ethical principles: be truthful, avoid bias, and prioritize customer satisfaction. "
    )

    # Role, Style, Emotion Prompting
    if sentiment == "negative":
        system_message_content += "Respond with an empathetic, apologetic, and solution-oriented tone."
    elif sentiment == "positive":
        system_message_content += "Respond with an enthusiastic and helpful tone."
    else:
        system_message_content += "Respond with a professional and informative tone."

    # Template-Based Prompting - Add specific instructions based on query type
    if query_type == "order_status":
        system_message_content += " Provide order status and tracking information clearly."
    elif query_type == "product_inquiry":
        system_message_content += " Provide detailed product information and recommendations."
    elif query_type == "return_request":
        system_message_content += " Guide the customer through the return process."
    elif query_type == "complaint":
        system_message_content += " Acknowledge the complaint, apologize, and offer concrete next steps."

    messages = [SystemMessage(system_message_content)]

    # Few-Shot Prompting (if examples exist)
    for ex in examples:
        messages.append(HumanMessage(content=ex["query"]))
        messages.append(AIMessage(content=ex["response"]))
    
    messages.append(HumanMessage(content="{query}"))
    return ChatPromptTemplate.from_messages(messages)

# --- 5. Reasoning Engine (Rephrase and Respond, Prompt Chains Placeholder) ---
# For simplicity, Rephrase and Respond is integrated into the main flow for a single retry.
# Prompt Chains for complex multi-step tasks would involve more intricate Langchain Agent setups.

# --- 6. Validation Layer (Simplified LLM-based evaluation - Autoraters/LLMEVAL Placeholder) ---
def validate_response_llm(original_query: str, generated_response: str) -> str:
    validation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an AI assistant tasked with validating customer support responses. "
            "Assess if the 'generated_response' accurately and appropriately answers the 'original_query'. "
            "Consider truthfulness, helpfulness, and adherence to ethical guidelines (e.g., no bias, no misinformation). "
            "Respond with 'VALID' if it passes, otherwise provide a brief reason for 'INVALID' followed by the reason."
        ),
        HumanMessage(content=f"Original Query: {original_query}\nGenerated Response: {generated_response}")
    ])
    
    validation_chain = validation_prompt | llm | StrOutputParser()
    validation_result = validation_chain.invoke({"original_query": original_query, "generated_response": generated_response})
    return validation_result

# --- FastAPI Endpoint ---
class CustomerQuery(BaseModel):
    query: str

@app.post("/support")
async def get_support_response(customer_query: CustomerQuery):
    query = customer_query.query
    print(f"Received query: {query}")

    try:
        # 1. Classify Query
        query_type = query_classifier_chain.invoke({"query": query}).strip().lower()
        print(f"Classified as: {query_type}")
        if query_type not in few_shot_examples and query_type != "general_inquiry":
            query_type = "general_inquiry" # Fallback for unhandled types

        # 2. Analyze Sentiment
        sentiment = analyze_sentiment(query)
        print(f"Detected sentiment: {sentiment}")

        # 3. Retrieve Few-Shot Examples
        examples = get_few_shot_examples(query_type)

        # 4. Generate Dynamic Prompt
        dynamic_prompt = create_dynamic_prompt(query, query_type, sentiment, examples)
        
        # Create the main LLM chain
        response_chain = dynamic_prompt | llm | StrOutputParser()

        # 5. Get Initial Response
        initial_response = response_chain.invoke({"query": query})
        print(f"Initial response: {initial_response}")

        # 6. Validate Response
        validation_result = validate_response_llm(query, initial_response)
        print(f"Validation result: {validation_result}")

        final_response = initial_response

        # 7. Rephrase and Respond (Simplified - retry if initial validation fails)
        if validation_result.startswith("INVALID"):
            print("Initial response invalid, attempting to rephrase.")
            rephrase_prompt = ChatPromptTemplate.from_messages([
                SystemMessage("The previous response was deemed invalid because: " + validation_result[7:] + ". Please rephrase and improve your answer to the customer's query: '{original_query}'. Your previous answer was: '{previous_response}'."),
                HumanMessage(content="{original_query}")
            ])
            rephrase_chain = rephrase_prompt | llm | StrOutputParser()
            rephrased_response = rephrase_chain.invoke({"original_query": query, "previous_response": initial_response})
            
            # Re-validate rephrased response
            re_validation_result = validate_response_llm(query, rephrased_response)
            print(f"Rephrased response validation: {re_validation_result}")
            if re_validation_result.startswith("VALID"):
                final_response = rephrased_response
            else:
                final_response = "I apologize, I'm having trouble providing a satisfactory answer at the moment. Please try again later or contact our human support team." # Fallback
        
        return {"query": query, "response": final_response, "query_type": query_type, "sentiment": sentiment, "validation_status": validation_result if validation_result.startswith("VALID") else re_validation_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this FastAPI application:
# 1. Save the code as customer_support_assistant.py
# 2. Set your OpenAI API key: export OPENAI_API_KEY="your_api_key_here"
# 3. Install dependencies: pip install fastapi uvicorn langchain-openai langchain-core pydantic
# 4. Run: uvicorn customer_support_assistant:app --reload
# Then access via http://127.0.0.1:8000/support with a POST request.