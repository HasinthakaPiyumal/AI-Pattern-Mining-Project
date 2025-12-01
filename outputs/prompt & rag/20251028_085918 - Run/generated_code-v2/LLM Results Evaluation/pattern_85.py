import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your OpenAI API key as an environment variable or replace directly
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

def get_order_data(query_keywords):
    # Simulate fetching order data based on keywords
    if "order status" in query_keywords or "where is my order" in query_keywords:
        return {"order_number": "#12345", "status": "Shipped", "estimated_delivery": "2-3 business days", "tracking_link": "https://example.com/track/12345"}
    elif "return policy" in query_keywords:
        return {"policy": "Items can be returned within 30 days of purchase, provided they are in their original condition. Refunds are processed within 5-7 business days.", "exceptions": "Final sale items are not eligible for return."}
    elif "payment options" in query_keywords:
        return {"options": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.", "installments": "Klarna is available for installment payments on orders over $50."}
    else:
        return {"info": "No specific order data found for your query. Please provide more details."}

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Prompt for rephrasing the question
rephrase_prompt = PromptTemplate(
    input_variables=["customer_query"],
    template=(
        "You are an AI assistant designed to help customer support by first understanding the query deeply. "
        "Rephrase and expand the following customer query to ensure a comprehensive understanding of the user's intent, "
        "including any implied information or potential sub-questions. Focus on clarity and detail. "
        "Customer Query: {customer_query}\n\nRephrased and Expanded Query:"
    )
)

rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt)

# Prompt for generating the final response using the rephrased question and simulated data
respond_prompt = PromptTemplate(
    input_variables=["rephrased_query", "simulated_data"],
    template=(
        "You are a helpful and friendly e-commerce customer support chatbot. "
        "Based on the following rephrased customer query and the provided simulated order/policy data, "
        "generate a comprehensive and polite response to the customer. "
        "If the simulated data is relevant, incorporate it naturally into your answer. "
        "If the data indicates no specific information, kindly inform the user."
        "\n\nRephrased Query: {rephrased_query}"
        "\nSimulated Data: {simulated_data}"
        "\n\nCustomer Support Response:"
    )
)

respond_chain = LLMChain(llm=llm, prompt=respond_prompt)

def chatbot_response(customer_query):
    # Step 1: Rephrase and expand the customer's query
    rephrased_query_output = rephrase_chain.invoke({"customer_query": customer_query})
    rephrased_query = rephrased_query_output["text"].strip()

    # Step 2: Simulate data fetching based on the original query keywords
    # A more sophisticated system would parse the rephrased query for keywords
    # but for this demo, we use simple keyword matching on the original query.
    simulated_data = get_order_data(customer_query.lower())

    # Step 3: Generate the final response using the rephrased query and simulated data
    final_response_output = respond_chain.invoke({"rephrased_query": rephrased_query, "simulated_data": simulated_data})
    final_response = final_response_output["text"].strip()
    
    return f"**Rephrased Query:** {rephrased_query}\n\n**Chatbot Response:** {final_response}"

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything about your order or our policies..."),
    outputs="markdown",
    title="E-commerce Customer Support Chatbot (Rephrase and Respond)",
    description="This chatbot first rephrases your query for better understanding, then provides a detailed response based on simulated data."
)

if __name__ == "__main__":
    iface.launch()