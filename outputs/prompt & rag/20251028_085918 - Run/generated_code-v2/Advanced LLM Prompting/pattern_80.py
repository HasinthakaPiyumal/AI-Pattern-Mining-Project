from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def detect_emotion(text):
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "LABEL_0":
        return "negative"
    elif label == "LABEL_1":
        return "neutral"
    elif label == "LABEL_2":
        return "positive"
    return "neutral"

emotion_phrases = {
    "negative": "The customer is expressing significant frustration and needs a very understanding, empathetic, and resolution-focused response. It's crucial to address their core problem with utmost care and speed.",
    "positive": "The customer seems pleased. A friendly and helpful response will reinforce their positive experience and ensure continued satisfaction.",
    "neutral": "The customer query is straightforward. Provide a clear, concise, and helpful response.",
}

def generate_emotion_prompt(customer_query, emotion):
    system_message = f"You are a helpful and empathetic customer support assistant. {emotion_phrases.get(emotion, emotion_phrases['neutral'])}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{query}")
    ])
    return prompt_template.format(query=customer_query)

llm = ChatOpenAI(temperature=0.7)

def get_llm_response(prompt):
    response = llm.invoke(prompt)
    return response.content

def chatbot_interaction_loop():
    print("Welcome to the Emotion-Prompting Customer Support Chatbot! Type 'exit' to end the conversation.")
    while True:
        customer_input = input("You: ")
        if customer_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        detected_emotion = detect_emotion(customer_input)
        augmented_prompt = generate_emotion_prompt(customer_input, detected_emotion)
        llm_response = get_llm_response(augmented_prompt)
        print(f"Chatbot: {llm_response}")

if __name__ == "__main__":
    chatbot_interaction_loop()