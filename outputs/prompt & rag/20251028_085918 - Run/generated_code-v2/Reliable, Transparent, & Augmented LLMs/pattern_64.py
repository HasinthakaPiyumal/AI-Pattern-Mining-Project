import streamlit as st
from fastapi import FastAPI
import uvicorn
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

cultural_knowledge_base = {
    "Japan": {
        "greetings": "Bow slightly, don't hug unless close friends. Use honorifics like -san.",
        "dining": "Don't stick chopsticks upright in rice. Slurping noodles is a sign of enjoyment. Don't tip.",
        "etiquette": "Remove shoes before entering homes. Exchange business cards with two hands.",
        "taboos": "Don't be loud in public. Don't point with a single finger.",
    },
    "South Korea": {
        "greetings": "Bow to elders and those of higher status. Use two hands when giving or receiving.",
        "dining": "Wait for the eldest person to start eating. Don't leave chopsticks sticking upright in rice.",
        "etiquette": "Remove shoes before entering homes. Pour drinks for others, not yourself first. Refuse a drink gently a few times before accepting.",
        "taboos": "Don't write names in red ink. Don't blow your nose at the table.",
    },
    "Italy": {
        "greetings": "Kiss on both cheeks for close acquaintances. Handshake for formal greetings.",
        "dining": "Don't ask for extra cheese unless offered. Pasta is a first course, not a side dish. Don't put ketchup on pasta. Espresso is typically consumed standing at the bar.",
        "etiquette": "Dress well, especially in churches. Be punctual but don't expect others to be always.",
        "taboos": "Don't talk loudly on public transport. Don't be overly casual in dress in religious places.",
    },
}

def get_cultural_context(destination, query):
    context = []
    if destination in cultural_knowledge_base:
        destination_info = cultural_knowledge_base[destination]
        for category, info in destination_info.items():
            if any(word in query.lower() for word in category.split()) or \
               any(word in query.lower() for word in info.lower().split()):
                context.append(f"{category}: {info}")
        if not context:
             context.append(f"General etiquette for {destination}: {destination_info.get('etiquette', 'No general etiquette found.')}")
    return "\n".join(context) if context else f"No specific cultural context found for {destination} related to your query."

def translate_text(text, target_language="en"):
    return text

def get_llm_response(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful cultural travel assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with LLM: {e}"

def generate_cultural_advice(destination, query):
    cultural_context = get_cultural_context(destination, query)

    initial_prompt = (
        f"You are a travel guide. A user is asking about '{query}' for their trip to {destination}. "
        f"Here is some general cultural context for {destination}:\n{cultural_context}\n\n"
        f"Based on this, provide initial advice on '{query}' in {destination}."
    )
    initial_response = get_llm_response(initial_prompt)

    refinement_prompt = (
        f"The user wants advice on '{query}' for {destination}. I've received an initial response:\n\n"
        f"'{initial_response}'\n\n"
        f"Now, refine this advice specifically for {destination} culture, ensuring it is highly sensitive, "
        f"uses appropriate terminology, and highlights key customs or etiquette relevant to the query. "
        f"Emphasize what a traveler absolutely needs to know to avoid cultural faux pas and show respect. "
        f"Incorporate the following cultural context:\n{cultural_context}"
    )
    refined_response = get_llm_response(refinement_prompt)

    final_output = translate_text(refined_response, target_language="en")

    return final_output

app_fastapi = FastAPI()

@app_fastapi.post("/advice")
async def get_advice(destination: str, query: str):
    advice = generate_cultural_advice(destination, query)
    return {"cultural_advice": advice}

st.title("🌍 Cultural Travel Guide Assistant")
st.write("Get culturally sensitive advice for your travels!")

destination_input = st.text_input("Enter your travel destination (e.g., Japan, South Korea, Italy):")
query_input = st.text_area("What cultural aspect or situation are you curious about? (e.g., 'dining etiquette', 'greetings', 'visiting homes'):")

if st.button("Get Cultural Advice"):
    if destination_input and query_input:
        with st.spinner("Generating culturally sensitive advice..."):
            advice = generate_cultural_advice(destination_input, query_input)
            st.subheader(f"Advice for {destination_input} regarding '{query_input}':")
            st.info(advice)
    else:
        st.warning("Please enter both a destination and a query.")