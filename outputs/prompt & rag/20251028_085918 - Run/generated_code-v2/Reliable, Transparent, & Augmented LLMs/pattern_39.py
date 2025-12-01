
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def generate_culturally_aware_itinerary(destination: str, interests: str) -> str:
    # Initialize your LLM. Replace "YOUR_OPENAI_API_KEY" with your actual key
    # or ensure it's set as an environment variable.
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key="YOUR_OPENAI_API_KEY")

    # 1. Initial Itinerary Generation
    initial_prompt_template = PromptTemplate(
        input_variables=["destination", "interests"],
        template="Generate a detailed 3-day travel itinerary for {destination} focusing on {interests}. Provide specific activities, attractions, and suggested meal times for each day. Be informative and engaging."
    )
    initial_chain = initial_prompt_template | llm | StrOutputParser()
    basic_itinerary = initial_chain.invoke({"destination": destination, "interests": interests})

    # 2. Cultural Awareness Injection (Refinement 1)
    cultural_refinement_prompt_template = PromptTemplate(
        input_variables=["itinerary", "destination"],
        template="Given the following travel itinerary for {destination}:\n\n{itinerary}\n\nReview and refine this itinerary to ensure it is culturally sensitive and respectful of local customs and traditions in {destination}. Suggest modifications to activities, attire recommendations, appropriate behaviors, or timing where necessary to align with local etiquette and cultural norms. Highlight any crucial cultural considerations for visitors."
    )
    cultural_chain = (
        {"itinerary": lambda x: x, "destination": lambda x: destination}
        | cultural_refinement_prompt_template
        | llm
        | StrOutputParser()
    )
    culturally_refined_itinerary = cultural_chain.invoke(basic_itinerary)

    # 3. Culturally Relevant Language Injection (Refinement 2)
    language_refinement_prompt_template = PromptTemplate(
        input_variables=["itinerary", "destination"],
        template="Take the following culturally refined itinerary for {destination}:\n\n{itinerary}\n\nNow, subtly enhance this itinerary by naturally incorporating a few common and culturally relevant words, greetings, or short phrases from {destination}. For instance, if it's Japan, use 'Konnichiwa' or names of local dishes. If it's Italy, use 'Ciao' or names of Italian specialties. Do not translate the entire text, but strategically sprinkle in local language elements to make the itinerary feel more authentic and localized for a traveler, while still being easily understandable."
    )
    language_chain = (
        {"itinerary": lambda x: x, "destination": lambda x: destination}
        | language_refinement_prompt_template
        | llm
        | StrOutputParser()
    )
    final_itinerary = language_chain.invoke(culturally_refined_itinerary)

    return final_itinerary

st.title("🌍 Culturally Aware Travel Itinerary Generator")
st.markdown("Plan your trips with itineraries that respect local customs and traditions.")

destination_input = st.text_input("Enter your desired destination (e.g., Kyoto, Rome, Marrakech)", "Kyoto")
interests_input = st.text_area("What are your interests for this trip? (e.g., historical sites, local cuisine, art museums, nature walks)", "historical temples and traditional Japanese food")

if st.button("Generate Culturally Aware Itinerary"):
    if destination_input:
        with st.spinner("Crafting your culturally aware itinerary..."):
            generated_itinerary = generate_culturally_aware_itinerary(destination_input, interests_input)
            st.subheader(f"Your Culturally Aware Itinerary for {destination_input}")
            st.markdown(generated_itinerary)
    else:
        st.warning("Please enter a destination to generate an itinerary.")
