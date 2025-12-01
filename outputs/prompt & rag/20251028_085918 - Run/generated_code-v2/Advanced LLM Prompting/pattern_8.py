import streamlit as st

# --- 1. Sentiment Analysis (Simplified) ---
def detect_emotion(user_query: str) -> str:
    """
    A simplified function to detect emotion based on keywords.
    In a real application, a more sophisticated NLP model (e.g., from transformers library)
    would be used for sentiment analysis.
    """
    query_lower = user_query.lower()
    if any(keyword in query_lower for keyword in ["frustrated", "angry", "unhappy", "problem", "issue", "not working"]):
        return "frustrated"
    elif any(keyword in query_lower for keyword in ["happy", "satisfied", "great", "thanks", "good"]):
        return "happy"
    elif any(keyword in query_lower for keyword in ["confused", "don't understand", "clarify"]):
        return "confused"
    else:
        return "neutral"

# --- 2. Prompt Engineering Component ---
def create_emotion_prompt(user_query: str, emotion: str) -> str:
    """
    Constructs an 'emotion-prompted' prompt for the LLM.
    """
    emotion_phrases = {
        "frustrated": "A customer is very frustrated and this issue is critical for them. Please provide a calming and effective solution, focusing on empathy and quick resolution:",
        "happy": "A happy customer is asking a question. Please provide a friendly and helpful response:",
        "confused": "A customer seems confused and needs clarification. Please explain clearly and patiently:",
        "neutral": "A customer has a query. Please provide a helpful and professional response:"
    }
    
    prompt_prefix = emotion_phrases.get(emotion, emotion_phrases["neutral"])
    return f"{prompt_prefix} {user_query}"

# --- 3. LLM Interaction (Mock Function) ---
def get_llm_response(prompt: str) -> str:
    """
    A mock function to simulate an LLM response.
    In a real application, this would involve calling an actual LLM API
    (e.g., OpenAI, Hugging Face's transformers, etc.).
    """
    st.info("Simulating LLM response...")
    # Simulate different responses based on internal prompt cues or simply return a generic response
    if "frustrated" in prompt:
        return "I understand this is frustrating, and I'm here to help you resolve this quickly. Let's look into [user's query topic] immediately. Could you please provide more details on [specific detail]?"
    elif "happy" in prompt:
        return "That's wonderful to hear! I'm happy to assist you with [user's query topic]. How can I help further?"
    elif "confused" in prompt:
        return "I can certainly help clarify that for you. [Explain topic simply]. Does that make more sense?"
    else:
        return "Thank you for reaching out. I'm here to help with your request: [user's query topic]. Please let me know how I can assist you further."


# --- Streamlit UI ---
st.set_page_config(page_title="Empathy-Driven Chatbot", layout="centered")
st.title(" empathetic Customer Support Chatbot")
st.markdown("This chatbot uses **Emotion Prompting** to generate more empathetic and contextually aware responses.")

user_input = st.text_area("How can I help you today?", "I'm really frustrated with my internet connection, it's not working!")

if st.button("Get Empathic Response"):
    if user_input:
        st.subheader("Detected Emotion:")
        emotion = detect_emotion(user_input)
        st.write(f"Emotion: **{emotion.capitalize()}**")

        st.subheader("Emotion-Prompted LLM Input:")
        engineered_prompt = create_emotion_prompt(user_input, emotion)
        st.code(engineered_prompt, language="python")

        st.subheader("Chatbot's Empathic Response:")
        llm_response = get_llm_response(engineered_prompt)
        st.write(llm_response)
    else:
        st.warning("Please enter your query to get a response.")

st.markdown("""
<style>
.stButton>button {background-color: #4CAF50; color: white;}
</style>
""", unsafe_allow_html=True)
