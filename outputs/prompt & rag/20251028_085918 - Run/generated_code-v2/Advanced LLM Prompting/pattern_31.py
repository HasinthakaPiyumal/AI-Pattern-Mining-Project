import streamlit as st

def generate_marketing_copy(style, tone, genre, topic):
    """
    Simulates an LLM call to generate marketing copy based on specified style, tone, genre, and topic.
    In a real application, this would involve calling an actual LLM API (e.g., OpenAI, Hugging Face).
    """
    prompt = f"Generate a {genre} with a {style} and {tone} style about {topic}."
    
    # Placeholder for LLM API call
    # In a real application, you would integrate with an LLM like this:
    # from openai import OpenAI
    # client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful marketing copywriter."},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response.choices[0].message.content

    # Mock response for demonstration
    if "persuasive" in style.lower() and "enthusiastic" in tone.lower():
        return f"\n\n**Headline:** Unlock Your Potential with {topic}!\n**Body:** Experience the incredible power of {topic} and transform your world today! Don't miss out on this amazing opportunity. Get started now and achieve unparalleled success!"
    elif "humorous" in style.lower() and "casual" in tone.lower():
        return f"\n\n**Headline:** Why {topic} is Cooler Than Your Ex!\n**Body:** Let's be real, {topic} is here to make your life awesome. Ditch the boring stuff and dive into some serious fun with {topic}. Your future self will thank you (and probably send you a fruit basket)."
    elif "formal" in style.lower() and "informative" in tone.lower():
        return f"\n\n**Headline:** Comprehensive Analysis of {topic}\n**Body:** This document provides a detailed overview and in-depth analysis of {topic}, outlining its key features, benefits, and applications. Understanding {topic} is crucial for informed decision-making in the current landscape."
    else:
        return f"\n\n**Generated {genre} (placeholder):** Crafting content about {topic} with a {style} and {tone} approach. This is where your AI-generated copy would appear, tailored to your specific stylistic requests."

st.set_page_config(layout="wide", page_title="AI Marketing Copy Generator")

st.title("Marketing Copy Generator (Style Prompting Demo)")
st.markdown("Generate marketing content with specific style, tone, and genre using AI.")

with st.sidebar:
    st.header("Customize Your Copy")
    selected_style = st.selectbox(
        "Select Desired Style:",
        ["Persuasive", "Informative", "Humorous", "Formal", "Casual", "Enthusiastic", "Direct"],
        index=0
    )
    selected_tone = st.selectbox(
        "Select Desired Tone:",
        ["Enthusiastic", "Casual", "Formal", "Urgent", "Friendly", "Professional"],
        index=0
    )
    selected_genre = st.selectbox(
        "Select Content Genre:",
        ["Ad Copy", "Social Media Post", "Product Description", "Email Subject Line", "Blog Post Intro"],
        index=0
    )

st.subheader("Content Details")
content_topic = st.text_area(
    "Enter your content topic or keywords (e.g., 'new AI software features', 'eco-friendly product launch'):",
    height=100,
    placeholder="e.g., 'revolutionary fitness tracker', 'sustainable coffee beans'"
)

if st.button("Generate Marketing Copy", type="primary"):
    if content_topic:
        with st.spinner("Generating copy..."):
            generated_copy = generate_marketing_copy(selected_style, selected_tone, selected_genre, content_topic)
            st.subheader(f"Generated {selected_genre}:")
            st.write(generated_copy)
    else:
        st.warning("Please enter a content topic or keywords to generate copy.")

st.markdown("""
--- 
*This application demonstrates 'Style Prompting' where the AI's output is shaped by explicit stylistic instructions.*
""")