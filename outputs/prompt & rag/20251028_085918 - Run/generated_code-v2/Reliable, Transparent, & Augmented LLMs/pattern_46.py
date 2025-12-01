import streamlit as st
import requests
import json

# --- FastAPI Backend (Simulated for local execution) ---
# In a real application, this would be a separate FastAPI server
# running on a different port or deployed.

def generate_medical_response(query: str):
    # Simulate web search and retrieval of medical articles
    # In a real scenario, this would involve calling a search API (e.g., Google Custom Search)
    # and parsing content from authoritative medical websites.
    simulated_articles = {
        "diabetes treatment": [
            {
                "title": "Insulin Therapy for Type 1 Diabetes",
                "url": "https://www.medicaljournal.org/diabetes/insulin-therapy",
                "content": "Insulin therapy is crucial for individuals with type 1 diabetes. It involves administering insulin to regulate blood sugar levels. Various types of insulin are available, including rapid-acting, short-acting, intermediate-acting, and long-acting insulin. \n\n Patients often use a combination of basal and bolus insulin to mimic the body's natural insulin production. \n Education on proper injection techniques and carbohydrate counting is vital for effective management."
            },
            {
                "title": "Lifestyle Modifications for Type 2 Diabetes",
                "url": "https://www.healthylife.org/type2-diabetes/lifestyle",
                "content": "Lifestyle modifications, including diet and exercise, are fundamental in managing type 2 diabetes. \n \n A balanced diet low in processed sugars and saturated fats, coupled with regular physical activity, can significantly improve glycemic control. \n Weight loss often plays a key role in reversing insulin resistance."
            }
        ],
        "common cold remedies": [
            {
                "title": "Managing Symptoms of the Common Cold",
                "url": "https://www.cdc.gov/cold/symptoms",
                "content": "There is no cure for the common cold, but symptoms can be managed. Rest, hydration, and over-the-counter medications like pain relievers (e.g., ibuprofen, acetaminophen) and decongestants can provide relief. \n \n Antibiotics are ineffective against viral infections like the common cold."
            },
            {
                "title": "Home Remedies for Cold Relief",
                "url": "https://www.webmd.com/cold-flu/home-remedies-cold",
                "content": "Many home remedies can soothe cold symptoms. Gargling with salt water can relieve a sore throat. Drinking warm liquids like tea with honey can also be comforting. Using a humidifier may help with nasal congestion."
            }
        ]
    }

    # Select relevant articles based on query (simplified matching)
    relevant_articles = []
    for keyword, articles in simulated_articles.items():
        if keyword in query.lower():
            relevant_articles.extend(articles)
            break # Take the first match for simplicity

    if not relevant_articles:
        return {"answer": "I couldn't find specific medical information for that query at the moment. Please try a different query.", "references": []}

    # --- Simulate LLM Response Generation ---
    # In a real application, this would involve calling an actual LLM (e.g., OpenAI API)
    # with the query and the content from relevant_articles as context.
    # For this example, we'll construct a simple response based on the query and simulate references.

    generated_answer = f"Based on your query regarding '{query}', here is some general medical information. "
    references = []

    for article in relevant_articles:
        # Simulate extracting a key sentence/passage as a reference
        first_sentence = article["content"].split('. ')[0] + "."
        generated_answer += f"\n\n{first_sentence} "
        references.append({"passage": first_sentence, "url": article["url"], "title": article["title"]})
    
    generated_answer += "\n\nFor more detailed information, please consult the provided references and a healthcare professional."

    return {"answer": generated_answer, "references": references}

# --- Streamlit Frontend ---
st.set_page_config(layout="wide")
st.title("\u2728 Medical Information Assistant with References")
st.markdown(
    "This AI assistant provides medical information with supporting references to ensure factual accuracy and transparency. "
    "Enter a medical query below to get started."
)

query = st.text_area("Ask a medical question:", "Tell me about diabetes treatment")

if st.button("Get Information"):
    if query:
        with st.spinner("Searching for medical information and generating response..."):
            # In a real setup, this would be an HTTP POST request to the FastAPI backend
            # response = requests.post("http://localhost:8000/medical_query", json={"query": query})
            # data = response.json()

            # For this simulated local execution, call the function directly
            data = generate_medical_response(query)

            st.subheader("\u25B6 AI-Generated Answer:")
            st.write(data["answer"])

            if data["references"]:
                st.subheader("\u25B6 Supporting References:")
                for ref in data["references"]:
                    st.markdown(f"**From '{ref['title']}':**")
                    st.info(f"> {ref['passage']}")
                    st.markdown(f"[Source: {ref['url']}]({ref['url']})")
                    st.markdown("--- ")
            else:
                st.info("No specific references found for this query.")
    else:
        st.warning("Please enter a medical question.")

st.markdown("\n---\n*Disclaimer: This is an AI assistant for informational purposes only and should not be used as a substitute for professional medical advice.*")