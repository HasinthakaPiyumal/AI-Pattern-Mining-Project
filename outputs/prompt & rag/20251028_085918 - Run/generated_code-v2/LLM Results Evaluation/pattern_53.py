import streamlit as st
import requests

st.set_page_config(page_title="AI Customer Support Chatbot")
st.title("AI Customer Support Chatbot")

# User input
user_query = st.text_area("Ask your question here:", height=100)

if st.button("Get Answer"):
    if user_query:
        try:
            # Make a request to the FastAPI backend
            backend_url = "http://localhost:8000/chat"  # Adjust if your FastAPI runs on a different port/host
            response = requests.post(backend_url, json={"query": user_query})
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            chat_response = response.json()
            rephrased_question = chat_response.get("rephrased_question", "")
            final_answer = chat_response.get("final_answer", "")

            st.subheader("Rephrased Question:")
            st.write(rephrased_question)

            st.subheader("Final Answer:")
            st.write(final_answer)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please ensure the FastAPI server is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a question.")