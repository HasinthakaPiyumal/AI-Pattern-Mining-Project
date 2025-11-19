import streamlit as st
import requests

st.title("MedRec: Adaptive RAG Medical Information System")

user_query = st.text_area("Enter your medical query here:", height=150)

if st.button("Get Medical Information"):
    if user_query:
        st.info("Retrieving and generating information...")
        try:
            response = requests.post("http://localhost:8000/query/", json={"query": user_query})
            if response.status_code == 200:
                result = response.json()
                st.subheader("Generated Answer:")
                st.write(result.get("answer", "No answer generated."))
                
                if result.get("sources"):
                    st.subheader("Sources:")
                    for i, source in enumerate(result["sources"]):
                        st.write(f"- {source}")
            else:
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a query.")
