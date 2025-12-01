import streamlit as st
import time
import random

# --- Backend Simulation --- 

def simulate_knowledge_base_lookup(query):
    # Simulate a quick lookup in a pre-summarized knowledge base
    responses = {
        "contract drafting": "For contract drafting, it's essential to define parties, terms, conditions, and payment. Always consider jurisdiction.",
        "intellectual property": "Intellectual property generally covers patents, copyrights, trademarks, and trade secrets. Protecting it is crucial for business innovation.",
        "business formation": "Forming a business involves choosing a legal structure (sole proprietorship, LLC, corporation), registering with the state, and obtaining necessary licenses.",
        "employee hiring": "When hiring, ensure compliance with labor laws, prepare clear job descriptions, and have employment agreements in place."
    }
    return responses.get(query.lower(), "I can provide a general overview on that topic.")

def simulate_llm_deep_analysis(query):
    # Simulate a more complex, time-consuming LLM analysis
    time.sleep(random.uniform(3, 7)) # Simulate latency for deep analysis
    refined_responses = {
        "contract drafting": "A refined analysis for contract drafting emphasizes the inclusion of indemnification clauses, dispute resolution mechanisms (e.g., arbitration), force majeure events, and a comprehensive termination clause. Specificity in scope of work and deliverables is paramount. Consult with a legal professional for tailored agreements.",
        "intellectual property": "A deeper dive into intellectual property protection for small businesses reveals that a strategic combination of trademark registration for branding, copyright for original works (like marketing materials), and patent applications for novel inventions is often optimal. Confidentiality agreements are vital for trade secrets. Regular audits of IP assets are recommended.",
        "business formation": "Detailed guidance on business formation suggests that while an LLC offers liability protection and pass-through taxation benefits, a C-corp might be better for attracting venture capital due to its stock structure. Thoroughly research state-specific compliance requirements, including annual reports and registered agent services. Consider tax implications and consult an accountant.",
        "employee hiring": "For robust employee hiring, beyond basic labor law compliance (e.g., minimum wage, non-discrimination), it's crucial to implement a clear onboarding process, establish a comprehensive employee handbook covering company policies, and understand classification distinctions (employee vs. independent contractor) to avoid legal pitfalls. Background checks should be conducted in compliance with relevant regulations."
    }
    return refined_responses.get(query.lower(), "A more detailed analysis is being prepared for this query.")


# --- Streamlit Frontend --- 

st.set_page_config(page_title="AI Legal Advice Chatbot")
st.title("⚖️ AI Legal Advice Chatbot for Small Businesses")
st.markdown("Ask me about contract drafting, intellectual property, business formation, or employee hiring!")

user_query = st.text_input("Your legal question:", "")

if st.button("Get Advice"):
    if user_query:
        st.session_state.initial_response = None
        st.session_state.refined_response = None
        st.session_state.processing_refined = False

        st.write("\n--- Initial Response (Quick Overview) ---")
        with st.spinner("Generating quick initial response..."):
            initial_resp = simulate_knowledge_base_lookup(user_query)
            st.session_state.initial_response = initial_resp
            st.success(initial_resp)
        
        st.markdown("\n_A more accurate and detailed legal analysis is being processed in the background._")
        st.session_state.processing_refined = True

        # Start a thread or background task for the refined response (simulated here synchronously for simplicity)
        # In a real application, this would be an async call to a backend service
        with st.spinner("Performing deep legal analysis (this may take a few moments)..."):
            refined_resp = simulate_llm_deep_analysis(user_query)
            st.session_state.refined_response = refined_resp
            st.session_state.processing_refined = False
            st.success("Deep analysis complete!")

    else:
        st.warning("Please enter a legal question.")

# Display options for refined response
if "initial_response" in st.session_state and st.session_state.initial_response:
    if st.session_state.processing_refined:
        st.info("You can review the initial response above while the refined analysis continues.")
    elif "refined_response" in st.session_state and st.session_state.refined_response:
        st.write("\n---")
        if st.button("Show Refined Legal Advice"): # User explicitly chooses to see it
            st.write("\n--- Refined Legal Advice (Detailed Analysis) ---")
            st.markdown(st.session_state.refined_response)
        else:
             st.info("A more detailed legal analysis is ready. Click 'Show Refined Legal Advice' to view it.")

