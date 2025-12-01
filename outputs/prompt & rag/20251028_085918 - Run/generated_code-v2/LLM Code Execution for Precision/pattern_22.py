import streamlit as st
import os
from llm_service import generate_calculation_code, generate_final_response
from code_executor import execute_code

st.set_page_config(page_title="Personalized Financial Planning Assistant")
st.title("💰 Personalized Financial Planning Assistant")

st.write("Hello! I can help you with complex financial calculations and provide advice. Ask me anything about retirement planning, mortgage, investments, and more!")

# User input
user_query = st.text_area("Enter your financial question or scenario:", height=100)

if st.button("Get Financial Advice"):
    if user_query:
        st.info("Thinking and calculating...")
        
        try:
            # Step 1: LLM generates Python code for the calculation
            with st.spinner("Generating calculation logic..."):
                generated_code = generate_calculation_code(user_query)
            
            st.subheader("Generated Code (for transparency):")
            st.code(generated_code, language="python")

            # Step 2: Execute the generated code
            with st.spinner("Executing calculations..."):
                calculation_output, error = execute_code(generated_code)
            
            if error:
                st.error(f"Error during code execution: {error}")
                st.warning("I encountered an error during calculation. Please try rephrasing your query.")
            else:
                st.subheader("Calculation Result:")
                st.write(calculation_output)

                # Step 3: LLM integrates result and formulates final natural language response
                with st.spinner("Formulating your personalized advice..."):
                    final_advice = generate_final_response(user_query, calculation_output)
                
                st.subheader("Your Personalized Financial Advice:")
                st.success(final_advice)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.warning("Please check if your OpenAI API key is correctly set in your environment variables.")
    else:
        st.warning("Please enter your financial question to get started.")

st.markdown("---")
st.markdown("This assistant uses advanced AI to generate and execute code for precise financial calculations.")