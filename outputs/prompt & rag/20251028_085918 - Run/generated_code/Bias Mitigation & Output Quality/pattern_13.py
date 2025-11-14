import streamlit as st
import openai
from dotenv import dotenv_values
import os
import random

# --- DenseFAQGenerator Class ---
class DenseFAQGenerator:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        # Initialize OpenAI client with the API key
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        # Load pre-defined exemplar sets for few-shot prompting
        self.exemplar_sets = self._load_exemplars()

    def _load_exemplars(self):
        """
        Defines and returns multiple distinct exemplar subsets for few-shot prompting.
        Each subset contains example user queries and expected assistant responses.
        These subsets represent different angles or contexts for customer support.
        """
        return [
            # Exemplar Set 1: General Product Information & Ordering
            [
                {"role": "user", "content": "What is your return policy?"},
                {"role": "assistant", "content": "Our return policy allows returns within 30 days of purchase for a full refund, provided the item is unused and in its original packaging. Special conditions may apply to electronics or personalized items."},
                {"role": "user", "content": "How do I track my order?"},
                {"role": "assistant", "content": "You can track your order by clicking the 'Track Order' link in your shipping confirmation email or by logging into your account and visiting the 'Order History' page."},
            ],
            # Exemplar Set 2: Technical Support & Troubleshooting for Electronics
            [
                {"role": "user", "content": "My new headset isn't connecting to my computer."},
                {"role": "assistant", "content": "First, ensure your headset is fully charged. Then, try restarting both your headset and computer. Check your computer's Bluetooth settings to ensure it's discoverable and try re-pairing the device."},
                {"role": "user", "content": "The app keeps crashing on my phone."},
                {"role": "assistant", "content": "Try force-closing the app and reopening it. If the issue persists, clear the app's cache (in settings) or reinstall the app. Ensure your phone's operating system is up to date."},
            ],
            # Exemplar Set 3: Account Management & Billing
            [
                {"role": "user", "content": "How can I update my billing information?"},
                {"role": "assistant", "content": "To update your billing information, log into your account, go to 'Account Settings', and select 'Payment Methods'. You can add or edit your details there."},
                {"role": "user", "content": "I was charged twice for the same item."},
                {"role": "assistant", "content": "Please check your order history to confirm. If it's a duplicate charge, contact our billing support with your order number, and we'll investigate and issue a refund if applicable."},
            ],
        ]

    def _generate_response_from_llm(self, messages: list) -> str:
        """
        Sends a list of messages to the OpenAI LLM and returns the generated content.
        Includes basic error handling for API calls.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7, # Controls creativity; 0.7 is a good balance for informative tasks
                max_tokens=300, # Max length of the generated response
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            st.error(f"OpenAI API Error: {e}. Please check your API key and network connection.")
            return "An error occurred while fetching a response."
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred."

    def generate_faq_answer(self, query: str) -> str:
        """
        Generates an FAQ answer using Demonstration Ensembling (DENSE).
        It creates multiple prompts with distinct exemplar subsets,
        sends them to the LLM, collects the diverse responses,
        and then uses a secondary LLM call to synthesize a single, robust answer.
        """
        st.info("Initiating DENSE process: Generating multiple responses with different exemplars.")
        raw_responses = []

        # Iterate through each exemplar set to generate diverse responses
        for i, exemplars in enumerate(self.exemplar_sets):
            messages = []
            messages.extend([
                {"role": "system", "content": "You are a helpful customer support assistant. Provide concise and accurate answers."}
            ])
            messages.extend(exemplars) # Add few-shot examples
            messages.append({"role": "user", "content": query})

            st.write(f"  - Generating response using Exemplar Set {i+1}...")
            response = self._generate_response_from_llm(messages)
            if response and "An error occurred" not in response: # Check for actual content
                raw_responses.append(f"Response from Set {i+1}:\n{response}")
                st.write(f"    - Raw Response {i+1} Snippet: {response[:150]}...") # Show a snippet
            else:
                st.warning(f"  - Failed to get a valid response from Exemplar Set {i+1}.")

        if not raw_responses:
            st.error("No valid raw responses could be generated. Please check the API key and query.")
            return "Could not generate any responses. Please try again or check the logs."

        # Ensembling Strategy: Use a secondary LLM call to synthesize the final answer
        st.info("Aggregating and synthesizing responses for a robust final answer.")
        ensemble_prompt = (
            "You have received several potential answers to a customer query. "
            "Please review these answers and synthesize a single, comprehensive, and robust FAQ answer. "
            "Prioritize accuracy and clarity. If there are conflicting answers, try to find a consensus or explain the nuances. "
            "Ensure the final answer is helpful and directly addresses the customer's original query.\n\n"
            f"Customer Query: \"{query}\"\n\n"
            "Raw Responses for consideration:\n" + "\n\n".join(raw_responses) + "\n\n"
            "Synthesized Final FAQ Answer:"
        )

        ensemble_messages = [
            {"role": "system", "content": "You are an expert AI assistant tasked with synthesizing customer support FAQ answers from multiple sources. Your goal is to provide the most accurate, comprehensive, and helpful answer possible."},
            {"role": "user", "content": ensemble_prompt}
        ]

        final_answer = self._generate_response_from_llm(ensemble_messages)
        return final_answer if final_answer and "An error occurred" not in final_answer else "Failed to synthesize a final answer from the collected responses."

# --- Streamlit Frontend (`app.py` logic) ---
def main():
    st.set_page_config(page_title="Dynamic FAQ Generator", layout="wide")
    st.title("💡 Dynamic FAQ Generator for Customer Support (DENSE)")
    st.markdown(
        """
        This application demonstrates **Demonstration Ensembling (DENSE)** to generate highly accurate
        and robust answers for customer support FAQs. It uses few-shot prompting with multiple,
        distinct exemplar subsets and then intelligently aggregates these diverse outputs
        to produce a single, reliable response.
        """
    )

    # Load environment variables (e.g., OPENAI_API_KEY) from .env file
    config = dotenv_values(".env")
    openai_api_key = config.get("OPENAI_API_KEY")

    if not openai_api_key:
        st.error(
            "OpenAI API key not found. "
            "Please create a `.env` file in the same directory as this script "
            "and add your OpenAI API key in the format: `OPENAI_API_KEY=\"YOUR_API_KEY\"`."
        )
        st.stop() # Stop the Streamlit app if the API key is missing

    # Initialize the DENSE FAQ Generator
    generator = DenseFAQGenerator(openai_api_key=openai_api_key)

    st.subheader("Enter Customer Query:")
    user_query = st.text_area(
        "Type your customer support question here:",
        height=100,
        placeholder="e.g., My payment failed, what should I do? Or, What are the warranty terms for my new laptop?"
    )

    if st.button("Generate FAQ Answer", type="primary"):
        if user_query:
            with st.spinner("Processing your query and generating ensembled answers..."):
                final_faq_answer = generator.generate_faq_answer(user_query)
                st.subheader("🤖 Synthesized FAQ Answer:")
                st.success(final_faq_answer)
        else:
            st.warning("Please enter a query in the text box above to generate an FAQ answer.")

if __name__ == "__main__":
    main()