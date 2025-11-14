import random
from collections import Counter
from typing import List, Dict, Tuple
import re

# Mocking transformers for demonstration purposes.
# In a real scenario, these would be imported from the transformers library.
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class MockTokenizer:
    """A mock tokenizer to simulate tokenization without actual transformers library."""
    def encode(self, text: str, return_tensors: str = "pt"): # pylint: disable=unused-argument
        # In a real scenario, this would convert text to token IDs.
        # For mock, we just return a placeholder string that includes the original text.
        return f"mock_encoded_input_for_{text}"

    def decode(self, tokens: str): # pylint: disable=unused-argument
        # In a real scenario, this would convert token IDs back to text.
        # For mock, we assume the tokens are already the desired output string.
        return tokens

class MockModel:
    """A mock LLM to simulate text generation without actual transformers model loading."""
    def generate(self, input_ids_mock: str, max_length: int = 50): # pylint: disable=unused-argument
        # input_ids_mock will be in the format "mock_encoded_input_for_Query: [customer_query]\nClassification:"
        # Extract the customer_query from this mock string to simulate varied responses.
        match = re.search(r"Query: (.*?)\nClassification:", input_ids_mock)
        query = match.group(1).strip() if match else "unknown query"

        # Simulate LLM's classification and response based on keywords in the query.
        # This part mimics how a real LLM might classify and respond.
        query_lower = query.lower()
        if "return" in query_lower or "send back" in query_lower or "faulty" in query_lower or "refund" in query_lower:
            classification = "Returns"
            response = random.choice([
                "Please visit our returns page for instructions.",
                "Refunds are processed within 5-7 business days after item receipt.",
                "To initiate a return, please use our online portal."
            ])
        elif "order" in query_lower or "package" in query_lower or "delivery" in query_lower or "tracking" in query_lower:
            classification = "Shipping"
            response = random.choice([
                "Your order is on its way, you can check tracking details on our website.",
                "You can track your order status on our website.",
                "We are experiencing slight delays, but your package is being prepared for shipment."
            ])
        elif "account" in query_lower or "login" in query_lower or "password" in query_lower or "payment" in query_lower:
            classification = "Account Management"
            response = random.choice([
                "Please use the 'Forgot Password' link or contact support for account access.",
                "You can update your payment details in the 'Billing Information' section of your profile.",
                "For account issues, please log in and go to settings."
            ])
        elif "install" in query_lower or "setup" in query_lower or "compatible" in query_lower or "technical" in query_lower:
            classification = "Technical Support"
            response = random.choice([
                "Our technical support team can assist you with product setup. Please provide your product model.",
                "Please check the product specifications page or contact technical support with your device details.",
                "For installation guides, refer to the user manual or our online help center."
            ])
        else:
            classification = random.choice(["General Inquiry", "Unknown"])
            response = random.choice([
                "We will get back to you shortly.",
                "Could not process your request fully. Please provide more details.",
                "Thank you for contacting us. We'll forward your query to the relevant department."
            ])

        return f"Classification: {classification}\nResponse: {response}"


class DENSEChatbot:
    """Intelligent customer support chatbot leveraging Demonstration Ensembling (DENSE)."""

    def __init__(self, model_name: str, demonstrations: List[Dict], num_ensembles: int, demonstrations_per_ensemble: int):
        """
        Initializes the DENSEChatbot.

        Args:
            model_name (str): The name of the pre-trained language model (e.g., "google/flan-t5-small").
                              (Note: Mocks are used in this demonstration).
            demonstrations (List[Dict]): A list of example customer queries, their classifications, and response templates.
                                        Each dict: {"query": ..., "classification": ..., "response_template": ...}
            num_ensembles (int): The number of distinct prompt variations to generate for each customer query.
            demonstrations_per_ensemble (int): The number of example queries to include in each prompt variation.
        """
        # In a real application, uncomment the following lines and import from transformers:
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = MockTokenizer() # Using mock for demonstration
        self.model = MockModel()       # Using mock for demonstration

        self.demonstrations = demonstrations
        self.num_ensembles = num_ensembles
        self.demonstrations_per_ensemble = demonstrations_per_ensemble

        if not (0 < self.demonstrations_per_ensemble <= len(self.demonstrations)):
            raise ValueError("demonstrations_per_ensemble must be greater than 0 and less than or equal to the total number of demonstrations.")

    def _create_few_shot_prompt(self, subset_demos: List[Dict], customer_query: str) -> str:
        """
        Constructs a single few-shot prompt string for the LLM.

        Args:
            subset_demos (List[Dict]): A subset of demonstrations to include as examples in the prompt.
            customer_query (str): The new customer query to be classified and responded to.

        Returns:
            str: The formatted few-shot prompt string.
        """
        prompt_parts = []
        for demo in subset_demos:
            prompt_parts.append(
                f"Query: {demo['query']}\n"
                f"Classification: {demo['classification']}\n"
                f"Response: {demo['response_template']}\n\n"
            )
        prompt_parts.append(f"Query: {customer_query}\nClassification:")
        return "".join(prompt_parts)

    def _get_llm_output(self, prompt: str) -> Tuple[str, str]:
        """
        Interacts with the LLM to get a classification and response.

        Args:
            prompt (str): The fully constructed few-shot prompt.

        Returns:
            Tuple[str, str]: A tuple containing the predicted classification and generated response.
        """
        # In a real application, you would use:
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_length=50)
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # For demonstration, we simulate LLM output using the MockModel.
        mock_input_id = self.tokenizer.encode(prompt)
        simulated_raw_output = self.model.generate(mock_input_id)

        classification = "Unknown"
        response = "Could not generate a response."

        # Attempt to parse the simulated output (e.g., "Classification: ...\nResponse: ...")
        class_match = re.search(r"Classification: (.*?)(?=\nResponse:|$)", simulated_raw_output)
        if class_match:
            classification = class_match.group(1).strip()

        response_match = re.search(r"Response: (.*)", simulated_raw_output)
        if response_match:
            response = response_match.group(1).strip()

        return classification, response

    def predict(self, customer_query: str) -> Dict:
        """
        Main inference method for the chatbot, applying Demonstration Ensembling.

        Args:
            customer_query (str): The customer's new query.

        Returns:
            Dict: A dictionary containing the aggregated classification, final response, and all ensemble outputs.
        """
        predicted_classifications = []
        generated_responses = []

        for _ in range(self.num_ensembles):
            # 1. Demonstration Sampling: Randomly select distinct demonstrations for each ensemble member.
            subset_demos = random.sample(self.demonstrations, self.demonstrations_per_ensemble)

            # 2. Prompt Generation: Create a few-shot prompt with the sampled demonstrations.
            prompt = self._create_few_shot_prompt(subset_demos, customer_query)

            # 3. LLM Inference: Get classification and response from the LLM (or mock).
            classification, response = self._get_llm_output(prompt)

            predicted_classifications.append(classification)
            generated_responses.append(response)

        # 4. Aggregation:
        #   Classification Aggregation: Use majority voting.
        classification_counts = Counter(predicted_classifications)
        aggregated_classification = classification_counts.most_common(1)[0][0]

        #   Response Aggregation:
        #   Prioritize using a response template from the original demonstrations if it matches the aggregated classification.
        final_response = "We are currently experiencing high volume, please bear with us." # Default fallback
        found_template = False
        for demo in self.demonstrations:
            if demo["classification"] == aggregated_classification:
                final_response = demo["response_template"]
                found_template = True
                break

        # If no matching template from demonstrations, use the most frequent generated response from the LLM ensemble.
        if not found_template:
            response_counts = Counter(generated_responses)
            if response_counts:
                final_response = response_counts.most_common(1)[0][0]

        return {
            "query": customer_query,
            "aggregated_classification": aggregated_classification,
            "final_response": final_response,
            "all_classifications": predicted_classifications, # For transparency/debugging
            "all_responses": generated_responses # For transparency/debugging
        }

# Example Usage:
if __name__ == "__main__":
    # Define a sample list of demonstrations for the chatbot.
    sample_demonstrations = [
        {"query": "My order hasn't arrived yet.", "classification": "Shipping", "response_template": "Your order is on its way, you can check tracking details on our website."},
        {"query": "I want to change my delivery address.", "classification": "Shipping", "response_template": "Please update your delivery address in your account settings before shipment."},
        {"query": "How do I return a faulty product?", "classification": "Returns", "response_template": "Visit our returns portal to initiate a return for faulty items."},
        {"query": "What is your refund policy?", "classification": "Returns", "response_template": "Refunds are processed within 5-7 business days after item receipt."},
        {"query": "I can't log into my account.", "classification": "Account Management", "response_template": "Please use the 'Forgot Password' link or contact support for account access."},
        {"query": "How do I update my payment method?", "classification": "Account Management", "response_template": "You can update your payment details in the 'Billing Information' section of your profile."},
        {"query": "I need help with product setup.", "classification": "Technical Support", "response_template": "Our technical support team can assist you with product setup. Please provide your product model."},
        {"query": "Is this product compatible with my device?", "classification": "Technical Support", "response_template": "Please check the product specifications page or contact technical support with your device details."},
    ]

    # Instantiate the DENSEChatbot.
    # In a real application, "google/flan-t5-small" would typically be loaded from Hugging Face.
    chatbot = DENSEChatbot(
        model_name="google/flan-t5-small", # Placeholder model name for initialization
        demonstrations=sample_demonstrations,
        num_ensembles=5,  # Number of distinct prompt variations for ensembling
        demonstrations_per_ensemble=2 # Number of example demonstrations in each prompt
    )

    # Test the chatbot with various customer queries.
    queries = [
        "Where is my package?",
        "I want to send something back.",
        "My password is not working.",
        "How do I install this software?",
        "I have a question about billing.", # This query might lead to varied classifications and test fallback response logic.
        "I need help with a refund.",
        "When will my order arrive?"
    ]

    for query in queries:
        print(f"\n--- Customer Query: \"{query}\" ---")
        result = chatbot.predict(query)
        print(f"Aggregated Classification: {result['aggregated_classification']}")
        print(f"Final Response: {result['final_response']}")
        print(f"All classifications from ensembles: {result['all_classifications']}")
        print(f"All responses from ensembles: {result['all_responses']}")
