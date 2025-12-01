"""
This script demonstrates the concept of Behavior Cloning for an AI Customer Support Agent 
for an Internal CRM System using only Python built-in libraries.

The 'agent' learns to associate CRM states with appropriate human actions 
from a set of provided demonstrations. When presented with a new CRM state, 
it attempts to recall a similar learned state and suggest the corresponding action.

This is a conceptual implementation and does not involve actual machine learning models 
or NLP libraries, adhering to the constraint of using only built-in Python libraries.
"""

class BehaviorCloningCRMAgent:
    def __init__(self):
        self.learned_behaviors = {} # Stores {crm_state: human_action}

    def learn_from_demonstrations(self, demonstrations):
        """
        Teaches the agent by mapping CRM states to human actions.
        In a real scenario, this would be the 'fine-tuning' step.
        
        Args:
            demonstrations (list of dict): A list where each dict contains 
                                          'context' (CRM state description) 
                                          and 'action' (human command).
        """
        print("\n--- Learning from Demonstrations ---")
        for i, demo in enumerate(demonstrations):
            context = demo['context'].strip().lower()
            action = demo['action'].strip()
            
            if context in self.learned_behaviors:
                # For simplicity, if a context is seen multiple times, 
                # we keep the last action. A more robust system might handle conflicts 
                # or store multiple possible actions.
                print(f"Warning: Duplicate context '{context}' found. Overwriting action.")
            self.learned_behaviors[context] = action
            print(f"Learned: Context '{context}' -> Action '{action}'")
        print("--- Learning Complete ---")

    def predict_action(self, current_crm_state):
        """
        Predicts an action based on the current CRM state by finding the 
        best matching learned context.
        
        Args:
            current_crm_state (str): The current description of the CRM state.
            
        Returns:
            str: The predicted action or a default message if no similar state is found.
        """
        query_context = current_crm_state.strip().lower()
        
        # In a real NLP scenario, advanced similarity metrics would be used.
        # Here, we use a simple exact match for demonstration purposes.
        if query_context in self.learned_behaviors:
            print(f"\n--- Prediction for: '{current_crm_state}' ---")
            predicted_action = self.learned_behaviors[query_context]
            print(f"Found exact match. Predicted Action: '{predicted_action}'")
            return predicted_action
        else:
            # A more advanced pure-python approach could involve simple keyword matching
            # or substring search, but for clarity and adherence to behavior cloning's
            # direct mapping, we stick to exact match here.
            print(f"\n--- Prediction for: '{current_crm_state}' ---")
            print("No exact match found in learned behaviors. Suggesting a default action.")
            return "Please provide more details or try a different query."

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Collect Demonstrations (Human-issued commands in CRM context)
    # These mimic the (CRM_state, Human_Action) pairs
    human_demonstrations = [
        {"context": "Customer wants to check order status for ID 12345", 
         "action": "LOOKUP_ORDER 12345 STATUS"},
        {"context": "Customer requests to update shipping address for order 67890", 
         "action": "EDIT_ORDER 67890 SHIPPING_ADDRESS"},
        {"context": "Customer asks for refund process for damaged item", 
         "action": "INITIATE_REFUND DAMAGED_ITEM_POLICY"},
        {"context": "Agent needs to log a new customer complaint about billing", 
         "action": "CREATE_TICKET CATEGORY:BILLING PRIORITY:HIGH"},
        {"context": "Customer wants to know about product features of XYZ", 
         "action": "RETRIEVE_PRODUCT_INFO XYZ_PRODUCT"},
        {"context": "Customer wants to check order status for order id 54321", 
         "action": "LOOKUP_ORDER 54321 STATUS"}, # Duplicate context, demonstrating overwrite
    ]

    # 2. Initialize and Train (Fine-tune) the Agent using Behavior Cloning
    agent = BehaviorCloningCRMAgent()
    agent.learn_from_demonstrations(human_demonstrations)

    # 3. Use the Trained Agent (Inference)
    print("\n--- Agent in Action (Predicting Commands) ---")

    # Scenario 1: Exact match
    agent.predict_action("Customer wants to check order status for ID 12345")
    agent.predict_action("Customer requests to update shipping address for order 67890")
    
    # Scenario 2: Another exact match (demonstrates the overwritten action)
    agent.predict_action("Customer wants to check order status for order id 54321")

    # Scenario 3: No exact match
    agent.predict_action("Customer wants to cancel subscription service")
    agent.predict_action("find details on product ABC")
