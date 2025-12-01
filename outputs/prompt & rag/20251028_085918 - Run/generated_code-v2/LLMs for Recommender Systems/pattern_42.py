import json
import re
from memory_manager import MemoryManager
from recommendation_tools import RecommendationTools

class LLMDialogueModule:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.recommendation_tools = RecommendationTools()
        # Simulate an LLM - in a real application, this would be an actual LLM client (e.g., OpenAI, HuggingFace Transformers)
        self.llm_model = self._simulate_llm_response 

    def _simulate_llm_response(self, prompt):
        # A very basic LLM simulation based on keywords and simple logic
        prompt_lower = prompt.lower()
        
        # Tool call for recommendations
        if "recommend" in prompt_lower or "suggest" in prompt_lower:
            category_match = re.search(r"category (\w+)", prompt_lower)
            keywords_match = re.search(r"about (.+?)(?:$|\.)", prompt_lower)
            category = category_match.group(1) if category_match else None
            keywords = keywords_match.group(1).split() if keywords_match else None
            
            if category or keywords:
                return f"Thought: User is asking for recommendations. I should use the `get_product_recommendations` tool.\nAction: call_tool(get_product_recommendations, category={json.dumps(category)}, keywords={json.dumps(keywords)})"
            else:
                return "Thought: User is asking for recommendations but didn't specify details. I should ask for more information.\nSpeak: What kind of products are you looking for? Or do you have a specific category in mind?"

        # Tool call for product details
        if "details about" in prompt_lower or "tell me more about" in prompt_lower:
            product_name_match = re.search(r"(?:details about|tell me more about) (.+?)(?:$|\.)", prompt_lower)
            if product_name_match:
                product_name = product_name_match.group(1).strip()
                return f"Thought: User is asking for product details. I should use the `get_product_details` tool.\nAction: call_tool(get_product_details, product_name={json.dumps(product_name)})"
            else:
                return "Thought: User wants product details but didn't specify the product. I should ask for clarification.\nSpeak: Which product are you interested in?"

        # General conversational responses or memory updates
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Speak: Hello! How can I help you with your shopping today?"
        elif "my name is" in prompt_lower:
            name_match = re.search(r"my name is (\w+)", prompt_lower)
            if name_match:
                self.memory_manager.update_user_profile("name", name_match.group(1))
                return f"Speak: Nice to meet you, {name_match.group(1)}!"
        elif "i like" in prompt_lower or "i prefer" in prompt_lower:
            preference_match = re.search(r"i (?:like|prefer) (.+?)(?:$|\.)", prompt_lower)
            if preference_match:
                self.memory_manager.update_user_profile("preferences", self.memory_manager.get_user_profile().get("preferences", []) + [preference_match.group(1).strip()])
                return f"Speak: Noted! I'll keep that in mind."

        # Default response if no specific action or tool call
        return "Speak: I'm not sure how to help with that. Could you please rephrase or ask about products?"

    def _build_llm_prompt(self, user_input):
        context = self.memory_manager.get_context_for_llm()
        
        # Define available tools and their usage for the LLM
        tool_definitions = [
            "Tool: get_product_recommendations(category: str = None, keywords: list[str] = None, limit: int = 3) -> list[dict] - Returns a list of product recommendations based on category and keywords.",
            "Tool: get_product_details(product_name: str) -> dict - Returns detailed information about a specific product."
        ]
        
        prompt = f"""
        You are an E-commerce Conversational Shopping Assistant. Your goal is to help users find products and get information.
        You have access to the following tools:
        {'' .join(tool_definitions)}

        To use a tool, respond in the format:
        Thought: [Your thought process here]
        Action: call_tool(tool_name, arg1=value1, arg2=value2, ...)
        
        If you want to respond directly to the user, use the format:
        Thought: [Your thought process here]
        Speak: [Your response to the user]

        Current conversation and user profile:
        {context}

        User: {user_input}
        Assistant: """
        return prompt

    def process_user_input(self, user_input):
        self.memory_manager.add_message("User", user_input)
        
        llm_prompt = self._build_llm_prompt(user_input)
        print(f"\n[LLM PROMPT]\n{llm_prompt}") # For debugging
        
        llm_raw_response = self.llm_model(llm_prompt)
        print(f"\n[LLM RAW RESPONSE]\n{llm_raw_response}") # For debugging

        assistant_response = ""
        tool_output = None

        # Attempt to parse tool action
        action_match = re.search(r"Action: call_tool\((.*?)\)", llm_raw_response, re.DOTALL)
        
        if action_match:
            action_str = action_match.group(1)
            try:
                # Safely evaluate the action string to extract tool name and arguments
                # Using ast.literal_eval is safer but for this simulation, direct parsing is used.
                # In a real scenario, a more robust parsing (e.g., using a library or a structured output parser) would be needed.
                tool_name_match = re.match(r"(\w+)\(.*?\)", action_str)
                if tool_name_match:
                    tool_name = tool_name_match.group(1)
                    args_str = action_str[len(tool_name) + 1 : -1] # Extract args part
                    
                    args = {}
                    # Simple regex to extract key-value pairs. Needs improvement for complex cases.
                    arg_pairs = re.findall(r'(\w+)=(.*?)(?:, |$)', args_str)
                    for key, val_str in arg_pairs:
                        try:
                            # Attempt to parse common types
                            if val_str.lower() == 'true': args[key] = True
                            elif val_str.lower() == 'false': args[key] = False
                            elif val_str.lower() == 'none': args[key] = None
                            elif val_str.startswith('[') and val_str.endswith(']'): args[key] = json.loads(val_str)
                            elif val_str.startswith('"') and val_str.endswith('"'): args[key] = json.loads(val_str)
                            else: args[key] = json.loads(val_str) # Try general JSON load
                        except json.JSONDecodeError:
                            args[key] = val_str.strip('"') # Fallback if not valid JSON

                    if hasattr(self.recommendation_tools, tool_name):
                        tool_func = getattr(self.recommendation_tools, tool_name)
                        tool_output = tool_func(**args)
                        
                        # Now, integrate tool output back into a conversational response
                        if tool_name == "get_product_recommendations":
                            if tool_output:
                                products_str = ", ".join([p["name"] for p in tool_output])
                                assistant_response = f"Here are some recommendations for you: {products_str}. Is there anything specific you'd like details on?"
                            else:
                                assistant_response = "I couldn't find any recommendations based on your request. Can you try different criteria?"
                        elif tool_name == "get_product_details":
                            if tool_output:
                                assistant_response = f"Sure, here are the details for {tool_output['name']}:\nCategory: {tool_output['category']}\nPrice: ${tool_output['price']}\nDescription: {tool_output['description']}\nFeatures: {', '.join(tool_output['features'])}"
                            else:
                                assistant_response = f"I couldn't find details for that product. Please ensure the product name is correct."
                    else:
                        assistant_response = "I tried to use a tool, but it seems I encountered an unknown tool or an error. Please try again."
                else:
                    assistant_response = "I tried to call a tool, but I couldn't understand the tool name. Please try again."
            except Exception as e:
                print(f"Error executing tool: {e}")
                assistant_response = f"An error occurred while processing your request: {e}."
        else:
            # If no tool action, extract the 'Speak' part directly
            speak_match = re.search(r"Speak: (.+)", llm_raw_response, re.DOTALL)
            if speak_match:
                assistant_response = speak_match.group(1).strip()
            else:
                assistant_response = "I'm sorry, I couldn't understand your request fully. Could you please rephrase?"

        self.memory_manager.add_message("Assistant", assistant_response)
        return assistant_response

if __name__ == "__main__":
    dialogue_agent = LLMDialogueModule()

    print("Welcome to the E-commerce Shopping Assistant! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Assistant: Goodbye!")
            break
        
        response = dialogue_agent.process_user_input(user_input)
        print(f"Assistant: {response}")

        # Optional: Print memory for debugging
        # print("\n--- Current Memory ---")
        # print("History:", dialogue_agent.memory_manager.get_conversation_history())
        # print("Profile:", dialogue_agent.memory_manager.get_user_profile())
        # print("----------------------\n")
