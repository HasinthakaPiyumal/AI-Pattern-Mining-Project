import os
from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, Tool
from product_database import ProductDatabase
from recommendation_engine import RecommendationEngine

# Make sure to set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

class LLMPersonalShopper:
    def __init__(self, openai_api_key: str, temperature: float = 0.7, model_name: str = "text-davinci-003", memory_window_size: int = 5):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = OpenAI(temperature=temperature, model_name=model_name)
        self.product_db = ProductDatabase()
        self.recommendation_engine = RecommendationEngine(self.product_db)
        
        # Short-term memory for current conversation context
        self.conversation_memory = ConversationBufferWindowMemory(k=memory_window_size, memory_key="chat_history", return_messages=True)

        # Long-term memory for user profile and extracted facts
        self.user_profile: Dict[str, Any] = {"preferences": {}, "past_interactions": []}

        self.tools = self._initialize_tools()
        self.agent_chain = self._initialize_agent()

    def _initialize_tools(self) -> List[Tool]:
        """
        Initializes the tools (ProductDatabase and RecommendationEngine) for the LLM agent.
        """
        tools = [
            Tool(
                name="Product Search",
                func=self.product_db.search_products,
                description=(
                    "Useful for searching products based on keywords, category, gender, price range, or color. "
                    "Input should be a dictionary with keys like 'query', 'category', 'gender', 'min_price', 'max_price', 'color'. "
                    "Example: {'query': 'running shoes', 'gender': 'women', 'max_price': 100}"
                )
            ),
            Tool(
                name="Product Details",
                func=self.product_db.get_product_details,
                description=(
                    "Useful for getting detailed information about a specific product using its product ID. "
                    "Input should be a string representing the product ID. Example: 'P001'"
                )
            ),
            Tool(
                name="Get Recommendations",
                func=self.recommendation_engine.get_recommendations,
                description=(
                    "Useful for getting personalized product recommendations based on user preferences. "
                    "Input should be a dictionary with keys like 'category', 'brand', 'price_range'. "
                    "Example: {'category': 'Running Shoes', 'price_range': 'under $100'}"
                )
            ),
            Tool(
                name="Get Related Products",
                func=self.recommendation_engine.get_related_products,
                description=(
                    "Useful for finding products related to a specific product by its ID. "
                    "Input should be a string representing the product ID. Example: 'P001'"
                )
            )
        ]
        return tools

    def _initialize_agent(self):
        """
        Initializes the Langchain agent with the LLM, tools, and memory.
        """
        agent_kwargs = {
            "extra_prompt_messages": [
                ("system", "You are a helpful and friendly e-commerce personal shopper AI. "
                           "Your goal is to understand user preferences, recommend products, "
                           "and assist with their shopping needs. "
                           "Always try to be specific and helpful."
                )
            ]
        }

        agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.conversation_memory,
            agent_kwargs=agent_kwargs,
            handle_parsing_errors=True # For robust error handling
        )
        return agent

    def _update_user_profile(self, user_input: str, agent_response: str):
        """
        A simple method to simulate updating long-term user profile based on conversation.
        In a real system, this would involve LLM extraction or more sophisticated parsing.
        """
        # This is a very basic example. A real system would use another LLM call
        # or NLP techniques to extract facts and update the profile.
        if "likes" in user_input.lower():
            # Example: "I like blue items"
            if "blue" in user_input.lower():
                self.user_profile["preferences"]["color"] = "blue"
            # More sophisticated extraction would be needed here
        
        # Store past interactions for potential future context retrieval
        self.user_profile["past_interactions"].append({"user": user_input, "agent": agent_response})

        print(f"[DEBUG] User profile updated: {self.user_profile}")

    def retrieve_user_context(self, current_query: str) -> str:
        """
        Simulates retrieving relevant user facts from long-term memory to augment the current query.
        In a real system, this would involve embedding and vector search.
        """
        context_facts = []
        if self.user_profile["preferences"]:
            context_facts.append(f"User preferences: {self.user_profile['preferences']}")
        
        # A simple check for similar past interactions (could be enhanced with embeddings)
        for interaction in self.user_profile["past_interactions"][-3:]: # Check last few interactions
            if current_query.lower() in interaction["user"].lower() or \
               any(keyword in interaction["user"].lower() for keyword in current_query.lower().split()):
                context_facts.append(f"Past interaction: User asked '{interaction['user']}', Agent responded '{interaction['agent']}'")
        
        if context_facts:
            return "\n" + "\n".join(context_facts)
        return ""

    def chat(self, user_input: str) -> str:
        """
        Processes a user input and returns the agent's response.
        """
        # Augment user input with retrieved long-term context
        retrieved_context = self.retrieve_user_context(user_input)
        augmented_user_input = f"User's current query: {user_input}{retrieved_context}"
        
        try:
            response = self.agent_chain.run(input=augmented_user_input)
            self._update_user_profile(user_input, response) # Update long-term memory
            return response
        except Exception as e:
            print(f"An error occurred during chat: {e}")
            return "I apologize, but I encountered an issue. Could you please rephrase or try again?"

if __name__ == "__main__":
    # Replace with your actual OpenAI API key or set it as an environment variable
    # Example: os.environ["OPENAI_API_KEY"] = "sk-..."
    openai_api_key = os.getenv("OPENAI_API_KEY") 

    if not openai_api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        shopper_agent = LLMPersonalShopper(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        print("Hello! I'm your personal shopping assistant. How can I help you today?")
        
        while True:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit", "bye"]:
                print("Shopper AI: Goodbye! Happy shopping!")
                break
            
            agent_response = shopper_agent.chat(user_query)
            print(f"Shopper AI: {agent_response}")
