import random

class MockLLMClient:
    def create_chat_completion(self, model, messages, temperature=0.7):
        response_map = {
            "Tech Enthusiast": [
                "This product features a cutting-edge [specific tech detail] for optimal performance.",
                "You'll appreciate the [feature] which utilizes [technology] for incredible efficiency.",
                "Troubleshooting steps usually involve checking [component A] and ensuring [component B] is properly configured."
            ],
            "Empathetic Resolution Specialist": [
                "I understand this is frustrating. Let me see how we can resolve this for you.",
                "I apologize for the inconvenience you've experienced. We're here to help.",
                "Please bear with me, I'm looking into the best solution to address your concern."
            ],
            "Concise Information Specialist": [
                "The answer is: [Direct Answer].",
                "Here's what you need to know: [Key Point 1], [Key Point 2].",
                "Refer to section [X] of the manual for details on [topic]."
            ],
            "Helpful Advisor": [
                "Based on your interest, you might also find [related product/service] beneficial.",
                "To enhance your experience, consider upgrading to our premium [feature].",
                "Many customers who enjoyed [current product] also love [complementary product/service]."
            ]
        }
        
        system_message = messages[0]['content']
        user_message = messages[1]['content']

        persona_key = ""
        for persona, responses in response_map.items():
            if persona in system_message: # A simple way to extract the persona from the system prompt
                persona_key = persona
                break
        
        if persona_key and persona_key in response_map:
            response_content = random.choice(response_map[persona_key])
        else:
            response_content = "I'm a general support agent ready to assist you."
            
        return {"choices": [{"message": {"content": response_content}}]}

class CustomerSupportAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.roles = {
            "Tech Enthusiast": {
                "description": "a knowledgeable tech expert who loves to explain product specifications and troubleshooting steps in detail",
                "keywords": ["specifications", "troubleshoot", "performance", "technical", "problem", "fix"]
            },
            "Empathetic Resolution Specialist": {
                "description": "a kind and understanding customer service representative focused on resolving issues and apologizing for inconveniences",
                "keywords": ["complaint", "issue", "unhappy", "problem", "refund", "return", "apology"]
            },
            "Concise Information Specialist": {
                "description": "a direct and efficient information provider, giving clear and brief answers to common questions",
                "keywords": ["what is", "how to", "faq", "information", "explain", "definition"]
            },
            "Helpful Advisor": {
                "description": "a friendly advisor who helps customers discover new products or features that might interest them",
                "keywords": ["recommend", "suggest", "upgrade", "more features", "better experience"]
            }
        }
        self.base_system_instruction = "You are a customer support agent."

    def _select_persona(self, query):
        query_lower = query.lower()
        for role_name, role_info in self.roles.items():
            for keyword in role_info["keywords"]:
                if keyword in query_lower:
                    return role_name, role_info["description"]
        return "Concise Information Specialist", self.roles["Concise Information Specialist"]["description"] # Default to concise if no match

    def _construct_prompt(self, persona_description, user_query):
        system_prompt = f"Pretend you are {persona_description}. {self.base_system_instruction}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

    def handle_query(self, user_query):
        persona_name, persona_description = self._select_persona(user_query)
        print(f"[DEBUG] Selected Persona: {persona_name}")
        prompt_messages = self._construct_prompt(persona_description, user_query)

        # Simulate LLM interaction
        response = self.llm_client.create_chat_completion(
            model="gpt-3.5-turbo", 
            messages=prompt_messages,
            temperature=0.7
        )
        
        return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    mock_llm = MockLLMClient()
    agent = CustomerSupportAgent(mock_llm)

    print("Welcome to the Dynamic Customer Support. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = agent.handle_query(user_input)
        print(f"Agent: {response}")
