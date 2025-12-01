import uuid

# This class simulates a simple vector database for demonstration purposes.
# In a real application, you would use a dedicated library like ChromaDB, Pinecone, or FAISS
# for efficient storage and semantic retrieval of embeddings.
class SimpleVectorDB:
    def __init__(self):
        self.documents = []  # Stores (id, text, metadata)
        self.id_map = {}     # Maps document ID to its original text

    def add_document(self, text, metadata=None):
        doc_id = str(uuid.uuid4())
        self.documents.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        self.id_map[doc_id] = text
        # In a real vector DB, you would generate embeddings for 'text' here
        # and store them alongside the document ID for vector similarity search.
        return doc_id

    def retrieve_similar(self, query_text, top_k=2):
        # A highly simplified retrieval mechanism: it tries to find documents
        # that contain keywords from the query or just returns recent interactions.
        # A real vector database would perform semantic similarity search using embeddings.
        print(f"DEBUG: Attempting to retrieve memories for query: '{query_text}'")
        retrieved_texts = []
        query_words = set(query_text.lower().split())

        # Prioritize recent documents that contain keywords from the query
        for doc in reversed(self.documents):
            doc_words = set(doc["text"].lower().split())
            if any(word in doc_words for word in query_words if len(word) > 2): # Avoid very short common words
                retrieved_texts.append(doc["text"])
                if len(retrieved_texts) >= top_k:
                    break

        # If not enough keyword matches, add the most recent general interactions
        if len(retrieved_texts) < top_k:
            for doc in reversed(self.documents):
                if doc["text"] not in retrieved_texts: # Avoid duplicates
                    retrieved_texts.append(doc["text"])
                    if len(retrieved_texts) >= top_k:
                        break

        return retrieved_texts


# Manages interaction with the external memory (SimpleVectorDB).
class MemoryManager:
    def __init__(self):
        self.db = SimpleVectorDB()

    def add_memory(self, type, content, metadata=None):
        doc_text = f"[{type.upper()}] {content}"
        return self.db.add_document(doc_text, metadata={"type": type, **(metadata or {})})

    def retrieve_relevant_memories(self, query, top_k=3):
        return self.db.retrieve_similar(query, top_k)


# This class simulates an LLM for demonstration purposes.
# In a real application, you would integrate with an actual LLM API (e.g., OpenAI, Google Gemini).
class LLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        # For a real application:
        # import openai
        # self.client = openai.OpenAI()

    def generate_response(self, prompt, temperature=0.7):
        print(f"\nDEBUG: LLM is processing the following prompt:\n---\n{prompt}\n---")
        # Simulate LLM response based on keywords and memory content
        prompt_lower = prompt.lower()

        if "medication" in prompt_lower and "forget" in prompt_lower:
            return "Based on our past discussions and your history of occasionally forgetting evening medication, I suggest we set up a consistent reminder schedule. How about daily reminders at 7 PM? We can also explore setting up a pill organizer to make it easier."
        elif "diet" in prompt_lower and "vegetable intake" in prompt_lower:
             return "Considering your goal of increasing vegetable intake and your feedback about time for meal prep, how about we focus on simple, quick additions? Perhaps pre-washed greens for salads or frozen vegetables that can be easily steamed or roasted with meals. What do you think about adding a daily serving of spinach to your breakfast smoothie?"
        elif "exercise" in prompt_lower and "knee pain" in prompt_lower:
             return "Given your past experience with knee pain during high-impact activities, let's prioritize joint-friendly options. Swimming, cycling, or even gentle yoga are excellent choices that allow you to stay active without putting excessive strain on your knees. Remember your preference for low-impact exercises in the past. We can also explore specific strengthening exercises for the muscles around your knees."
        elif "past progress" in prompt_lower:
             return "Looking back at your recent records, you've shown great consistency! You successfully incorporated a 15-minute walk into your daily routine for the past two weeks, and you've been diligently tracking your meals, which is a fantastic habit. Keep up this excellent momentum!"
        elif "stress reduction" in prompt_lower and "mindfulness" in prompt_lower:
             return "Indeed, mindfulness exercises are a great tool for stress reduction, as we discussed previously. Are you looking for new techniques, or would you like to revisit some of the guided meditations we tried before? Consistency is key here!"
        
        return f"Hello, I am your AI health coach. Based on the information provided and what I've learned from our past interactions, my suggestion is: [LLM will provide a tailored response here. For example, it might offer general advice or ask for more specifics related to '{prompt_lower[:80]}...']"


# The main class orchestrating the AI Health Coach functionality.
class AIHealthCoach:
    def __init__(self, llm_model="gpt-3.5-turbo"):
        self.memory_manager = MemoryManager()
        self.llm = LLM(llm_model)
        self.user_profile = {}  # Could store more persistent, structured user data here

    def _format_prompt(self, user_query, relevant_memories):
        context_str = "\n".join([f"- {mem}" for mem in relevant_memories])
        
        if context_str:
            prompt = (
                f"You are a highly personalized and adaptive AI health coach. "
                f"Your goal is to provide continuous, context-aware, and empathetic health guidance. "
                f"You must leverage ALL relevant past health data, preferences, and interactions (provided below as 'Memory') "
                f"to inform your response and tailor it specifically to the user's history and current needs. "
                f"Always strive to learn and adapt based on past successes and failures described in the memory.\n\n"
                f"Memory of Past Interactions and Health Data:\n{context_str}\n\n"
                f"Current User Query: {user_query}\n\n"
                f"Your Personalized and Context-Aware Response:"
            )
        else:
            # If no relevant memories are found, provide a more general but still helpful response.
            prompt = (
                f"You are a highly personalized and adaptive AI health coach. "
                f"Your goal is to provide continuous, context-aware, and empathetic health guidance. "
                f"Since there's no specific past memory retrieved for this query, provide general helpful advice "
                f"and also ask clarifying questions to build up context for future interactions.\n\n"
                f"Current User Query: {user_query}\n\n"
                f"Your Personalized and Context-Aware Response:"
            )
        return prompt

    def interact(self, user_query):
        print(f"\n--- User Input: '{user_query}' ---")
        # 1. Retrieve relevant past memories from the external memory
        relevant_memories = self.memory_manager.retrieve_relevant_memories(user_query)
        print(f"DEBUG: Retrieved {len(relevant_memories)} relevant memories.")

        # 2. Format the prompt for the LLM, including retrieved memories
        prompt = self._format_prompt(user_query, relevant_memories)

        # 3. Get the personalized response from the LLM
        llm_response = self.llm.generate_response(prompt)

        # 4. Store the current interaction in memory for future context and learning
        self.memory_manager.add_memory("interaction", f"User: {user_query} | Coach: {llm_response}",
                                       metadata={"query": user_query, "response": llm_response})

        return llm_response


# Main execution block to demonstrate the AI Health Coach
if __name__ == "__main__":
    coach = AIHealthCoach()

    print("\n====================================")
    print("  Initializing Personalized AI Health Coach")
    print("====================================")

    # Simulate adding some initial user health data, past interactions, and feedback to memory.
    # This data will be leveraged by the coach for personalized responses.
    coach.memory_manager.add_memory("HEALTH_DATA", "User has a history of mild knee pain during high-impact exercises.")
    coach.memory_manager.add_memory("GOAL", "User aims to increase daily vegetable intake from 2 to 5 servings.")
    coach.memory_manager.add_memory("BEHAVIOR", "User sometimes forgets evening medication, prefers digital reminders.")
    coach.memory_manager.add_memory("PREFERENCE", "User preferred low-impact exercises like swimming and cycling in the past.")
    coach.memory_manager.add_memory("PROGRESS_NOTE", "User successfully walked 15 minutes daily for the last week, showing good consistency.")
    coach.memory_manager.add_memory("FEEDBACK", "User found last week's complex meal prep plan challenging due to lack of time.")
    coach.memory_manager.add_memory("INTERACTION", "User asked about stress reduction. Coach suggested mindfulness exercises and provided resources.",
                                   metadata={"query": "stress reduction", "response": "mindfulness"})
    coach.memory_manager.add_memory("GOAL", "User is working on improving sleep quality by establishing a consistent bedtime.")
    coach.memory_manager.add_memory("HEALTH_DATA", "User reported mild allergies to peanuts.")

    print("\n--- Starting Personalized Coaching Session ---")

    # --- Interaction 1: Medication reminder, leveraging memory about forgetting ---
    user_input_1 = "I'm having trouble remembering to take my evening medication. What should I do?"
    response_1 = coach.interact(user_input_1)
    print(f"Coach: {response_1}")

    # --- Interaction 2: Diet improvement, leveraging memory about vegetable intake goal and past feedback ---
    user_input_2 = "I want to improve my diet for lunch, focusing on healthy additions. Any suggestions, keeping my past goals and challenges in mind?"
    response_2 = coach.interact(user_input_2)
    print(f"Coach: {response_2}")

    # --- Interaction 3: Exercise advice, leveraging memory about knee pain and preferred activities ---
    user_input_3 = "What kind of exercises should I do? I want to be active but avoid aggravating old injuries like my knees."
    response_3 = coach.interact(user_input_3)
    print(f"Coach: {response_3}")

    # --- Interaction 4: Progress update, leveraging recent progress notes ---
    user_input_4 = "How have I been doing recently with my health goals? Tell me about my past progress."
    response_4 = coach.interact(user_input_4)
    print(f"Coach: {response_4}")

    # --- Interaction 5: New query, expecting the coach to adapt and ask clarifying questions if no direct memory ---
    user_input_5 = "I'm feeling a bit low on energy today. Any quick tips?"
    response_5 = coach.interact(user_input_5)
    print(f"Coach: {response_5}")

    # --- Interaction 6: Follow-up on a previous topic, demonstrating continuity ---
    user_input_6 = "Can we talk more about those mindfulness exercises for stress reduction?"
    response_6 = coach.interact(user_input_6)
    print(f"Coach: {response_6}")

    print("\n====================================")
    print("  Coaching Session Ended")
    print("====================================")