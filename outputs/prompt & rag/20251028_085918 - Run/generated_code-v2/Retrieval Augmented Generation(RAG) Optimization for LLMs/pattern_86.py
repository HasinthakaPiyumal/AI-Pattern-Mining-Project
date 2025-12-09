class KnowledgeBase:
    def __init__(self):
        self.documents = {
            "FAQ_General_Returns": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be in original condition.",
            "FAQ_Warranty_Policy": "All our electronics come with a 1-year limited warranty covering manufacturing defects. Extended warranties are available for purchase.",
            "Product_Manual_X100": "The X100 model features a 12MP camera and 4GB RAM. Troubleshooting error code E101: Check power supply. Error code E202: Reinstall software.",
            "Product_Manual_Z200": "The Z200 model features a 16MP camera and 6GB RAM. Troubleshooting error code E101: Check battery connection. Error code E303: Update firmware.",
            "Service_Log_X100_E101": "Common fix for X100 E101 error is to reseat the battery or use a different power adapter.",
            "External_Vendor_Guide_Battery": "Guide to common battery issues across various devices. Symptoms of failing battery include rapid discharge and device shutdown.",
            "Firmware_Update_Guide_Z200": "Steps to update firmware for Z200 model: Download from official site, connect via USB, run update tool."
        }

    def get_document(self, doc_id):
        return self.documents.get(doc_id)

    def search_documents(self, keywords):
        results = []
        for doc_id, content in self.documents.items():
            if any(keyword.lower() in doc_id.lower() or keyword.lower() in content.lower() for keyword in keywords):
                results.append((doc_id, content))
        return results


class Retriever:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def search(self, query_keywords):
        """
        Searches the knowledge base for documents relevant to the query keywords.
        Returns a list of (document_title, document_content) tuples.
        """
        if not query_keywords:
            return []
        print(f"  Retriever searching for keywords: {', '.join(query_keywords)}")
        return self.knowledge_base.search_documents(query_keywords)


class MockLLM:
    def generate(self, prompt: str) -> str:
        """
        A mock LLM that provides predefined responses or simple keyword-based generation.
        In a real scenario, this would be an API call to an actual LLM.
        """
        print(f"  LLM received prompt (first 200 chars): {prompt[:200]}...")

        # Simple logic to simulate iterative RAG behavior
        prompt_lower = prompt.lower()

        if "warranty" in prompt_lower and "1-year limited warranty" in prompt_lower:
            return "Based on the warranty policy, your device is covered by a 1-year limited warranty for manufacturing defects."
        elif "error code e101" in prompt_lower and "x100" in prompt_lower and "power supply" in prompt_lower:
            return "For error code E101 on an X100 model, you should check the power supply or reseat the battery. Do you need more specific troubleshooting steps?"
        elif "error code e101" in prompt_lower and "z200" in prompt_lower and "battery connection" in prompt_lower:
            return "For error code E101 on a Z200 model, ensure the battery connection is secure. Do you need more info on Z200 troubleshooting?"
        elif "return policy" in prompt_lower and "30 days" in prompt_lower:
            return "Our standard return policy allows returns within 30 days of purchase with the original receipt and condition."
        elif "x100" in prompt_lower and ("problem" in prompt_lower or "issue" in prompt_lower) and "more details" not in prompt_lower:
             return "I understand you have an issue with your X100. Could you please provide more details, such as any specific error codes or symptoms?"
        elif "z200" in prompt_lower and "update firmware" in prompt_lower:
            return "To update the firmware for your Z200, please download the latest version from our official website and follow the update tool instructions."
        elif "e303" in prompt_lower and "z200" in prompt_lower and "update firmware" in prompt_lower:
            return "Error E303 on a Z200 typically indicates a firmware issue. You should try updating the firmware according to the guide."
        elif "specific model" in prompt_lower:
            return "I need a specific model to give you more tailored troubleshooting."
        else:
            return "I\'m processing your request. Please bear with me. Is there a specific product or error code you are referring to? (more info needed)"


class CustomerSupportAgent:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.retriever = Retriever(self.knowledge_base)
        self.llm = MockLLM()
        self.conversation_history = []

    def _identify_info_needs(self, current_conversation_history, llm_response):
        """
        Simulates the LLM or a controller identifying if more information is needed
        and what to search for based on the current context and LLM\'s partial response.
        """
        llm_response_lower = llm_response.lower()
        if "more details" in llm_response_lower or "unclear" in llm_response_lower or "more info needed" in llm_response_lower or "specific model" in llm_response_lower:
            last_user_query = ""
            for role, text in reversed(current_conversation_history):
                if role == "user":
                    last_user_query = text
                    break
            
            keywords = []
            if "specific model" in llm_response_lower:
                keywords.append("product manual") # Broaden search for model info

            if last_user_query:
                # Simple keyword extraction for demo
                keywords.extend([word for word in last_user_query.lower().split() if len(word) > 3 and word not in ["the", "a", "is", "of", "how", "what", "problem", "issue", "my", "i", "can", "you", "help", "me", "with"]])

            # If LLM explicitly asks for error code or model, add those as keywords
            if "error code" in llm_response_lower:
                keywords.append("error code")
            
            # Remove duplicates and ensure some keywords exist
            keywords = list(set(keywords))
            if keywords:
                return True, keywords
        return False, []

    def _augment_context(self, current_context, retrieved_docs):
        """Augments the current context with newly retrieved documents."""
        if not retrieved_docs:
            return current_context

        new_context = current_context + "\n\n--- Additional Information ---\n"
        for doc_title, doc_content in retrieved_docs:
            new_context += f"Document: {doc_title}\nContent: {doc_content}\n"
        return new_context

    def process_query(self, user_query, max_iterations=3):
        self.conversation_history.append(("user", user_query))
        current_context = f"User Query: {user_query}\n"
        full_response = ""
        
        print(f"\n--- Processing new query: \"{user_query}\" ---")

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")
            
            # Step 1: Determine search keywords for this iteration
            search_keywords = []
            if iteration == 0:
                # Initial search based on user query
                search_keywords = [word for word in user_query.lower().split() if len(word) > 2]
            else:
                # Subsequent searches based on LLM's identified needs
                needs_more_info, keywords_from_llm = self._identify_info_needs(self.conversation_history, full_response)
                if needs_more_info:
                    search_keywords = keywords_from_llm
                else:
                    print("  LLM indicates no further information is immediately needed for this turn.")
                    break # Stop iterative retrieval if LLM doesn't need more info
            
            # Step 2: Retrieval
            newly_retrieved_docs = []
            if search_keywords:
                newly_retrieved_docs = self.retriever.search(search_keywords)
                if newly_retrieved_docs:
                    print(f"  Retrieved {len(newly_retrieved_docs)} new documents.")
                    current_context = self._augment_context(current_context, newly_retrieved_docs)
                else:
                    print(f"  No new documents retrieved for keywords: {', '.join(search_keywords)}.")
            else:
                print("  No keywords to search for in this iteration.")

            # Step 3: LLM Generation
            print("  Generating response with LLM...")
            llm_input = f"Conversation History:\n{self._format_conversation_history()}\n\nCurrent Context:\n{current_context}\n\nAgent Partial Response:"
            llm_partial_response = self.llm.generate(llm_input)
            full_response = llm_partial_response # Only keep the latest partial response for iterative decision
            self.conversation_history.append(("agent_partial", llm_partial_response))
            print(f"  Agent's partial response: {llm_partial_response[:100]}...")

        # Final Generation based on accumulated context and last partial response
        final_llm_input = f"Based on the following conversation and the final context provided, give a comprehensive and helpful answer to the user's last query.\nConversation History:\n{self._format_conversation_history()}\n\nFinal Context Used:\n{current_context}\n\nAgent Final Answer:"
        final_response = self.llm.generate(final_llm_input)
        self.conversation_history.append(("agent", final_response))
        return final_response

    def _format_conversation_history(self):
        formatted_history = ""
        for role, text in self.conversation_history:
            formatted_history += f"{role.capitalize()}: {text}\n"
        return formatted_history


# --- Main execution for demonstration ---
if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("--- Welcome to the Iterative RAG Customer Support Agent ---")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break

        response = agent.process_query(user_input)
        print(f"\nAgent Final Response: {response}")
