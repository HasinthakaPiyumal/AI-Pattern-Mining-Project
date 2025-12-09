import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
MAX_LLM_INPUT_TOKENS = 4000 # Example, adjust based on actual LLM

# Global variables for chatbot state
llm = None
embeddings_model = None
vectorstore = None
conversation_history = []

def _get_token_count(text: str) -> int:
    # Simple approximation for token count, actual tokenizers are more complex
    return len(text.split()) / 0.75 # Roughly 4 chars per token, 0.75 words per token

def _truncate_context(context_parts: list[str], max_tokens: int) -> str:
    combined_context = " ".join(context_parts)
    if _get_token_count(combined_context) <= max_tokens:
        return combined_context

    truncated_context_parts = []
    current_tokens = 0
    # Prioritize later parts of the context (more recent/relevant)
    for part in reversed(context_parts):
        part_tokens = _get_token_count(part)
        if current_tokens + part_tokens <= max_tokens:
            truncated_context_parts.insert(0, part)
            current_tokens += part_tokens
        else:
            # If even a single part exceeds, just take a portion
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 0:
                words = part.split()
                truncated_words = words[:int(remaining_tokens * 0.75)] # Rough word count
                truncated_context_parts.insert(0, " ".join(truncated_words))
            break
    return " ".join(truncated_context_parts)

def initialize_chatbot():
    global llm, embeddings_model, vectorstore
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, model_name="gpt-3.5-turbo")
    embeddings_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings_model)

def _manage_context(current_query: str, session_history: list[dict], max_tokens: int) -> str:
    # Retrieve relevant documents from ChromaDB based on the current query
    # Using a small k for demonstration, adjust as needed
    retrieved_docs = vectorstore.similarity_search(current_query, k=5)
    
    # Convert retrieved docs to strings
    retrieved_context_parts = [doc.page_content for doc in retrieved_docs]

    # Format session history into context parts
    formatted_session_history_parts = []
    for entry in session_history:
        formatted_session_history_parts.append(f"{entry['role']}: {entry['content']}")

    # Combine all context parts
    # Prioritize recent session history, then retrieved long-term context
    all_context_parts = formatted_session_history_parts + retrieved_context_parts

    # Truncate if necessary
    prepared_context = _truncate_context(all_context_parts, max_tokens)
    return prepared_context

def get_chatbot_response(user_query: str) -> str:
    global conversation_history, llm, vectorstore

    if llm is None or vectorstore is None:
        raise RuntimeError("Chatbot not initialized. Call initialize_chatbot() first.")

    # Manage long context
    managed_context = _manage_context(user_query, conversation_history, MAX_LLM_INPUT_TOKENS - _get_token_count(user_query) - 100) # Reserve tokens for query and response

    # Prepare the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent e-commerce customer support assistant. Use the provided context to answer customer queries accurately and concisely. If the context does not contain the answer, state that you don't know."),
        ("system", "Historical context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Format chat history for LangChain
    formatted_chat_history = []
    for entry in conversation_history:
        if entry['role'] == 'human':
            formatted_chat_history.append(HumanMessage(content=entry['content']))
        elif entry['role'] == 'ai':
            formatted_chat_history.append(AIMessage(content=entry['content']))

    # Create the chain
    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(lambda x: managed_context)) # Pass managed_context to the prompt
        | prompt
        | llm
    )

    # Invoke the chain
    response_message = chain.invoke({
        "input": user_query,
        "chat_history": formatted_chat_history,
        "context": managed_context # Explicitly pass context here for the prompt
    })

    bot_response = response_message.content

    # Store current interaction in session history
    conversation_history.append({"role": "human", "content": user_query})
    conversation_history.append({"role": "ai", "content": bot_response})

    # Store current interaction in ChromaDB for long-term memory
    vectorstore.add_texts(
        texts=[user_query, bot_response],
        metadatas=[
            {"role": "human", "timestamp": "<current_timestamp>"},
            {"role": "ai", "timestamp": "<current_timestamp>"}
        ]
    )
    vectorstore.persist()

    return bot_response

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment variables
    # Example: export OPENAI_API_KEY='your_openai_api_key_here'
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script.")
    else:
        print("Initializing chatbot...")
        initialize_chatbot()
        print("Chatbot initialized. Type 'exit' to quit.")

        # Simulate some initial data in ChromaDB (e.g., user profile info or past issues)
        vectorstore.add_texts(
            texts=[
                "Customer John Doe frequently orders electronics, especially headphones and smartwatches. Has a history of issues with delivery speed.",
                "Last week, customer Jane Smith inquired about the warranty for a 'SuperX' smartphone, order ID #12345.",
                "A common customer issue is difficulty tracking orders, leading to multiple support requests."
            ],
            metadatas=[
                {"type": "user_profile", "user_id": "john_doe"},
                {"type": "past_issue", "user_id": "jane_smith", "product": "SuperX smartphone"},
                {"type": "common_problem"}
            ]
        )
        vectorstore.persist()
        print("Initial long-term memory populated.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break

            try:
                response = get_chatbot_response(user_input)
                print(f"Chatbot: {response}")
            except Exception as e:
                print(f"An error occurred: {e}")
                # Simple error handling for demonstration
