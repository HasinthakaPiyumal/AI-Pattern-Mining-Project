from fastapi import FastAPI
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

class RereadingChatbot:
    def _is_complex(self, query: str) -> bool:
        # Simple heuristic for demonstration: consider a query complex if it contains multiple question marks
        # or certain keywords like "explain" or "how to"
        return query.count('?') > 1 or "explain" in query.lower() or "how to" in query.lower() or len(query.split()) > 15

    def _simulate_llm_response(self, prompt: str) -> str:
        # In a real application, this would interact with an actual LLM API
        if "Read the question again:" in prompt:
            original_query = prompt.replace("Read the question again: ", "")
            return f"[LLM Processed after rereading '{original_query}']: I understand your complex query and here is my detailed answer. {original_query.upper()} has been carefully considered, leading to this precise response."
        else:
            return f"[LLM Processed]: I received your query '{prompt}' and here is my straightforward answer."

    def process_query(self, customer_query: str) -> str:
        if self._is_complex(customer_query):
            print(f"Detected complex query: '{customer_query}'. Applying Rereading pattern.")
            reread_prompt = f"Read the question again: {customer_query}"
            llm_response = self._simulate_llm_response(reread_prompt)
        else:
            print(f"Detected simple query: '{customer_query}'. Processing directly.")
            llm_response = self._simulate_llm_response(customer_query)
        return llm_response

app = FastAPI()
chatbot = RereadingChatbot()

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: QueryRequest):
    response_text = chatbot.process_query(request.query)
    return ChatResponse(response=response_text)