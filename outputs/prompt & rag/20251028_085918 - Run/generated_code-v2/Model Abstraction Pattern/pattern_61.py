import os
from dotenv import load_dotenv
from pydantic import BaseSettings, Field
from typing import Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    huggingface_api_token: str = Field("", env="HF_API_TOKEN")
    huggingface_model_name: str = Field("google/flan-t5-xxl", env="HF_MODEL_NAME")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

load_dotenv()
settings = Settings()

llm_openai = ChatOpenAI(api_key=settings.openai_api_key, model="gpt-3.5-turbo", temperature=0.7)
llm_gemini = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=settings.gemini_api_key, temperature=0.7)

llm_huggingface_base = HuggingFaceHub(
    repo_id=settings.huggingface_model_name,
    huggingfacehub_api_token=settings.huggingface_api_token,
    model_kwargs={"temperature": 0.7, "max_length": 500}
)
llm_huggingface = ChatHuggingFace(llm=llm_huggingface_base)

class LLMRouter:
    def __init__(self, llm_providers: Dict[str, Any]):
        self.llm_providers = llm_providers

    def route_llm(self, query: str, routing_policy: Literal["cost_effective", "high_performance", "complexity_based"] = "cost_effective") -> Any:
        if routing_policy == "cost_effective":
            if "summarize" in query.lower() or "explain" in query.lower() or "translate" in query.lower():
                return self.llm_providers.get("gemini", self.llm_providers["openai"])
            return self.llm_providers.get("openai", self.llm_providers["openai"])

        elif routing_policy == "high_performance":
            return self.llm_providers.get("openai", self.llm_providers["openai"])

        elif routing_policy == "complexity_based":
            if len(query.split()) < 8 and "simple" in query.lower():
                return self.llm_providers.get("huggingface", self.llm_providers["openai"])
            return self.llm_providers.get("openai", self.llm_providers["openai"])

        return self.llm_providers.get("openai", self.llm_providers["openai"])

class CustomerSupportAgent:
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="history")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI customer support assistant. Provide helpful and concise answers. Keep the conversation flowing."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])

    def get_response(self, query: str, routing_policy: Literal["cost_effective", "high_performance", "complexity_based"] = "cost_effective") -> str:
        llm_model = self.llm_router.route_llm(query, routing_policy)

        chain = LLMChain(
            llm=llm_model,
            prompt=self.prompt,
            memory=self.memory,
            verbose=False
        )

        try:
            response_data = chain.invoke({"query": query})
            if isinstance(response_data, dict) and "text" in response_data:
                response_text = response_data["text"]
            elif isinstance(response_data, BaseMessage):
                 response_text = response_data.content
            else:
                 response_text = str(response_data)

            return response_text
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to get response from LLM.")

app = FastAPI(title="AI Customer Support Assistant")

llm_providers_map = {
    "openai": llm_openai,
    "gemini": llm_gemini,
    "huggingface": llm_huggingface,
}
llm_router_instance = LLMRouter(llm_providers=llm_providers_map)
customer_agent = CustomerSupportAgent(llm_router=llm_router_instance)

@app.post("/chat")
async def chat_with_assistant(query: str, routing_policy: Literal["cost_effective", "high_performance", "complexity_based"] = "cost_effective"):
    try:
        response = customer_agent.get_response(query, routing_policy)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred.")