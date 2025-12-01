from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import Dict, Any, List, Union
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Modules Start ---

# modules/llm.py
class MockLLM(BaseChatModel):
    def _generate(self, messages: List[Any], stop: List[str] = None, **kwargs: Any) -> Any:
        # Simple mock logic: if tool_calls exist, acknowledge them, otherwise respond generally.
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            response_content = f"Okay, I'll try to use the tool: {tool_call.name} with args: {tool_call.args}"
        else:
            response_content = f"This is a mock LLM response to: {last_message.content}"
        
        # Simulate a simple AI message for demonstration
        from langchain_core.outputs import ChatGeneration, AIMessage
        return ChatGeneration(message=AIMessage(content=response_content))

    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def invoke(self, input: Union[str, List[Any]], config: Dict[str, Any] = None, **kwargs: Any) -> Any:
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        generation = self._generate(messages, **kwargs)
        return generation.message


# modules/memory.py
class WorkingMemory:
    def __init__(self):
        self.history: Dict[str, List[Union[HumanMessage, AIMessage]]] = {}
        self.context: Dict[str, Dict[str, Any]] = {}

    def add_message(self, session_id: str, message: Union[HumanMessage, AIMessage]):
        if session_id not in self.history:
            self.history[session_id] = []
        self.history[session_id].append(message)

    def get_history(self, session_id: str) -> List[Union[HumanMessage, AIMessage]]:
        return self.history.get(session_id, [])

    def update_context(self, session_id: str, key: str, value: Any):
        if session_id not in self.context:
            self.context[session_id] = {}
        self.context[session_id][key] = value

    def get_context(self, session_id: str) -> Dict[str, Any]:
        return self.context.get(session_id, {})


# modules/knowledge.py
class KnowledgeBase:
    def get_product_details(self, product_id: str) -> str:
        if product_id == "P123":
            return "Product P123: Wireless Headphones, noise-cancelling, 20-hour battery life, $199.99."
        return f"No details found for product ID: {product_id}"

    def search_faq(self, query: str) -> str:
        if "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition."
        return f"No FAQ entry found for query: '{query}'. Please rephrase or contact support."

    def get_policy_info(self, policy_type: str) -> str:
        if policy_type == "shipping":
            return "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days and costs an additional $15."
        return f"No policy information found for type: {policy_type}"


# modules/actions.py
class ActionExecutor:
    def get_order_status(self, order_id: str) -> str:
        # Simulate fetching order status from an e-commerce backend
        if order_id == "ORD456":
            return "Order ORD456: Shipped, expected delivery by end of week."
        return f"Order ID {order_id} not found or invalid."

    def process_refund(self, order_id: str, reason: str) -> str:
        # Simulate processing a refund
        if order_id == "ORD789":
            return f"Refund for order {order_id} for reason '{reason}' initiated successfully. Funds will be returned within 5-7 business days."
        return f"Could not process refund for order ID {order_id}. Invalid order or system error."

    def suggest_related_products(self, product_id: str) -> str:
        if product_id == "P123":
            return "Related products for P123: Bluetooth Speaker, Smartwatch, Phone Case."
        return f"No related products found for {product_id}."


# modules/tools.py - LangChain Tools Definition
def get_langchain_tools(knowledge_base: KnowledgeBase, action_executor: ActionExecutor):
    @tool
    def get_product_details_tool(product_id: str) -> str:
        """Get detailed information about a specific product using its ID."""
        return knowledge_base.get_product_details(product_id)

    @tool
    def search_faq_tool(query: str) -> str:
        """Search the Frequently Asked Questions (FAQ) database for answers related to a specific query."""
        return knowledge_base.search_faq(query)

    @tool
    def get_policy_info_tool(policy_type: str) -> str:
        """Retrieve information about a specific company policy (e.g., shipping, return)."""
        return knowledge_base.get_policy_info(policy_type)

    @tool
    def get_order_status_tool(order_id: str) -> str:
        """Check the current status of a customer order using its order ID."""
        return action_executor.get_order_status(order_id)

    @tool
    def process_refund_tool(order_id: str, reason: str) -> str:
        """Initiate a refund for a given order ID and reason."""
        return action_executor.process_refund(order_id, reason)

    @tool
    def suggest_related_products_tool(product_id: str) -> str:
        """Suggest related products based on a given product ID."""
        return action_executor.suggest_related_products(product_id)

    return [
        get_product_details_tool,
        search_faq_tool,
        get_policy_info_tool,
        get_order_status_tool,
        process_refund_tool,
        suggest_related_products_tool,
    ]

# --- Modules End ---


# agent.py
class CustomerSupportAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        working_memory: WorkingMemory,
        knowledge_base: KnowledgeBase,
        action_executor: ActionExecutor,
    ):
        self.llm = llm
        self.working_memory = working_memory
        self.knowledge_base = knowledge_base
        self.action_executor = action_executor

        self.tools = get_langchain_tools(self.knowledge_base, self.action_executor)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an helpful AI customer support agent for an e-commerce store. Respond concisely."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

        self.conversational_agent = RunnableWithMessageHistory(
            self.agent_executor,
            lambda session_id: self.working_memory.get_history(session_id),
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def run(self, session_id: str, query: str) -> str:
        response = self.conversational_agent.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        ai_message_content = response["output"]
        
        # LangChain's RunnableWithMessageHistory handles adding messages to history
        # but we need to ensure the underlying working_memory is updated if we were to
        # manually manage history outside of RunnableWithMessageHistory.
        # For this setup, RunnableWithMessageHistory directly accesses and updates history
        # via the get_history method, so manual add_message here might be redundant 
        # if RunnableWithMessageHistory directly modifies the list returned by get_history.
        # However, for a robust memory system, ensure the memory module's add_message is called
        # after each turn if RunnableWithMessageHistory doesn't directly persist.
        # For this mock, we assume RunnableWithMessageHistory interacts correctly.
        
        return ai_message_content


# app.py
app = FastAPI(title="E-commerce Customer Support Agent")

# Initialize modules
llm_client = MockLLM()
working_memory = WorkingMemory()
knowledge_base = KnowledgeBase()
action_executor = ActionExecutor()

# Initialize the agent
customer_support_agent = CustomerSupportAgent(
    llm=llm_client,
    working_memory=working_memory,
    knowledge_base=knowledge_base,
    action_executor=action_executor,
)

class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    agent_response = customer_support_agent.run(request.session_id, request.query)
    return ChatResponse(response=agent_response)

# To run this application:
# 1. Save the code as app.py
# 2. Install necessary libraries: pip install fastapi uvicorn pydantic langchain_core langchain
# 3. Run from your terminal: uvicorn app:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI. 