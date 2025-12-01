from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Iterator

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool, ToolException
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, FunctionCall
from langchain_core.messages.tool import ToolCall

class CRMLookupInput(BaseModel):
    customer_id: str = Field(description="The ID of the customer to look up.")

def get_customer_info(customer_id: str) -> str:
    if customer_id == "C123":
        return "Customer C123: John Doe, Email: john.doe@example.com"
    return f"Customer {customer_id} not found."

class OrderManagementInput(BaseModel):
    order_id: str = Field(description="The ID of the order to retrieve details for.")

def get_order_details(order_id: str) -> str:
    if order_id == "ORD001":
        return "Order ORD001: Product X, Quantity: 2, Status: Shipped"
    return f"Order {order_id} not found."

class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(description="The search query for the knowledge base.")

def search_knowledge_base(query: str) -> str:
    if "return policy" in query.lower():
        return "Our return policy allows returns within 30 days of purchase with a valid receipt."
    if "shipping options" in query.lower():
        return "We offer standard, expedited, and express shipping options. See our shipping page for details."
    return f"No relevant articles found for '{query}'."

class RefundProcessingInput(BaseModel):
    order_id: str = Field(description="The ID of the order for which to process a refund.")
    amount: float = Field(description="The amount to refund.")

def process_refund(order_id: str, amount: float) -> str:
    if order_id == "ORD001" and amount <= 100.0:
        return f"Refund of ${amount:.2f} processed for order {order_id}."
    return f"Failed to process refund for order {order_id} with amount ${amount:.2f}. Invalid order or amount."

class MockChatModel(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], stop: Union[List[str], None] = None, **kwargs: Any) -> Any:
        last_message_content = messages[-1].content.lower()
        if "customer info" in last_message_content or "customer details" in last_message_content:
            return AIMessage(tool_calls=[ToolCall(name="get_customer_info", args={"customer_id": "C123"}, id="1")])
        elif "order status" in last_message_content or "order details" in last_message_content:
            return AIMessage(tool_calls=[ToolCall(name="get_order_details", args={"order_id": "ORD001"}, id="2")])
        elif "refund" in last_message_content:
            return AIMessage(tool_calls=[ToolCall(name="process_refund", args={"order_id": "ORD001", "amount": 50.0}, id="3")])
        elif "how to" in last_message_content or "problem with" in last_message_content or "policy" in last_message_content or "shipping" in last_message_content:
            return AIMessage(tool_calls=[ToolCall(name="search_knowledge_base", args={"query": last_message_content}, id="4")])
        return AIMessage(content="I can help with customer information, order details, refunds, or general knowledge base queries. How can I assist you?")

    def _stream(self, messages: List[BaseMessage], stop: Union[List[str], None] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        response_message = self._generate(messages, stop, **kwargs)
        if isinstance(response_message, AIMessage) and response_message.tool_calls:
            yield ChatGenerationChunk(message=AIMessage(tool_calls=response_message.tool_calls))
        else:
            for char in response_message.content:
                yield ChatGenerationChunk(message=AIMessage(content=char))

    def _get_parameters(self) -> Dict[str, Any]:
        return {}

crm_tool = tool(get_customer_info, args_schema=CRMLookupInput)
order_tool = tool(get_order_details, args_schema=OrderManagementInput)
kb_tool = tool(search_knowledge_base, args_schema=KnowledgeBaseSearchInput)
refund_tool = tool(process_refund, args_schema=RefundProcessingInput)

all_tools = [crm_tool, order_tool, kb_tool, refund_tool]

llm = MockChatModel()
llm_with_tools = llm.bind_tools(all_tools)

system_message = (
    "You are an AI-powered customer support agent. "
    "You have access to the following tools to assist customers: "
    f"{all_tools[0].name}: {all_tools[0].description} (Input: {CRMLookupInput.schema_json()})\n" # type: ignore
    f"{all_tools[1].name}: {all_tools[1].description} (Input: {OrderManagementInput.schema_json()})\n" # type: ignore
    f"{all_tools[2].name}: {all_tools[2].description} (Input: {KnowledgeBaseSearchInput.schema_json()})\n" # type: ignore
    f"{all_tools[3].name}: {all_tools[3].description} (Input: {RefundProcessingInput.schema_json()})\n" # type: ignore
    "Respond to customer queries by utilizing these tools when appropriate. "
    "If a tool call is needed, provide the appropriate parameters. "
    "Otherwise, respond directly to the customer."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{input}"),
])

def _run_tools(messages: List[BaseMessage]) -> BaseMessage:
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        tool_results = []
        for tool_call in messages[-1].tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

            tool_found = False
            for t in all_tools:
                if t.name == tool_name:
                    try:
                        result = t.func(**tool_args)
                        tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_id))
                    except Exception as e:
                        tool_results.append(ToolMessage(content=f"Error executing tool {tool_name}: {e}", tool_call_id=tool_id))
                    tool_found = True
                    break
            if not tool_found:
                tool_results.append(ToolMessage(content=f"Tool {tool_name} not found.", tool_call_id=tool_id))
        return AIMessage(content="", tool_calls=messages[-1].tool_calls, tool_outputs=tool_results)
    return messages[-1]

agent_chain = (
    RunnablePassthrough.assign(agent_scratchpad=lambda x: _run_tools([AIMessage(tool_calls=[tc for tc in x["intermediate_steps"] if isinstance(tc, ToolCall)]) if isinstance(x["intermediate_steps"], list) else []])) # This part might need adjustment depending on how intermediate_steps is structured if not used in a direct agent executor
    | prompt
    | llm_with_tools
    | RunnableLambda(_run_tools)
)

def process_query(query: str) -> str:
    response = agent_chain.invoke({"input": query})
    
    # If the response is an AIMessage with tool_outputs, concatenate them or use specific logic
    if isinstance(response, AIMessage) and response.tool_outputs:
        output_content = []
        for i, tool_output in enumerate(response.tool_outputs):
            tool_call = response.tool_calls[i] if i < len(response.tool_calls) else None
            tool_name = tool_call.name if tool_call else "Unknown Tool"
            output_content.append(f"Tool '{tool_name}' output: {tool_output.content}")
        return "\n".join(output_content)
    elif isinstance(response, AIMessage):
        return response.content
    return str(response)

if __name__ == "__main__":
    print("AI Customer Support Agent (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        agent_response = process_query(user_input)
        print(f"Agent: {agent_response}")