import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from typing import List, Dict, Any
import json
import threading
import time
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_order_status_mock(order_id: str) -> Dict[str, Any]:
    if order_id == "ORDER123":
        return {"order_id": order_id, "status": "Shipped", "estimated_delivery": "2023-11-15"}
    elif order_id == "ORDER456":
        return {"order_id": order_id, "status": "Processing", "estimated_delivery": "2023-11-20"}
    else:
        return {"order_id": order_id, "status": "Not Found", "message": "Order ID not recognized."}

def search_products_mock(query: str, category: str = None, color: str = None) -> List[Dict[str, Any]]:
    products = []
    if "shoes" in query.lower() and (color and "blue" in color.lower()):
        products.append({"name": "Blue Elegant Heels", "id": "P001", "price": 89.99, "category": "Footwear", "color": "Blue"})
        products.append({"name": "Navy Blue Sneakers", "id": "P002", "price": 55.00, "category": "Footwear", "color": "Navy Blue"})
    elif "shirt" in query.lower():
        products.append({"name": "Classic White Shirt", "id": "P003", "price": 35.00, "category": "Apparel", "color": "White"})
    return products if products else [{"message": "No products found matching your criteria."}]

def process_return_mock(item_name: str, reason: str, order_id: str = None) -> Dict[str, Any]:
    if "shirt" in item_name.lower():
        return {"status": "Return Initiated", "item": item_name, "reason": reason, "return_id": "RET789"}
    return {"status": "Failed", "item": item_name, "message": "Could not initiate return for this item."}

@tool
def get_order_status(order_id: str) -> Dict[str, Any]:
    return get_order_status_mock(order_id)

@tool
def search_products(query: str, category: str = None, color: str = None) -> List[Dict[str, Any]]:
    return search_products_mock(query, category, color)

@tool
def process_return(item_name: str, reason: str, order_id: str = None) -> Dict[str, Any]:
    return process_return_mock(item_name, reason, order_id)

tools = [get_order_status, search_products, process_return]

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent e-commerce customer support agent.
    Your goal is to assist users with complex queries by breaking them down into actionable steps.
    For each step, identify if it requires a tool, and if so, which tool and its parameters.
    The available tools are:
    - get_order_status(order_id: str): Get the status of a customer's order.
    - search_products(query: str, category: str = None, color: str = None): Search for products in the catalog.
    - process_return(item_name: str, reason: str, order_id: str = None): Initiate a return for an item.

    Your output should be a JSON array of tasks. Each task should have:
    - "task_description": A clear description of the task.
    - "tool_name": (Optional) The name of the tool to use for this task.
    - "tool_args": (Optional) A dictionary of arguments for the tool.
    - "depends_on": (Optional) A list of indices (0-based) of tasks that this task depends on.

    Example of output:
    [
        {"task_description": "Identify the order ID for the return.", "tool_name": null},
        {"task_description": "Process return for the shirt.", "tool_name": "process_return", "tool_args": {"item_name": "shirt", "reason": "damaged"}, "depends_on": [0]},
        {"task_description": "Find blue shoes.", "tool_name": "search_products", "tool_args": {"query": "shoes", "color": "blue"}}
    ]
    """),
    ("user", "{query}")
])

class JsonListOutputParser(StrOutputParser):
    def parse(self, text: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [{"error": "Failed to parse JSON plan.", "raw_output": text}]

planner_chain = planning_prompt | llm | JsonListOutputParser()

def execute_plan(plan_list: List[Dict[str, Any]], tools_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for i, task in enumerate(plan_list):
        task_description = task.get("task_description", "")
        tool_name = task.get("tool_name")
        tool_args = task.get("tool_args", {})
        
        result_item = {"task_description": task_description, "status": "skipped", "output": None}

        if tool_name and tool_name in tools_map:
            try:
                tool_func = tools_map[tool_name]
                output = tool_func.func(**tool_args)
                result_item["status"] = "executed"
                result_item["output"] = output
            except Exception as e:
                result_item["status"] = "error"
                result_item["output"] = str(e)
        else:
            result_item["status"] = "no_tool_needed_or_tool_not_found"
            result_item["output"] = f"Task: {task_description}"
        results.append(result_item)
    return results

tools_map_for_execution = {tool.name: tool for tool in tools}

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class AgentResponse(BaseModel):
    initial_plan: List[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    final_summary: str

@app.post("/query", response_model=AgentResponse)
async def process_user_query(request: QueryRequest):
    user_query = request.user_query
    
    initial_plan_raw = planner_chain.invoke({"query": user_query})
    
    execution_results = execute_plan(initial_plan_raw, tools_map_for_execution)

    summary_parts = []
    for result in execution_results:
        summary_parts.append(f"Task: {result['task_description']} -> Status: {result['status']}. Output: {result['output']}")
    final_summary = "\n".join(summary_parts)

    return AgentResponse(
        initial_plan=initial_plan_raw,
        execution_results=execution_results,
        final_summary=final_summary
    )

def gradio_interface(user_query: str):
    import requests
    
    try:
        response = requests.post("http://127.0.0.1:8000/query", json={"user_query": user_query})
        response.raise_for_status()
        data = response.json()
        
        plan_str = "## Initial Plan:\n"
        for i, task in enumerate(data["initial_plan"]):
            plan_str += f"{i+1}. {task.get("task_description")}"
            if task.get("tool_name"):
                plan_str += f" (Tool: {task['tool_name']}, Args: {task['tool_args']})"
            plan_str += "\n"

        results_str = "## Execution Results:\n"
        for i, res in enumerate(data["execution_results"]):
            results_str += f"{i+1}. Task: {res['task_description']} - Status: {res['status']}\n"
            results_str += f"   Output: {res['output']}\n"
        
        final_output = f"{plan_str}\n{results_str}\n## Final Summary:\n{data['final_summary']}"
        return final_output
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the FastAPI backend. Make sure it's running at http://127.0.0.1:8000."
    except requests.exceptions.RequestException as e:
        return f"Error during API call: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()

    time.sleep(2)

    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(lines=2, placeholder="Enter your complex query here..."),
        outputs="textbox",
        title="Intelligent E-commerce Customer Support Agent",
        description="Ask complex questions, and the AI agent will decompose, plan, and execute tasks."
    )
    iface.launch(share=False)