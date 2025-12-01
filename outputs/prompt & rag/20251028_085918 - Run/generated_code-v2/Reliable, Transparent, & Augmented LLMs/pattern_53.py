import time
import random
import re
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class KnowledgeBaseOutput(BaseModel):
    query: str = Field(...)
    answer: str = Field(...)
    source_url: Optional[str] = Field(None)
    confidence: float = Field(..., ge=0.0, le=1.0)

class CustomerInfo(BaseModel):
    customer_id: str = Field(...)
    name: str = Field(...)
    email: str = Field(...)
    loyalty_status: str = Field(...)
    open_tickets: int = Field(..., ge=0)

class CRMOutput(BaseModel):
    action: str = Field(...)
    status: str = Field(...)
    message: str = Field(...)
    customer_data: Optional[CustomerInfo] = Field(None)
    ticket_id: Optional[str] = Field(None)

class OrderItem(BaseModel):
    product_name: str = Field(...)
    quantity: int = Field(..., ge=1)
    price_per_unit: float = Field(..., gt=0)

class OrderManagementOutput(BaseModel):
    order_id: str = Field(...)
    status: str = Field(...)
    customer_id: str = Field(...)
    order_date: str = Field(...)
    total_amount: float = Field(..., gt=0)
    items: List[OrderItem] = Field(...)
    tracking_number: Optional[str] = Field(None)

class ValidationErrorInfo(BaseModel):
    tool_name: str
    error_type: str
    message: str
    details: Optional[dict]

class KnowledgeBaseTool:
    def search(self, query: str) -> Dict[str, Any]:
        print(f"[KnowledgeBaseTool] Searching for: {query}")
        time.sleep(0.5)
        if "return policy" in query.lower():
            return {
                "query": query,
                "answer": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be unused and in their original packaging.",
                "source_url": "https://example.com/returns",
                "confidence": 0.95
            }
        elif "shipping cost" in query.lower():
            return {
                "query": query,
                "answer": "Standard shipping within the continental US is $5.99. Expedited options are available at checkout.",
                "source_url": "https://example.com/shipping",
                "confidence": 0.90
            }
        elif "malicious content" in query.lower():
            return {
                "query": query,
                "answer": "<script>alert('XSS Attack!');</script> This is a normal answer following some bad code.",
                "source_url": "https://bad-example.com",
                "confidence": 0.7
            }
        elif "low confidence info" in query.lower():
            return {
                "query": query,
                "answer": "Some uncertain information about a product.",
                "source_url": "https://example.com/product-faq",
                "confidence": 0.3
            }
        else:
            return {
                "query": query,
                "answer": "I could not find a direct answer to your query in our knowledge base. Please try rephrasing.",
                "source_url": None,
                "confidence": 0.6
            }

class CRMIntegrationTool:
    def get_customer_info(self, customer_id: str) -> Dict[str, Any]:
        print(f"[CRMIntegrationTool] Retrieving info for customer: {customer_id}")
        time.sleep(0.3)
        if customer_id == "cust123":
            return {
                "action": "get_info",
                "status": "success",
                "message": "Customer information retrieved successfully.",
                "customer_data": {
                    "customer_id": "cust123",
                    "name": "Alice Smith",
                    "email": "alice.smith@example.com",
                    "loyalty_status": "Gold",
                    "open_tickets": 1
                }
            }
        elif customer_id == "malicious_id_injection":
            return {
                "action": "get_info",
                "status": "success",
                "message": "Customer information retrieved successfully.",
                "customer_data": {
                    "customer_id": "<script>alert('SQL Injection Attempt!');</script>",
                    "name": "Bob Johnson",
                    "email": "bob.johnson@example.com",
                    "loyalty_status": "Silver",
                    "open_tickets": 0
                }
            }
        else:
            return {
                "action": "get_info",
                "status": "failure",
                "message": "Customer not found.",
                "customer_data": None
            }

    def update_ticket(self, customer_id: str, ticket_id: str, new_status: str) -> Dict[str, Any]:
        print(f"[CRMIntegrationTool] Updating ticket {ticket_id} for customer {customer_id} to {new_status}")
        time.sleep(0.4)
        if customer_id == "cust123" and ticket_id == "TKT-001":
            return {
                "action": "update_ticket",
                "status": "success",
                "message": f"Ticket {ticket_id} updated to {new_status}.",
                "ticket_id": ticket_id
            }
        else:
            return {
                "action": "update_ticket",
                "status": "failure",
                "message": "Failed to update ticket. Check customer ID or ticket ID."
            }

class OrderManagementTool:
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        print(f"[OrderManagementTool] Getting status for order: {order_id}")
        time.sleep(0.6)
        if order_id == "ORD789":
            return {
                "order_id": "ORD789",
                "status": "shipped",
                "customer_id": "cust123",
                "order_date": "2023-10-26",
                "total_amount": 125.50,
                "items": [
                    {"product_name": "Wireless Headphones", "quantity": 1, "price_per_unit": 99.99},
                    {"product_name": "USB-C Cable", "quantity": 2, "price_per_unit": 12.75}
                ],
                "tracking_number": "TRK987654321"
            }
        elif order_id == "ORD101_invalid_amount":
            return {
                "order_id": "ORD101",
                "status": "processing",
                "customer_id": "cust456",
                "order_date": "2023-11-01",
                "total_amount": -10.00,
                "items": [
                    {"product_name": "Laptop Bag", "quantity": 1, "price_per_unit": 50.00}
                ]
            }
        elif order_id == "ORD102_missing_items":
             return {
                "order_id": "ORD102",
                "status": "delivered",
                "customer_id": "cust789",
                "order_date": "2023-10-20",
                "total_amount": 75.00,
                "items": []
            }
        else:
            return {
                "order_id": order_id,
                "status": "not_found",
                "customer_id": "N/A",
                "order_date": "N/A",
                "total_amount": 0.0,
                "items": [],
                "tracking_number": None
            }

class ToolOutputValidator:
    def __init__(self):
        self.harmful_patterns = [
            re.compile(r"<script>.*?</script>", re.IGNORECASE),
            re.compile(r"DROP TABLE", re.IGNORECASE),
            re.compile(r"alert\\(.*?\\)", re.IGNORECASE)
        ]
        self.suspicious_keywords = ["xss", "injection", "malware", "exploit"]

    def _check_harmful_content(self, text: str) -> bool:
        for pattern in self.harmful_patterns:
            if pattern.search(text):
                return True
        for keyword in self.suspicious_keywords:
            if keyword in text.lower():
                return True
        return False

    def validate_knowledge_base_output(self, output: Dict[str, Any]) -> Optional[ValidationErrorInfo]:
        try:
            validated_output = KnowledgeBaseOutput(**output)

            if validated_output.confidence < 0.5:
                return ValidationErrorInfo(
                    tool_name="KnowledgeBaseTool",
                    error_type="ContentConsistency",
                    message="Knowledge base answer confidence is too low.",
                    details={"confidence": validated_output.confidence}
                )

            if self._check_harmful_content(validated_output.answer) or \
               (validated_output.source_url and self._check_harmful_content(validated_output.source_url)):
                return ValidationErrorInfo(
                    tool_name="KnowledgeBaseTool",
                    error_type="HarmfulContent",
                    message="Detected potentially harmful content in knowledge base output."
                )

            return None
        except ValidationError as e:
            return ValidationErrorInfo(
                tool_name="KnowledgeBaseTool",
                error_type="SchemaValidation",
                message="Knowledge base output schema mismatch.",
                details=e.errors()
            )
        except Exception as e:
            return ValidationErrorInfo(
                tool_name="KnowledgeBaseTool",
                error_type="UnknownError",
                message=f"An unexpected error occurred during knowledge base output validation: {e}"
            )

    def validate_crm_output(self, output: Dict[str, Any]) -> Optional[ValidationErrorInfo]:
        try:
            validated_output = CRMOutput(**output)

            if validated_output.status == "failure":
                return ValidationErrorInfo(
                    tool_name="CRMIntegrationTool",
                    error_type="ToolExecutionFailure",
                    message=f"CRM tool reported a failure: {validated_output.message}",
                    details={"action": validated_output.action}
                )
            
            if validated_output.customer_data:
                if self._check_harmful_content(validated_output.customer_data.customer_id) or \
                   self._check_harmful_content(validated_output.customer_data.name) or \
                   self._check_harmful_content(validated_output.customer_data.email):
                    return ValidationErrorInfo(
                        tool_name="CRMIntegrationTool",
                        error_type="HarmfulContent",
                        message="Detected potentially harmful content in CRM customer data."
                    )

            return None
        except ValidationError as e:
            return ValidationErrorInfo(
                tool_name="CRMIntegrationTool",
                error_type="SchemaValidation",
                message="CRM output schema mismatch.",
                details=e.errors()
            )
        except Exception as e:
            return ValidationErrorInfo(
                tool_name="CRMIntegrationTool",
                error_type="UnknownError",
                message=f"An unexpected error occurred during CRM output validation: {e}"
            )

    def validate_order_management_output(self, output: Dict[str, Any]) -> Optional[ValidationErrorInfo]:
        try:
            validated_output = OrderManagementOutput(**output)

            calculated_total = sum(item.quantity * item.price_per_unit for item in validated_output.items)
            if abs(calculated_total - validated_output.total_amount) > 0.01:
                return ValidationErrorInfo(
                    tool_name="OrderManagementTool",
                    error_type="ContentConsistency",
                    message="Order total amount inconsistent with sum of items.",
                    details={"declared_total": validated_output.total_amount, "calculated_total": calculated_total}
                )
            
            if not validated_output.items and validated_output.total_amount > 0:
                 return ValidationErrorInfo(
                    tool_name="OrderManagementTool",
                    error_type="ContentConsistency",
                    message="Order has a total amount but no items."
                )
            
            if self._check_harmful_content(validated_output.order_id) or \
               self._check_harmful_content(validated_output.status) or \
               self._check_harmful_content(validated_output.customer_id):
                return ValidationErrorInfo(
                    tool_name="OrderManagementTool",
                    error_type="HarmfulContent",
                    message="Detected potentially harmful content in Order Management output."
                )
            for item in validated_output.items:
                if self._check_harmful_content(item.product_name):
                     return ValidationErrorInfo(
                        tool_name="OrderManagementTool",
                        error_type="HarmfulContent",
                        message="Detected potentially harmful content in Order Management item product name."
                    )

            return None
        except ValidationError as e:
            return ValidationErrorInfo(
                tool_name="OrderManagementTool",
                error_type="SchemaValidation",
                message="Order Management output schema mismatch.",
                details=e.errors()
            )
        except Exception as e:
            return ValidationErrorInfo(
                tool_name="OrderManagementTool",
                error_type="UnknownError",
                message=f"An unexpected error occurred during order management output validation: {e}"
            )

class LLMCore:
    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"[LLM Core] Processing query: \"{query}\"")

        if "return policy" in query.lower() or "shipping cost" in query.lower() or "knowledge base" in query.lower():
            return {"tool": "knowledge_base", "params": {"query": query}}
        elif "customer info" in query.lower() or "update ticket" in query.lower():
            customer_id_match = re.search(r"customer (\\w+)", query, re.IGNORECASE)
            customer_id = customer_id_match.group(1) if customer_id_match else "cust123"
            if "update ticket" in query.lower():
                ticket_id_match = re.search(r"ticket (\\w+)", query, re.IGNORECASE)
                ticket_id = ticket_id_match.group(1) if ticket_id_match else "TKT-001"
                status_match = re.search(r"to (\\w+ status)", query, re.IGNORECASE)
                new_status = status_match.group(1) if status_match else "resolved"
                return {"tool": "crm_update_ticket", "params": {"customer_id": customer_id, "ticket_id": ticket_id, "new_status": new_status}}
            return {"tool": "crm_get_info", "params": {"customer_id": customer_id}}
        elif "order status" in query.lower():
            order_id_match = re.search(r"order (\\w+)", query, re.IGNORECASE)
            order_id = order_id_match.group(1) if order_id_match else "ORD789"
            return {"tool": "order_management", "params": {"order_id": order_id}}
        else:
            return {"tool": "none", "response": "I'm sorry, I can only assist with knowledge base, customer, or order-related queries at the moment."}

class ToolOrchestrator:
    def __init__(self):
        self.kb_tool = KnowledgeBaseTool()
        self.crm_tool = CRMIntegrationTool()
        self.om_tool = OrderManagementTool()

    def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[Tool Orchestrator] Invoking {tool_name} with params: {params}")
        if tool_name == "knowledge_base":
            return self.kb_tool.search(**params)
        elif tool_name == "crm_get_info":
            return self.crm_tool.get_customer_info(**params)
        elif tool_name == "crm_update_ticket":
            return self.crm_tool.update_ticket(**params)
        elif tool_name == "order_management":
            return self.om_tool.get_order_status(**params)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

class ResponseGenerator:
    def generate_response(self, user_query: str, validated_output: Optional[Dict[str, Any]], tool_name: str, validation_error: Optional[Any] = None) -> str:
        if validation_error:
            print(f"[Response Generator] Validation Error: {validation_error.message}")
            return f"I encountered an issue processing your request due to an invalid or potentially harmful response from our system. Error details: {validation_error.message}. Please try again or rephrase your query. (Internal Error Type: {validation_error.error_type})"
        
        if not validated_output:
            return f"I'm sorry, I couldn't retrieve the necessary information for your query: '{user_query}'"

        if tool_name == "knowledge_base":
            return f"Regarding your query about '{validated_output['query']}': {validated_output['answer']} (Source: {validated_output['source_url'] or 'N/A'})"
        elif tool_name == "crm_get_info":
            customer = validated_output.get('customer_data')
            if customer:
                return f"Here's the information for customer {customer['name']} (ID: {customer['customer_id']}): Email: {customer['email']}, Loyalty Status: {customer['loyalty_status']}, Open Tickets: {customer['open_tickets']}."
            return f"I couldn't find details for the requested customer. CRM message: {validated_output.get('message', 'N/A')}"
        elif tool_name == "crm_update_ticket":
            return f"Successfully updated ticket {validated_output.get('ticket_id', 'N/A')}. Status: {validated_output.get('status', 'N/A')}. Message: {validated_output.get('message', 'N/A')}"
        elif tool_name == "order_management":
            if validated_output['status'] == 'not_found':
                return f"I could not find order '{validated_output['order_id']}'. Please double-check the order ID."
            items_str = ", ".join([f"{item['quantity']}x {item['product_name']}" for item in validated_output['items']])
            return f"Order {validated_output['order_id']} (Customer {validated_output['customer_id']}) is currently {validated_output['status']}. Total: ${validated_output['total_amount']:.2f}. Items: {items_str}. Tracking: {validated_output['tracking_number'] or 'N/A'}."
        else:
            return validated_output.get('response', f"I'm sorry, I couldn't process your request: '{user_query}'")

def run_customer_agent(user_query: str):
    llm_core = LLMCore()
    tool_orchestrator = ToolOrchestrator()
    validator = ToolOutputValidator()
    response_generator = ResponseGenerator()

    llm_decision = llm_core.process_query(user_query)
    tool_name = llm_decision.get("tool")

    if tool_name == "none":
        final_response = response_generator.generate_response(user_query, None, tool_name, None)
    else:
        tool_output = tool_orchestrator.invoke_tool(tool_name, llm_decision.get("params", {}))
        print(f"[Main] Raw Tool Output: {json.dumps(tool_output, indent=2)}")

        validation_error = None
        if tool_name == "knowledge_base":
            validation_error = validator.validate_knowledge_base_output(tool_output)
        elif tool_name in ["crm_get_info", "crm_update_ticket"]:
            validation_error = validator.validate_crm_output(tool_output)
        elif tool_name == "order_management":
            validation_error = validator.validate_order_management_output(tool_output)
        
        if validation_error:
            final_response = response_generator.generate_response(user_query, None, tool_name, validation_error)
        else:
            final_response = response_generator.generate_response(user_query, tool_output, tool_name, None)
    
    print(f"\n[Agent Response]: {final_response}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    print("--- Robust Customer Support Agent Demo ---")
    print("Try queries like: \"What is your return policy?\" ")
    print("Also try adversarial/bad inputs: \"knowledge base with malicious content\" \n")

    run_customer_agent("What is your return policy?")
    run_customer_agent("Get customer info for cust123")
    run_customer_agent("What is the status of order ORD789?")
    run_customer_agent("Update ticket TKT-001 for customer cust123 to resolved status")

    run_customer_agent("Search knowledge base with malicious content")
    run_customer_agent("What is some low confidence info?")
    run_customer_agent("Get customer info for malicious_id_injection")
    run_customer_agent("What is the status of order ORD101_invalid_amount?")
    run_customer_agent("What is the status of order ORD102_missing_items?")
    run_customer_agent("Find something about unknown topic")