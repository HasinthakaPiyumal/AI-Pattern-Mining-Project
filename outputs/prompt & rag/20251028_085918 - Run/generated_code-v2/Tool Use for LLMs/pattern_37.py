from pydantic import BaseModel, Field, conlist
from typing import Optional, Dict, Any, List

class APIArgument(BaseModel):
    name: str = Field(..., description="The name of the API argument.")
    type: str = Field(..., description="The data type of the argument (e.g., string, integer, boolean, object).")
    required: bool = Field(..., description="Whether the argument is required for the API call.")
    description: Optional[str] = Field(None, description="A description of the argument's purpose and expected values.")
    example: Optional[Any] = Field(None, description="An example value for the argument.")

class StructuredAPIDocumentation(BaseModel):
    domain: str = Field(..., description="The high-level domain or category of the API (e.g., E-commerce, Payment, Shipping).")
    framework: Optional[str] = Field(None, description="The specific framework or platform the API belongs to (e.g., Shopify, Stripe, FedEx).")
    functionality: str = Field(..., description="A high-level description of what the API or endpoint does (e.g., order processing, inventory management, customer communication).")
    apiname: str = Field(..., description="A unique name for the API or endpoint within its domain.")
    apicall: str = Field(..., description="The actual API endpoint path or method signature (e.g., /orders/{order_id}/fulfill, update_product_inventory).")
    http_method: Optional[str] = Field(None, description="The HTTP method if applicable (e.g., GET, POST, PUT, DELETE).")
    apiarguments: conlist(APIArgument, default_factory=list) = Field(..., description="A list of arguments required or optional for the API call.")
    environmentrequirements: Optional[str] = Field(None, description="Any specific environment setup or credentials required.")
    examplecode: Optional[str] = Field(None, description="An example code snippet demonstrating how to call the API.")
    performance: Optional[str] = Field(None, description="Notes on expected performance or rate limits.")
    description: str = Field(..., description="A detailed description of the API's purpose and usage.")
