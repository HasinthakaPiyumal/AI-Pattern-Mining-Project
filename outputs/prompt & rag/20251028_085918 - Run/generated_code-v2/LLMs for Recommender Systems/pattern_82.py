import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Pydantic Models ---
class Item(BaseModel):
    item_id: str
    name: str
    category: str
    description: str
    price: float
    color: str
    occasion: Optional[str] = None
    season: Optional[str] = None

class UserProfile(BaseModel):
    user_id: str
    name: str
    preferences: List[str]  # e.g., [