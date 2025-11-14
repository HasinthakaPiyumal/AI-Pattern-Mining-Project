"""
This module implements an AI-Powered Project Assistant with Dynamic Task Planning.
It demonstrates the Adaptive Planning and Task Decomposition pattern by breaking down complex project goals,
managing constraints, and adapting plans based on feedback.

Libraries Used:
- pydantic: For defining structured data models (Tasks, Constraints).
- networkx: For managing task dependencies as a Directed Acyclic Graph (DAG).
- langchain: For simulating LLM interactions (though using a mock here).
- streamlit: For building an interactive user interface.
"""

import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import networkx as nx
import time

# --- 1. Pydantic Models for Task and Constraint Management ---

class Task(BaseModel):
    id: str
    name: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    assigned_to: Optional[str] = None
    estimated_duration_hours: int = 1

class Constraint(BaseModel):
    id: str
    type: str  # e.g., "deadline", "resource_availability", "skill_requirement"
    value: Any
    applies_to_task_id: Optional[str] = None
    description: Optional[str] = None

# --- 2. LLM Core (Mock Implementation) ---

class MockLLM:
    """A mock LLM to simulate response generation without actual API calls."""
    def predict(self, prompt: str) -> str:
        if "decompose" in prompt.lower() and "website" in prompt.lower():
            return """
            [{{"id": "t1", "name": "Design UI", "description": "Create wireframes and mockups.", "dependencies": []}},
            {{"id": "t2", "name": "Develop Backend API", "description": "Implement user authentication and data storage.", "dependencies": []}},
            {{"id": "t3", "name": "Implement Frontend", "description": "Build interactive user interface using React.", "dependencies": ["t1", "t2"]}},
            {{"id": "t4", "name": "Database Setup", "description": "Configure PostgreSQL database.", "dependencies": ["t2"]}},
            {{"id": "t5", "name": "Deploy Application", "description": "Deploy frontend and backend to cloud server.", "dependencies": ["t3", "t4"]}}]
            """
        elif "optimize plan" in prompt.lower():
            return 