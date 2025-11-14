from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class TaskStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    DELAYED = "DELAYED"

class Task(BaseModel):
    name: str = Field(..., description="Name of the task")
    description: Optional[str] = Field(None, description="Detailed description of the task")
    status: TaskStatus = Field(TaskStatus.NOT_STARTED, description="Current status of the task")
    dependencies: List[str] = Field(default_factory=list, description="List of task names that this task depends on")
    estimated_duration_hours: Optional[int] = Field(None, description="Estimated duration in hours")
    assigned_to: Optional[str] = Field(None, description="Team member assigned to the task")
    actual_start_time: Optional[str] = Field(None, description="Actual start timestamp (ISO format)")
    actual_end_time: Optional[str] = Field(None, description="Actual end timestamp (ISO format)")

class ProjectPlan(BaseModel):
    project_goal: str = Field(..., description="The high-level goal of the project")
    tasks: List[Task] = Field(default_factory=list, description="List of all tasks in the project")
    constraints: List[str] = Field(default_factory=list, description="List of project constraints (e.g., budget, deadline, compliance)")
    overall_status: TaskStatus = Field(TaskStatus.NOT_STARTED, description="Overall status of the project")
    notes: Optional[str] = Field(None, description="Any additional notes or observations")

class ProjectGoalInput(BaseModel):
    goal: str = Field(..., description="A detailed description of the project's high-level goal.")
    initial_constraints: List[str] = Field(default_factory=list, description="Any initial hard or soft constraints for the project.")
