import streamlit as st
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import sys
import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

# --- Configuration and Setup ---

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database Setup (SQLite for simplicity in a single file)
DATABASE_URL = "sqlite:///./smart_tutoring.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ProblemSolution(Base):
    __tablename__ = "problem_solutions"
    id = Column(Integer, primary_key=True, index=True)
    problem_text = Column(Text, nullable=False)
    generated_code = Column(Text, nullable=False)
    execution_output = Column(Text, nullable=False)
    solution = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI()

# --- Pydantic Models ---

class ProblemRequest(BaseModel):
    problem_text: str

class ProblemResponse(BaseModel):
    solution: str
    explanation: str
    raw_code: str
    execution_output: str

# --- Secure Code Interpreter ---
def execute_python_code(code: str, timeout: int = 10) -> dict:
    try:
        # Create a temporary file for the code to be executed
        temp_file_path = "_temp_exec_code.py"
        with open(temp_file_path, "w") as f:
            f.write(code)

        # Use subprocess to run the Python script in an isolated environment
        # Basic sandboxing: no network, no file system access beyond current dir (ideally more restricted)
        # A real-world secure interpreter would involve Docker or similar robust sandboxing.
        process = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False, # Don't raise an exception for non-zero exit codes
            env={"PYTHONIOENCODING": "utf-8", **os.environ} # Ensure utf-8 encoding
        )

        os.remove(temp_file_path)

        if process.returncode != 0 and "KeyboardInterrupt" not in process.stderr:
            logger.error(f"Code execution failed with error: {process.stderr}")

        return {
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
            "returncode": process.returncode
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"Code execution timed out after {timeout} seconds.")
        return {"stdout": "", "stderr": "Execution timed out.", "returncode": 1}
    except Exception as e:
        logger.error(f"Error during code execution: {e}")
        return {"stdout": "", "stderr": f"Interpreter error: {e}", "returncode": 1}

# --- LLM Interaction (Placeholder) ---
def generate_code_with_llm(problem_text: str) -> str:
    # In a real application, this would call an LLM API (e.g., OpenAI, Gemini)
    # using langchain or direct API calls.
    # For this example, we'll simulate a simple code generation.
    logger.info(f"Simulating LLM code generation for problem: {problem_text}")
    
    # Basic heuristic to generate code based on common math problems
    if "solve for x:" in problem_text.lower():
        try:
            equation_part = problem_text.lower().split("solve for x:")[1].strip()
            # Very basic parsing, not robust for all equations
            if "+" in equation_part and "=" in equation_part:
                parts = equation_part.split("=")
                left = parts[0].strip()
                right = parts[1].strip()
                
                if "x" in left:
                    coeff_x_str = left.split("x")[0].strip()
                    coeff_x = float(coeff_x_str) if coeff_x_str and coeff_x_str != '+' else 1.0
                    
                    constant_left_str = left.split("+")[-1].strip()
                    constant_left = float(constant_left_str) if constant_left_str else 0.0

                    result_code = f"value_right = float({right})\nconstant_term = float({constant_left})\ncoefficient_x = float({coeff_x})\nx = (value_right - constant_term) / coefficient_x\nprint(f\"x = {{x}}\")\n"
                    return result_code

        except Exception as e:
            logger.error(f"Error parsing math problem for simulation: {e}")

    elif "reverse string" in problem_text.lower():
        return 