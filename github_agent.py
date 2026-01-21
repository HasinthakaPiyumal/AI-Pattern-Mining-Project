#!/usr/bin/env python3
"""
GitHub Repository Discovery Agent for AI Pattern Mining.

Built with LangChain and LangGraph.
Requires:
    pip install langchain-core langchain-openai langgraph requests pandas python-dotenv

Environment Variables:
    OPENAI_API_KEY: for LLM-based pattern confirmation
    GITHUB_TOKEN: for GitHub API access

Usage:
    python github_agent.py --topic "retrieval augmented generation" --max-repos 50
    python github_agent.py --test-heuristics  # Run internal unit tests
"""

import os
import sys
import time
import json
import logging
import argparse
import re
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Union
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- Configuration & Setup ---
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GitHubAgent")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GITHUB_TOKEN:
    logger.warning("GITHUB_TOKEN not found. API rate limits will be very strict.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found. Pattern detection will rely solely on heuristics.")

# --- Data Models ---

class RepoMetadata(TypedDict):
    full_name: str
    html_url: str
    description: str
    stars: int
    language: str
    default_branch: str

class ValidationResult(TypedDict):
    is_valid: bool
    flags: List[str]

class RepoResult(TypedDict):
    full_name: str
    html_url: str
    description: str
    stars: int
    detected_patterns: List[str]
    pattern_confidence: str  # "high" (LLM verified) or "medium" (heuristic only)
    summary: str
    validation_flags: List[str]
    timestamp: str

class AgentState(TypedDict):
    topic: str
    max_repos: int
    repos_to_process: List[RepoMetadata]
    processed_count: int
    csv_path: str
    search_page: int
    search_complete: bool

# --- Tools & Core Functions ---

def github_api_get(url: str, params: Dict = None) -> Dict:
    """Helper to make GitHub API requests with rate limit handling."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403 or response.status_code == 429:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset_time - time.time(), 1) + 1
                if wait > 60: # If wait is too long, maybe just backoff shorter or abort
                    logger.warning(f"Rate limit hit. Waiting {wait} seconds...")
                else:
                    logger.warning(f"Rate limit hit. Backing off...")
                time.sleep(min(wait, 60)) # Cap wait at 60s for this script's interactive nature
                continue
            else:
                logger.error(f"GitHub API Error {response.status_code}: {response.text}")
                return {}
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            time.sleep(2 ** attempt)
    return {}

def github_advanced_search(topic: str, page: int = 1, per_page: int = 30) -> List[RepoMetadata]:
    """Search GitHub repositories by topic."""
    query = f"{topic} stars:>10" # Basic enforcement of some quality
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page
    }
    
    data = github_api_get(url, params)
    items = data.get("items", [])
    
    results = []
    for item in items:
        results.append({
            "full_name": item["full_name"],
            "html_url": item["html_url"],
            "description": item["description"] or "",
            "stars": item["stargazers_count"],
            "language": item["language"] or "Unknown",
            "default_branch": item["default_branch"]
        })
    return results

def get_repo_structure(full_name: str, branch: str = "main") -> List[str]:
    """Fetch the file tree of a repository."""
    url = f"https://api.github.com/repos/{full_name}/git/trees/{branch}?recursive=1"
    data = github_api_get(url)
    
    if data.get("truncated", False):
        logger.warning(f"Tree for {full_name} is truncated.")
        
    tree = data.get("tree", [])
    paths = [item["path"] for item in tree if item["type"] == "blob"]
    return paths

def read_readme(full_name: str, branch: str) -> str:
    """Fetch README content."""
    # Try different standard filenames
    for filename in ["README.md", "readme.md", "README.rst", "README.txt"]:
        url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{filename}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.text
        except:
            continue
    return ""

def validate_repo(metadata: RepoMetadata, file_list: List[str]) -> ValidationResult:
    """Check if the repository is relevant for AI/ML."""
    flags = []
    
    # Check for code files
    has_code = any(f.endswith(('.py', '.ipynb', '.ts', '.js', '.go', '.rs')) for f in file_list)
    if not has_code:
        flags.append("no_code")
        
    # Check for keywords in description
    desc = metadata["description"].lower()
    ai_keywords = ["ai", "llm", "gpt", "rag", "agent", "transformer", "diffusion", "model"]
    is_ai_relevant = any(k in desc for k in ai_keywords)
    
    if not is_ai_relevant and not has_code:
        return {"is_valid": False, "flags": flags + ["irrelevant_topic"]}
        
    return {"is_valid": True, "flags": flags}

KEYWORDS_MAP = {
    "RAG": [r"\brag\b", r"retrieval[- ]augmented", r"vector database", r"pinecone", r"weaviate"],
    "Agents": [r"\bagent(s)?\b", r"autonomous", r"langchain", r"langgraph", r"crewai", r"autogen"],
    "Chain-of-Thought": [r"chain[- ]of[- ]thought", r"\bcot\b", r"reasoning steps"],
    "Fine-tuning": [r"fine[- ]tuning", r"\bpeft\b", r"\blora\b", r"training script"],
    "Multimodal": [r"multimodal", r"vision[- ]language", r"clip", r"whisper", r"diffusion"],
    "Human-in-the-loop": [r"human[- ]in[- ]the[- ]loop", r"\bhitl\b", r"human approval"]
}

def detect_patterns_heuristic(text: str, file_list: List[str]) -> List[str]:
    """Heuristic pattern detection using keywords."""
    patterns = []
    text_lower = text.lower()
    
    for category, regexes in KEYWORDS_MAP.items():
        for regex in regexes:
            if re.search(regex, text_lower):
                patterns.append(category)
                break
                
    # File-based heuristics
    if any(f.endswith("tools.py") for f in file_list):
        patterns.append("Tool Use")
    
    return list(set(patterns))

def summarize_and_confirm_patterns(readme: str, heuristics: List[str], model: ChatOpenAI) -> Dict:
    """Use LLM to confirm patterns and generate a summary."""
    if not OPENAI_API_KEY:
        return {"patterns": heuristics, "confidence": "medium", "summary": "LLM skipped (no key)."}
    
    # Truncate README to avoid huge tokens
    readme_snippet = readme[:4000]
    
    prompt = f"""
    Analyze this GitHub README for AI patterns.
    Heuristics detected: {heuristics}
    
    README snippet:
    {readme_snippet}
    
    Task:
    1. Confirm if the heuristic patterns are accurate.
    2. Identify other high-level AI patterns (e.g. Agentic, RAG, Fine-tuning).
    3. Write a 1-sentence summary of what the repo does.
    
    Return JSON format:
    {{
        "verified_patterns": ["Pattern1", "Pattern2"],
        "summary": "This repo implements..."
    }}
    """
    
    try:
        if hasattr(model, "with_structured_output"):
             # For newer LC versions that support structured output natively
            class AnalysisSchema(BaseModel):
                verified_patterns: List[str] = Field(description="List of AI patterns confirmed")
                summary: str = Field(description="A concise 1-sentence summary")
            
            structured_llm = model.with_structured_output(AnalysisSchema)
            result = structured_llm.invoke(prompt)
            return {"patterns": result.verified_patterns, "confidence": "high", "summary": result.summary}
        else:
             # Fallback
            response = model.invoke([HumanMessage(content=prompt)])
            # Naive JSON parsing
            content = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return {"patterns": data.get("verified_patterns", []), "confidence": "high", "summary": data.get("summary", "")}
            
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return {"patterns": heuristics, "confidence": "medium", "summary": "LLM failed."}

def save_progress(results: List[RepoResult], csv_path: str):
    """Save results to CSV idempotently."""
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # If file exists, check for duplicates before appending
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Filter out repos already in existing_df
        existing_urls = set(existing_df["html_url"])
        new_rows = df[~df["html_url"].isin(existing_urls)]
        
        if not new_rows.empty:
            new_rows.to_csv(csv_path, mode='a', header=False, index=False)
            logger.info(f"Appended {len(new_rows)} new repos to {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        logger.info(f"Created {csv_path} with {len(df)} repos")

# --- LangGraph Nodes ---

def search_node(state: AgentState):
    """Node: Search GitHub for repos."""
    logger.info(f"Searching: Topic '{state['topic']}' (Page {state['search_page']})")
    
    repos = github_advanced_search(state['topic'], page=state['search_page'])
    
    if not repos:
        logger.info("No more repos found.")
        return {"repos_to_process": [], "search_complete": True}
        
    return {
        "repos_to_process": repos,
        "search_page": state['search_page'] + 1,
        "search_complete": False
    }

def processing_node(state: AgentState):
    """Node: Process each repo (Fetch Content -> Validate -> Detect Patterns)."""
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini") if OPENAI_API_KEY else None
    
    results = []
    
    for repo in state["repos_to_process"]:
        if state["processed_count"] >= state["max_repos"]:
            break
            
        logger.info(f"Processing {repo['full_name']}...")
        
        # 1. Get Details
        file_tree = get_repo_structure(repo['full_name'], repo['default_branch'])
        readme_text = read_readme(repo['full_name'], repo['default_branch'])
        
        # 2. Validate
        validation = validate_repo(repo, file_tree)
        
        # 3. Detect Patterns
        heuristics = detect_patterns_heuristic(readme_text + " " + repo["description"], file_tree)
        
        # 4. LLM Verification (Only if valid code or promising heuristics)
        if validation["is_valid"] or heuristics:
            analysis = summarize_and_confirm_patterns(readme_text, heuristics, model)
        else:
            analysis = {"patterns": [], "confidence": "none", "summary": "Skipped LLM (irrelevant)"}
            
        repo_result: RepoResult = {
            "full_name": repo["full_name"],
            "html_url": repo["html_url"],
            "description": repo["description"],
            "stars": repo["stars"],
            "detected_patterns": ', '.join(analysis["patterns"]),
            "pattern_confidence": analysis["confidence"],
            "summary": analysis["summary"],
            "validation_flags": ', '.join(validation["flags"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Filter: Only keep repos with detected patterns OR code-validation pass
        if analysis["patterns"] or validation["is_valid"]:
            results.append(repo_result)
            
    return {
        "processed_results": results, 
        "processed_count": state["processed_count"] + len(results),
        "repos_to_process": [] # Clear buffer
    }

def saving_node(state: AgentState):
    """Node: Save progress to disk."""
    save_progress(state["processed_results"], state["csv_path"])
    return {"processed_results": []} # Clear processed results from state after saving

def limit_check_node(state: AgentState):
    """Conditional Edge: Continue searching or stop."""
    if state["processed_count"] >= state["max_repos"]:
        logger.info(f"Target reached: {state['processed_count']} repos processed.")
        return "end"
    if state["search_complete"]:
        logger.info("Search exhausted.")
        return "end"
    return "continue"

# --- Main Construction ---

def create_agent_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("search", search_node)
    graph.add_node("process", processing_node)
    graph.add_node("save", saving_node)
    
    graph.set_entry_point("search")
    
    graph.add_edge("search", "process")
    graph.add_edge("process", "save")
    
    graph.add_conditional_edges(
        "save",
        limit_check_node,
        {
            "continue": "search",
            "end": END
        }
    )
    
    return graph.compile()

# --- Unit Tests ---

def test_heuristics():
    """Unit test for heuristic pattern detection."""
    print("Running Heuristic Tests...")
    
    sample_text = "This repository uses retrieval augmented generation (RAG) with a vector database."
    file_list = ["src/main.py"]
    patterns = detect_patterns_heuristic(sample_text, file_list)
    assert "RAG" in patterns, f"Expected RAG, got {patterns}"
    
    sample_text_2 = "Built with LangChain and implements autonomous agents."
    patterns_2 = detect_patterns_heuristic(sample_text_2, file_list)
    assert "Agents" in patterns_2, f"Expected Agents, got {patterns_2}"
    
    print("ALL TESTS PASSED.")

# --- entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Agent for AI Pattern Discovery")
    parser.add_argument("--topic", type=str, default="AI Agents", help="Topic to search for")
    parser.add_argument("--max-repos", type=int, default=10, help="Max repos to process")
    parser.add_argument("--test-heuristics", action="store_true", help="Run internal tests")
    
    args = parser.parse_args()
    
    if args.test_heuristics:
        test_heuristics()
        sys.exit(0)
    
    print(f"Starting Agent... Topic: {args.topic}, Max: {args.max_repos}")
    
    agent = create_agent_graph()
    
    initial_state = {
        "topic": args.topic,
        "max_repos": args.max_repos,
        "repos_to_process": [],
        "processed_results": [],
        "processed_count": 0,
        "csv_path": "discovered_repos.csv",
        "search_page": 1,
        "search_complete": False
    }
    
    try:
        agent.invoke(initial_state)
        print(f"\nDone! Results saved to discovered_repos.csv")
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        # In case of crash, we rely on the periodic saves in the saving_node
