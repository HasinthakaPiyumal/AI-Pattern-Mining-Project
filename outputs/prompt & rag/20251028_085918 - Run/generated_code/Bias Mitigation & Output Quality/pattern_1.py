"""Python code for an Intelligent & Ethical Customer Support Agent.

This agent leverages advanced LLM prompting techniques (Demonstration Ensembling, Balanced Demonstrations, Cultural Awareness) to enhance accuracy, reduce bias, and provide culturally sensitive responses in automated customer service.

Key Components:
- FastAPI for API Gateway.
- LangChain for LLM orchestration.
- OpenAI for LLM and Embeddings.
- ChromaDB for Knowledge & Exemplar Base.
- SQLAlchemy (SQLite) for Conversation History.
- Pydantic for data validation.
- Dotenv for environment variables.
- Loguru for logging.
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

# SQLAlchemy for Conversation History
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# LangChain for LLM Orchestration and Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# --- Configuration --- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables. Please set it.")
    raise ValueError("OPENAI_API_KEY not set")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customer_support.db")
CHROMA_PERSIST_DIR = "./chroma_db"

# --- Database Setup (SQLite for simplicity) --- #
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=True)
    region = Column(String, nullable=True)
    inquiry = Column(Text, nullable=False)
    initial_prompts = Column(Text, nullable=True) # Storing a string representation of prompts
    llm_raw_responses = Column(Text, nullable=True)
    final_agent_response = Column(Text, nullable=False)
    bias_detected = Column(String, nullable=True) # Store as 'True'/'False'
    bias_disclaimer = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models --- #
class CustomerInquiry(BaseModel):
    question: str
    customer_id: Optional[str] = None
    region: str = "default" # e.g., "US", "EU", "Asia", "Latin America"

class AgentResponse(BaseModel):
    answer: str
    bias_detected: bool = False
    bias_disclaimer: Optional[str] = None

# --- Vector Database (ChromaDB) Setup and Data Population --- #
class KnowledgeBase:
    def __init__(self, persist_directory: str, embedding_function):
        self.vectorstore = Chroma(
            collection_name="customer_support_knowledge",
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        self._populate_knowledge_base()

    def _populate_knowledge_base(self):
        # Check if the collection is empty before adding data
        if self.vectorstore._collection.count() == 0:
            logger.info("Populating ChromaDB with initial knowledge and exemplars.")
            documents = [
                # Exemplars for Few-Shot Prompting (with implicit bias tags via content)
                {"page_content": "How do I reset my password? A: Go to settings, then security, then 'Reset Password'. This is a common tech issue.", "metadata": {"type": "tech_support", "sentiment": "neutral", "bias_focus": "none"}},
                {"page_content": "My internet is not working. A: First, try restarting your router. If that doesn't work, contact support. This is a common tech issue.", "metadata": {"type": "tech_support", "sentiment": "neutral", "bias_focus": "none"}},
                {"page_content": "I want to cancel my subscription. A: You can cancel via your account settings under 'Subscriptions'. You will not be charged for the next billing cycle. This is a common billing issue.", "metadata": {"type": "billing", "sentiment": "negative", "bias_focus": "none"}},
                {"page_content": "I was overcharged for my last bill. A: We apologize for the inconvenience. Please provide your account details so we can investigate the discrepancy. This is a common billing issue, often requiring human review.", "metadata": {"type": "billing", "sentiment": "negative", "bias_focus": "resolution_focused"}},
                {"page_content": "Can I get a refund for a recent purchase? A: Our refund policy states that returns are accepted within 30 days of purchase with a valid receipt. Some items may be non-refundable. This is a common refund inquiry.", "metadata": {"type": "billing", "sentiment": "neutral", "bias_focus": "policy_adherence"}},
                {"page_content": "How can I update my payment method? A: Log into your account, navigate to 'Payment Methods', and add or update your card details. This is a common account management issue.", "metadata": {"type": "account_management", "sentiment": "neutral", "bias_focus": "none"}},
                {"page_content": "I need help setting up my new device. A: Please refer to the quick start guide included with your device. For further assistance, visit our online support portal. This is a tech setup issue.", "metadata": {"type": "tech_support", "sentiment": "neutral", "bias_focus": "external_resource"}},
                {"page_content": "Where can I find information about your data privacy policy? A: Our full data privacy policy is available on our website under the 'Legal' section. This addresses privacy concerns.", "metadata": {"type": "policy", "sentiment": "neutral", "bias_focus": "data_privacy"}},
                {"page_content": "Is your service available in my country? A: Our service is currently available in regions X, Y, and Z. Check our website for an updated list of supported countries. This is a regional availability question.", "metadata": {"type": "availability", "sentiment": "neutral", "bias_focus": "geographical"}},
                {"page_content": "I have a complaint about a customer service representative. A: We take all feedback seriously. Please provide details of your experience, and a manager will review your case. This addresses service complaints.", "metadata": {"type": "complaint", "sentiment": "negative", "bias_focus": "customer_satisfaction"}},

                # Cultural Contexts (embedded as documents for semantic retrieval if needed, or direct lookup)
                {"page_content": "Cultural context for US: Emphasize directness, efficiency, and clear terms. Be solution-oriented.", "metadata": {"type": "cultural_context", "region": "US"}},
                {"page_content": "Cultural context for EU: Focus on data privacy, consumer rights, and polite, formal language. Be comprehensive.", "metadata": {"type": "cultural_context", "region": "EU"}},
                {"page_content": "Cultural context for Asia: Respectful language, indirect communication can be preferred, focus on harmony and clear step-by-step guidance. Politeness is key.", "metadata": {"type": "cultural_context", "region": "Asia"}},
                {"page_content": "Cultural context for Latin America: Emphasize personal connection, warmth, and be patient. Provide detailed explanations. Politeness and respect are highly valued.", "metadata": {"type": "cultural_context", "region": "Latin America"}},
            ]
            self.vectorstore.add_texts([d["page_content"] for d in documents], metadatas=[d["metadata"] for d in documents])
            self.vectorstore.persist()
        else:
            logger.info("ChromaDB already populated.")

    def get_relevant_docs(self, query: str, k: int = 5, filter_type: Optional[str] = None) -> List[Dict]:
        if filter_type:
            docs = self.vectorstore.similarity_search(query, k=k, filter={"type": filter_type})
        else:
            docs = self.vectorstore.similarity_search(query, k=k)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

# --- Prompt Engineering & Context Manager --- #
class CulturalAwarenessModule:
    def __init__(self):
        self.cultural_directives = {
            "US": "Be direct, efficient, and focus on practical solutions. Use clear, concise language.",
            "EU": "Prioritize data privacy, consumer rights, and formal, polite language. Provide comprehensive information.",
            "Asia": "Use highly respectful and polite language. Provide clear, step-by-step guidance, and emphasize harmony. Indirect communication might be preferred.",
            "Latin America": "Be warm, personable, and patient. Provide detailed explanations and show high respect. Personal connection is important.",
            "default": "Be helpful, clear, and polite. Focus on resolving the customer's issue efficiently."
        }

    def inject_cultural_context(self, base_prompt: str, region: str) -> str:
        directive = self.cultural_directives.get(region, self.cultural_directives["default"])
        logger.debug(f"Applying cultural directive for region {region}: {directive}")
        return f"{directive}\n\n{base_prompt}"


class BalancedDemonstrationsSelector:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def select_demonstrations(
        self, query: str, num_examples: int = 4, balance_criteria_key: str = "type"
    ) -> List[str]:
        """Selects balanced demonstrations by querying the vector store and trying to get diverse types."""
        all_relevant_docs = self.knowledge_base.get_relevant_docs(query, k=num_examples * 2) # Get more to choose from
        
        # Group docs by balance criteria
        grouped_docs = {}
        for doc in all_relevant_docs:
            criteria_value = doc["metadata"].get(balance_criteria_key, "other")
            if criteria_value not in grouped_docs:
                grouped_docs[criteria_value] = []
            grouped_docs[criteria_value].append(doc)
        
        selected_demonstrations = []
        # Distribute selections across groups to ensure balance
        num_groups = len(grouped_docs)
        if num_groups == 0: # No relevant docs found
            logger.warning("No relevant documents found for demonstrations.")
            return []

        # Simple round-robin selection to pick from different types
        group_keys = list(grouped_docs.keys())
        idx_in_group = {key: 0 for key in group_keys}

        for i in range(num_examples):
            if not group_keys: # All groups exhausted
                break
            
            current_group_key = group_keys[i % len(group_keys)]
            if idx_in_group[current_group_key] < len(grouped_docs[current_group_key]):
                selected_demonstrations.append(grouped_docs[current_group_key][idx_in_group[current_group_key]]["page_content"])
                idx_in_group[current_group_key] += 1
            else:
                # If a group is exhausted, remove it from consideration for round-robin
                group_keys.pop(i % len(group_keys))
                i -= 1 # Re-evaluate the current index with the new list
            
        logger.debug(f"Selected {len(selected_demonstrations)} demonstrations.")
        return selected_demonstrations


class DenseModule:
    def create_dense_prompts(self, base_system_message: str, user_question: str, demonstrations: List[str], num_variations: int = 3) -> List[List[SystemMessage | HumanMessage]]:
        """Generates multiple distinct few-shot prompts with varying exemplar subsets."""
        if not demonstrations:
            logger.warning("No demonstrations provided for DENSE module.")
            return [[SystemMessage(content=base_system_message), HumanMessage(content=user_question)]]

        prompts = []
        for i in range(num_variations):
            # Shuffle and take a subset of demonstrations for each prompt variation
            shuffled_demonstrations = random.sample(demonstrations, min(len(demonstrations), max(1, len(demonstrations) // num_variations)))
            
            current_prompt_messages = [SystemMessage(content=base_system_message)]
            for demo in shuffled_demonstrations:
                # Assuming demonstrations are simple Q&A pairs embedded in 'page_content'
                qa_parts = demo.split('A:', 1)
                if len(qa_parts) == 2:
                    current_prompt_messages.append(HumanMessage(content=qa_parts[0].strip()))
                    current_prompt_messages.append(AIMessage(content=qa_parts[1].strip()))
                else:
                    # If not a Q&A, treat as general context or instruction
                    current_prompt_messages.append(SystemMessage(content=demo))
            
            current_prompt_messages.append(HumanMessage(content=user_question))
            prompts.append(current_prompt_messages)

        logger.debug(f"Generated {len(prompts)} DENSE prompts.")
        return prompts


# --- Bias Mitigation & Refinement Module --- #
class BiasMitigationModule:
    def detect_and_mitigate_bias(self, response: str) -> Tuple[str, bool, Optional[str]]:
        """Simulates bias detection and suggests mitigation."""
        bias_keywords = [
            "unfortunately, we cannot", "only for premium members", "based on your demographics",
            "typical for someone from", "you are not eligible", "restricted to certain regions"
        ]
        
        detected = False
        disclaimer = None

        # Simple keyword-based detection
        for keyword in bias_keywords:
            if keyword in response.lower():
                detected = True
                disclaimer = "Disclaimer: This response might contain language that could be perceived as biased or exclusionary. We are continuously working to improve fairness and inclusivity in our automated responses. For further assistance, please contact a human agent."
                logger.warning(f"Bias detected based on keyword: '{keyword}' in response.")
                break
        
        # In a real system, this would involve a separate LLM call for rephrasing or a more sophisticated bias classifier.
        # For this demo, we just add a disclaimer.
        return response, detected, disclaimer


# --- LLM Orchestrator --- #
class LLMOrchestrator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)

    def process_inquiry(self, prompts: List[List[SystemMessage | HumanMessage]]) -> Tuple[str, List[str]]:
        """Sends multiple prompts to the LLM and aggregates responses."""
        raw_responses = []
        for prompt_messages in prompts:
            try:
                logger.debug(f"Sending prompt to LLM: {prompt_messages}")
                response = self.llm.invoke(prompt_messages)
                raw_responses.append(response.content)
            except Exception as e:
                logger.error(f"Error invoking LLM: {e}")
                raw_responses.append(f"Error processing inquiry: {e}")
        
        # Simple aggregation: If multiple responses, combine them or take the first non-error one.
        # A more sophisticated approach would involve consensus building, voting, or summarization.
        if not raw_responses:
            aggregated_response = "I am sorry, I could not process your request at this time."
        elif len(raw_responses) == 1:
            aggregated_response = raw_responses[0]
        else:
            # For demonstration, let's pick the first non-error response, or combine if all are fine.
            valid_responses = [res for res in raw_responses if not res.startswith("Error processing inquiry")]
            if valid_responses:
                aggregated_response = valid_responses[0] # Simplistic: take first valid
                if len(valid_responses) > 1:
                    logger.info("Multiple valid responses received. Using the first one for aggregation.")
                    # Or, for more complex aggregation:
                    # aggregated_response = f"I received a few perspectives: {'. '.join(valid_responses)}"
            else:
                aggregated_response = "I encountered multiple issues and cannot provide a clear answer."

        logger.debug(f"Aggregated LLM response: {aggregated_response[:100]}...")
        return aggregated_response, raw_responses


# --- FastAPI Application --- #
app = FastAPI(
    title="Intelligent & Ethical Customer Support Agent",
    description="An AI agent leveraging advanced LLM prompting for accurate, unbiased, and culturally sensitive customer service."
)

# Global instances of modules
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
knowledge_base: KnowledgeBase = None # Will be initialized in startup event
cultural_awareness_module = CulturalAwarenessModule()
balanced_demos_selector: BalancedDemonstrationsSelector = None # Will be initialized in startup event
dense_module = DenseModule()
bias_mitigation_module = BiasMitigationModule()
llm_orchestrator = LLMOrchestrator(openai_api_key=OPENAI_API_KEY)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing ChromaDB and modules.")
    global knowledge_base, balanced_demos_selector
    knowledge_base = KnowledgeBase(CHROMA_PERSIST_DIR, embeddings)
    balanced_demos_selector = BalancedDemonstrationsSelector(knowledge_base)
    logger.info("ChromaDB and modules initialized.")

@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(
    inquiry: CustomerInquiry,
    db: Session = Depends(get_db)
):
    logger.info(f"Received inquiry from customer {inquiry.customer_id} (Region: {inquiry.region}): {inquiry.question}")

    # 1. Cultural Awareness
    base_system_prompt = "You are an AI customer support agent. Provide helpful, accurate, and concise answers."
    culturally_aware_system_prompt = cultural_awareness_module.inject_cultural_context(
        base_system_prompt, inquiry.region
    )

    # 2. Balanced Demonstrations Selection
    demonstrations = balanced_demos_selector.select_demonstrations(inquiry.question)
    
    # 3. Demonstration Ensembling (DENSE) Prompt Generation
    dense_prompts = dense_module.create_dense_prompts(
        culturally_aware_system_prompt, inquiry.question, demonstrations
    )
    
    # 4. LLM Inference & Aggregation
    aggregated_llm_response, llm_raw_responses = llm_orchestrator.process_inquiry(dense_prompts)

    # 5. Bias Mitigation & Refinement
    final_answer, bias_detected, bias_disclaimer = bias_mitigation_module.detect_and_mitigate_bias(
        aggregated_llm_response
    )

    # 6. Store Conversation History
    conversation = ConversationHistory(
        customer_id=inquiry.customer_id,
        region=inquiry.region,
        inquiry=inquiry.question,
        initial_prompts=str([str(p) for p in dense_prompts]), # Convert list of messages to string
        llm_raw_responses="\n---\n".join(llm_raw_responses), # Join raw responses
        final_agent_response=final_answer,
        bias_detected=str(bias_detected),
        bias_disclaimer=bias_disclaimer
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    logger.info(f"Conversation logged with ID: {conversation.id}")

    # 7. Monitoring & Evaluation (simulated)
    logger.info(f"[Monitoring] Bias Detected: {bias_detected}")
    logger.info(f"[Monitoring] Final Answer Length: {len(final_answer)}")

    return AgentResponse(
        answer=final_answer,
        bias_detected=bias_detected,
        bias_disclaimer=bias_disclaimer
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Agent is running."}

# Example of how to run this application:
# 1. Create a .env file with OPENAI_API_KEY="your_openai_api_key_here"
# 2. Install dependencies: pip install fastapi uvicorn python-dotenv loguru sqlalchemy langchain-openai langchain-community chromadb
# 3. Run: uvicorn main:app --reload --port 8000
# 4. Access API at http://localhost:8000/docs
