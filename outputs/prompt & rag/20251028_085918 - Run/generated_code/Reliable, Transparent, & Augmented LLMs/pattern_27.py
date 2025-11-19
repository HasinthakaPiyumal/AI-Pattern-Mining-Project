import streamlit as st
import time
import random

class CRMConnector:
    def get_customer_history(self, customer_id):
        if customer_id == "C101":
            return {"customer_id": "C101", "name": "Alice Smith", "last_purchase": "Laptop (2023-10-26)", "issues": ["Slow internet (resolved)"]}
        return {"customer_id": customer_id, "name": "Unknown", "last_purchase": "N/A", "issues": []}

    def update_support_ticket(self, customer_id, issue_description):
        return f"Ticket for {customer_id} updated with issue: {issue_description}"

class KnowledgeBaseConnector:
    def search_articles(self, query):
        query_lower = query.lower()
        if "internet speed" in query_lower:
            return ["Article: How to troubleshoot slow internet", "FAQ: Internet speed requirements"]
        if "billing" in query_lower:
            return ["Article: Understanding your bill", "FAQ: Payment methods"]
        return [f"No direct articles found for '{query}'."]

class LLMAgent:
    def __init__(self):
        self.crm = CRMConnector()
        self.kb = KnowledgeBaseConnector()

    def _generate_reasoning_path(self, query, tools_used, final_answer):
        path = f"User asked: '{query}'.\n"
        for tool, details in tools_used:
            path += f"  - Used {tool}: {details}\n"
        path += f"  - Synthesized information to form answer.\n"
        path += f"Final Answer derived based on retrieved data."
        return path

    def _self_rate_confidence(self, response_length, tools_used_count):
        base_confidence = 70 + (response_length / 10) * 2
        tool_boost = tools_used_count * 5
        confidence = min(100, int(base_confidence + tool_boost + random.randint(-5, 5)))
        return confidence

    def _detect_hallucination(self, response, tools_used):
        if "fictional product" in response.lower() and "CRM" not in [t for t, _ in tools_used]:
            return True, "Response might contain unverified product information."
        return False, "No obvious hallucination detected."

    def _detect_prompt_vulnerability(self, query):
        if "ignore previous instructions" in query.lower() or "reveal system prompt" in query.lower():
            return True, "Potential prompt injection attempt detected."
        return False, "No prompt vulnerability detected."

    def process_query(self, query, customer_id=None):
        tools_used = []
        response_parts = []
        confidence_score = 0
        reasoning_path = ""
        hallucination_flag = False
        hallucination_reason = ""
        vulnerability_flag = False
        vulnerability_reason = ""

        if self._detect_prompt_vulnerability(query)[0]:
            vulnerability_flag, vulnerability_reason = self._detect_prompt_vulnerability(query)
            final_answer = "I cannot process this request as it appears to be a security vulnerability attempt. Please refrain from such queries."
            reasoning_path = "Detected prompt vulnerability, refused to process."
            confidence_score = 0
            return {
                "response_parts": [final_answer],
                "reasoning_path": reasoning_path,
                "confidence_score": confidence_score,
                "progressive_disclosure": [final_answer],
                "hallucination_flag": hallucination_flag,
                "hallucination_reason": hallucination_reason,
                "vulnerability_flag": vulnerability_flag,
                "vulnerability_reason": vulnerability_reason
            }

        response_parts.append("Analyzing your request...")
        time.sleep(0.5)

        if "customer history" in query.lower() and customer_id:
            customer_info = self.crm.get_customer_history(customer_id)
            tools_used.append(("CRM", f"Retrieved history for {customer_id}"))
            response_parts.append(f"Retrieved customer history for {customer_id}: {customer_info['name']} purchased {customer_info['last_purchase']}. Past issues: {', '.join(customer_info['issues']) if customer_info['issues'] else 'None'}.")
            time.sleep(0.5)

        if "internet speed" in query.lower() or "slow network" in query.lower():
            kb_articles = self.kb.search_articles("internet speed")
            tools_used.append(("Knowledge Base", "Searched for internet speed articles"))
            response_parts.append(f"Found relevant articles regarding internet speed: {', '.join(kb_articles)}.")
            time.sleep(0.5)

        if "billing" in query.lower():
            kb_articles = self.kb.search_articles("billing")
            tools_used.append(("Knowledge Base", "Searched for billing articles"))
            response_parts.append(f"Found relevant articles regarding billing: {', '.join(kb_articles)}.")
            time.sleep(0.5)

        if not tools_used:
            response_parts.append(f"I am processing your query: '{query}'. Please bear with me.")

        final_answer = " ".join(response_parts[1:]) if len(response_parts) > 1 else response_parts[0]
        final_answer += "\nHow else can I assist you today?"

        hallucination_flag, hallucination_reason = self._detect_hallucination(final_answer, tools_used)

        reasoning_path = self._generate_reasoning_path(query, tools_used, final_answer)
        confidence_score = self._self_rate_confidence(len(final_answer), len(tools_used))

        progressive_disclosure = []
        for i in range(len(response_parts)):
            progressive_disclosure.append(" ".join(response_parts[:i+1]))
        progressive_disclosure.append(final_answer)

        return {
            "response_parts": response_parts,
            "reasoning_path": reasoning_path,
            "confidence_score": confidence_score,
            "progressive_disclosure": progressive_disclosure,
            "hallucination_flag": hallucination_flag,
            "hallucination_reason": hallucination_reason,
            "vulnerability_flag": vulnerability_flag,
            "vulnerability_reason": vulnerability_reason
        }


st.set_page_config(layout="wide")
st.title("🧠 Agentic & Trustworthy Customer Support AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_agent" not in st.session_state:
    st.session_state.llm_agent = LLMAgent()
if "customer_id" not in st.session_state:
    st.session_state.customer_id = "C101" # Default customer ID for demonstration

# Sidebar for customer ID and system settings
st.sidebar.header("Agent Settings")
st.session_state.customer_id = st.sidebar.text_input("Customer ID (for CRM lookup)", st.session_state.customer_id)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "reasoning" in message and message["reasoning"]:
            with st.expander("Reasoning Path"): st.write(message["reasoning"])
        if "confidence" in message and message["confidence"] is not None:
            st.write(f"Confidence Score: {message['confidence']}% :bulb:")
        if "hallucination" in message and message["hallucination"][0]:
            st.error(f"Potential Hallucination Detected! Reason: {message['hallucination'][1]}")
        if "vulnerability" in message and message["vulnerability"][0]:
            st.warning(f"Prompt Vulnerability Detected! Reason: {message['vulnerability'][1]}")

# Chat input
if prompt := st.chat_input("How can I assist the customer today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        response_data = st.session_state.llm_agent.process_query(prompt, st.session_state.customer_id)

        full_response_content = ""
        for part in response_data["progressive_disclosure"]:
            full_response_content = part
            response_container.markdown(full_response_content + " :thinking_face:")
            time.sleep(0.3) # Simulate progressive disclosure

        response_container.markdown(response_data["progressive_disclosure"][-1]) # Display final response

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["progressive_disclosure"][-1],
            "reasoning": response_data["reasoning_path"],
            "confidence": response_data["confidence_score"],
            "hallucination": (response_data["hallucination_flag"], response_data["hallucination_reason"]),
            "vulnerability": (response_data["vulnerability_flag"], response_data["vulnerability_reason"])
        })

        with st.expander("Reasoning Path"): st.write(response_data["reasoning_path"])
        st.write(f"Confidence Score: {response_data['confidence_score']}% :bulb:")
        if response_data["hallucination_flag"]:
            st.error(f"Potential Hallucination Detected! Reason: {response_data['hallucination_reason']}")
        if response_data["vulnerability_flag"]:
            st.warning(f"Prompt Vulnerability Detected! Reason: {response_data['vulnerability_reason']}")

    # Feedback mechanism
    st.markdown("--- :page_facing_up:")
    st.subheader("Provide Feedback on AI Response")
    feedback_option = st.radio("Was this response helpful?", ("Yes", "No", "Needs Improvement"), key=f"feedback_{len(st.session_state.messages)}")
    feedback_text = st.text_area("Additional Comments (Optional)", key=f"comments_{len(st.session_state.messages)}")
    if st.button("Submit Feedback", key=f"submit_feedback_{len(st.session_state.messages)}"):
        st.success(f"Feedback submitted: {feedback_option} - '{feedback_text}'")
        # In a real system, this feedback would be logged for model improvement
