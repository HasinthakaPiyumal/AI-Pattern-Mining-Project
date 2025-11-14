import streamlit as st
import autogen
import os

# --- Configuration for AutoGen and LLMs ---
# In a real application, you would load these from environment variables or a config file
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# For demonstration, we'll use a mock LLM config. 
# If using actual OpenAI, uncomment and configure as below:
# config_list = autogen.config_list_from_json(
#     "OAI_CONFIG_LIST",
#     filter_json={"model": ["gpt-4", "gpt-3.5-turbo"]},
# )

# Mock config for demonstration purposes without actual API calls
config_list = [
    {
        "model": "gpt-4-mock",  # Placeholder model name
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # Placeholder API key
        "base_url": "http://localhost:1234/v1" # Placeholder base URL
    }
]

# --- AutoGen Agent Definitions ---

def create_essay_evaluation_agents(essay_text):
    llm_config = {
        "config_list": config_list,
        "temperature": 0.7,
        "timeout": 120,
    }

    # User Proxy Agent to initiate the conversation
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("FINAL GRADE:"),
        code_execution_config=False,  # No code execution needed for this task
        llm_config=llm_config
    )

    # Expert Agents with specific personas
    grammar_expert = autogen.AssistantAgent(
        name="GrammarExpert",
        llm_config=llm_config,
        system_message=f"You are a highly analytical Grammar Expert. Your task is to meticulously review the provided essay for grammatical errors, spelling mistakes, punctuation issues, and syntactical correctness. Provide specific examples and suggestions for improvement. After your initial assessment, discuss with other experts to reach a consensus. The essay to evaluate is: '''{essay_text}'''",
    )

    content_expert = autogen.AssistantAgent(
        name="ContentCoherenceExpert",
        llm_config=llm_config,
        system_message=f"You are a Content Coherence Expert. Focus on the logical flow of ideas, organization, topic sentences, paragraph transitions, and overall clarity of the essay's content. Assess if the arguments are well-supported and easy to follow. Provide constructive feedback. After your initial assessment, discuss with other experts to reach a consensus. The essay to evaluate is: '''{essay_text}'''",
    )

    critical_thinking_expert = autogen.AssistantAgent(
        name="CriticalThinkingExpert",
        llm_config=llm_config,
        system_message=f"You are a Critical Thinking Expert. Evaluate the essay's depth of analysis, originality of thought, ability to address counterarguments, and the strength of its thesis. Assess if the student demonstrates insightful reasoning and independent thought. Provide specific areas for improvement. After your initial assessment, discuss with other experts to reach a consensus. The essay to evaluate is: '''{essay_text}'''",
    )

    creativity_expert = autogen.AssistantAgent(
        name="CreativityExpert",
        llm_config=llm_config,
        system_message=f"You are a Creativity Expert. Assess the essay for originality, engaging language, unique perspectives, and imaginative expression. Is the essay captivating and does it go beyond typical responses? Provide feedback on how to enhance creative elements. After your initial assessment, discuss with other experts to reach a consensus. The essay to evaluate is: '''{essay_text}'''",
    )

    return user_proxy, grammar_expert, content_expert, critical_thinking_expert, creativity_expert

def run_essay_evaluation(essay_text):
    if not essay_text.strip():
        return "", "Please enter an essay to evaluate."

    user_proxy, grammar_expert, content_expert, critical_thinking_expert, creativity_expert = create_essay_evaluation_agents(essay_text)

    # Simulate initial evaluations and then a debate
    # In a real AutoGen scenario, agents would communicate directly based on their system messages.
    # For this demonstration, we'll guide the conversation to a resolution.
    groupchat = autogen.GroupChat(
        agents=[user_proxy, grammar_expert, content_expert, critical_thinking_expert, creativity_expert],
        messages=[], 
        max_round=15,
        speaker_selection_method="auto",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config_list)

    # Initiate the conversation
    st.info("Starting multi-perspective essay evaluation...")
    chat_result = user_proxy.initiate_chat(
        manager,
        message=f"Please evaluate the following essay from your respective expert perspectives and then collectively agree on a final grade (A+, A, B+, B, C+, C, D, F) and detailed consolidated feedback. The essay is: '''{essay_text}'''\n\nEach expert should first provide their initial assessment and proposed grade, then debate to refine and agree on a final collective grade and feedback. The final output must end with 'FINAL GRADE:' followed by the grade and then the consolidated feedback."
    )

    # Extract final grade and feedback
    final_message = chat_result.last_message["content"]
    if "FINAL GRADE:" in final_message:
        parts = final_message.split("FINAL GRADE:", 1)
        grade = parts[1].strip().split('\n')[0].strip()
        feedback = parts[1].strip()
        # Remove the grade line from feedback for clean display
        feedback = feedback.replace(grade, "", 1).strip()
        return grade, feedback
    else:
        return "N/A", "Could not extract final grade and feedback. Review chat history for details."

# --- Streamlit UI --- 
st.set_page_config(page_title="Multi-Perspective LLM Essay Grader", layout="wide")
st.title("📝 Multi-Perspective LLM Essay Grader")
st.markdown("Upload your essay below and get feedback from various expert LLM agents!")

essay_input = st.text_area(
    "Enter your essay here:", 
    height=300,
    placeholder="Paste your essay here..."
)

if st.button("Evaluate Essay"):
    if essay_input:
        with st.spinner("Evaluating essay with multiple LLM experts..."):
            final_grade, detailed_feedback = run_essay_evaluation(essay_input)
        
        st.subheader("Final Evaluation")
        st.success(f"**Grade:** {final_grade}")
        st.markdown("**Detailed Feedback:**")
        st.write(detailed_feedback)

        st.info("Note: In a real application, the LLM calls would be made to an actual provider like OpenAI. This demo uses a mock configuration.")
    else:
        st.warning("Please enter an essay to evaluate.")
