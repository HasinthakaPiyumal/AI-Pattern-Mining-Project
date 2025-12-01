from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate


@tool
def get_medical_info(query: str) -> str:
    """Provides information on diseases, symptoms, and treatments. Input should be a specific medical query."""
    medical_data = {
        "fever": "Fever is a temporary increase in your body temperature, often due to an illness. Treatment often involves rest and hydration.",
        "headache": "Headaches are a common condition that most people will experience many times in their lives. Treatments range from over-the-counter pain relievers to prescription medications.",
        "diabetes": "Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Management involves diet, exercise, and medication.",
        "hypertension": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Treatment often includes lifestyle changes and medication."
    }
    return medical_data.get(query.lower(), "Information not found for your query. Please try a more specific medical term.")

@tool
def check_drug_interactions(drugs: str) -> str:
    """Checks for potential drug-drug interactions. Input should be a comma-separated list of drug names."""
    drugs_list = [d.strip().lower() for d in drugs.split(',')]
    interactions = {
        ("ibuprofen", "warfarin"): "Increased risk of bleeding.",
        ("metformin", "alcohol"): "Increased risk of lactic acidosis.",
        ("amoxicillin", "methotrexate"): "Increased methotrexate toxicity."
    }
    found_interactions = []
    for i in range(len(drugs_list)):
        for j in range(i + 1, len(drugs_list)):
            pair1 = tuple(sorted((drugs_list[i], drugs_list[j])))
            if pair1 in interactions:
                found_interactions.append(f"Interaction between {drugs_list[i].capitalize()} and {drugs_list[j].capitalize()}: {interactions[pair1]}")
    
    if not found_interactions:
        return "No significant interactions found for the specified drugs, or drugs are unknown."
    return "\n".join(found_interactions)

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculates the Body Mass Index (BMI). Requires weight in kilograms and height in meters."""
    if height_m <= 0:
        return "Error: Height must be greater than zero."
    bmi = weight_kg / (height_m ** 2)
    category = ""
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"
    return f"Your BMI is {bmi:.2f}, which falls into the '{category}' category."


# Initialize LLM
# Ensure OPENAI_API_KEY environment variable is set
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Define tools
tools = [
    get_medical_info,
    check_drug_interactions,
    calculate_bmi
]

# Define the agent's prompt
prompt = PromptTemplate.from_template("""You are a helpful medical assistant. You have access to various medical tools.
Use the tools to answer questions and provide information to healthcare professionals.
If a question requires multiple steps or tools, break it down and use the tools sequentially.
Always be polite and informative.

TOOLS:
{tools}

FORMAT INSTRUCTIONS:
{format_instructions}

USER'S INPUT: {input}

{agent_scratchpad}""")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("Welcome to the Medical Diagnostic Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("\nMedical Professional: ")
        if user_input.lower() == 'exit':
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print("Assistant:", response["output"])
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your query or check the input format.")
