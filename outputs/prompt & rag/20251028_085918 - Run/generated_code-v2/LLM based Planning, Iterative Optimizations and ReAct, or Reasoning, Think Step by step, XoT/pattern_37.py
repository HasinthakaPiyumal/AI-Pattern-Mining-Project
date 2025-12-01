
import random
import json

class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class KnowledgeBase:
    def __init__(self):
        self.data = {
            "shipping_policy": "Our standard shipping takes 3-5 business days. Expedited options are available.",
            "return_policy": "Items can be returned within 30 days of purchase with a valid receipt.",
            "refund_process": "Refunds are processed within 5-7 business days after the returned item is received.",
            "account_reset": "To reset your account password, please visit our website and click 'Forgot Password'."
        }

    def search(self, query):
        query_lower = query.lower()
        for key, value in self.data.items():
            if key in query_lower or query_lower in value.lower():
                return f"Found information about '{key}': {value}"
        return "No relevant information found in the knowledge base."

class StudentLLM:
    def __init__(self, name="StudentLLM"):
        self.name = name
        self.error_rate = 0.3 # Simulate errors

    def generate_response(self, query, conversation_history, available_tools):
        history_str = "\n".join(conversation_history)
        prompt = f"Conversation History:\n{history_str}\nCustomer: {query}\nAgent: "

        response = ""
        tool_call = None

        if "shipping" in query.lower() or "delivery" in query.lower():
            response = "Let me check our shipping policy for you."
            tool_call = {"tool": "search_knowledge_base", "args": {"query": "shipping policy"}}
        elif "return" in query.lower() or "refund" in query.lower():
            response = "I can help you with returns and refunds."
            tool_call = {"tool": "search_knowledge_base", "args": {"query": "return policy"}}
        elif "password" in query.lower() or "account" in query.lower():
            response = "I can help with account related issues."
            tool_call = {"tool": "search_knowledge_base", "args": {"query": "account reset"}}
        else:
            response = "Thank you for contacting customer support. How can I assist you further?"

        # Introduce simulated errors for the student model
        if random.random() < self.error_rate:
            if tool_call and tool_call["tool"] == "search_knowledge_base":
                tool_call["args"]["query"] = "non_existent_policy" # Wrong tool argument
                response = "I'm having trouble finding that information, please hold."
            else:
                response = "I'm sorry, I didn't understand that. Can you please rephrase?" # Generic error

        return response, tool_call

class TeacherLLM:
    def __init__(self, name="TeacherLLM"):
        self.name = name

    def correct_and_complete_trajectory(self, initial_segment, student_trajectory, available_tools):
        print(f"[{self.name}] Correcting and completing trajectory...")
        corrected_trajectory = list(initial_segment) # Start with the valid initial segment

        # Simulate sophisticated correction logic based on common errors
        for i in range(len(initial_segment), len(student_trajectory)):
            step = student_trajectory[i]
            customer_query = step.get("customer_query")
            student_response = step.get("agent_response")
            student_tool_call = step.get("tool_call")
            tool_output = step.get("tool_output")

            corrected_agent_response = student_response
            corrected_tool_call = student_tool_call

            # Example correction: If student used a wrong tool query, correct it
            if student_tool_call and student_tool_call["tool"] == "search_knowledge_base" and \
               student_tool_call["args"].get("query") == "non_existent_policy":
                print(f"[{self.name}] Correcting tool call for query: '{customer_query}'")
                if "shipping" in customer_query.lower() or "delivery" in customer_query.lower():
                    corrected_tool_call["args"]["query"] = "shipping policy"
                    corrected_agent_response = "Let me properly look up the shipping policy for you."
                elif "return" in customer_query.lower() or "refund" in customer_query.lower():
                    corrected_tool_call["args"]["query"] = "return policy"
                    corrected_agent_response = "I've identified the correct return policy to share."
                else:
                    # Fallback or more advanced logic for other cases
                    corrected_tool_call = None # Discard incorrect tool call if unsure
                    corrected_agent_response = "Apologies for the confusion, let me re-evaluate your request."
            elif "didn't understand" in student_response.lower() or "apologies" in student_response.lower():
                 print(f"[{self.name}] Completing generic error response for query: '{customer_query}'")
                 if "shipping" in customer_query.lower():
                     corrected_agent_response = "I understand you're asking about shipping. Here's our policy: [SHIPPING_POLICY_PLACEHOLDER]"
                     corrected_tool_call = {"tool": "search_knowledge_base", "args": {"query": "shipping policy"}}
                 elif "return" in customer_query.lower():
                     corrected_agent_response = "You're looking for return information. Let me find that for you: [RETURN_POLICY_PLACEHOLDER]"
                     corrected_tool_call = {"tool": "search_knowledge_base", "args": {"query": "return policy"}}
                 else:
                     corrected_agent_response = "Let's try that again. Could you please specify your query?"
                     corrected_tool_call = None

            # Simulate tool execution for corrected calls to get proper output
            corrected_tool_output = None
            if corrected_tool_call:
                tool_name = corrected_tool_call["tool"]
                tool_args = corrected_tool_call["args"]
                if tool_name in available_tools:
                    print(f"[{self.name}] Executing corrected tool: {tool_name} with args {tool_args}")
                    # In a real scenario, this would call the actual tool object
                    # For simulation, we manually call the KB search
                    if tool_name == "search_knowledge_base":
                        kb_tool = available_tools[tool_name]
                        corrected_tool_output = kb_tool(**tool_args)
                else:
                    corrected_tool_output = f"Error: Tool '{tool_name}' not found."

            corrected_trajectory.append({
                "turn": len(corrected_trajectory) + 1,
                "customer_query": customer_query,
                "agent_response": corrected_agent_response,
                "tool_call": corrected_tool_call,
                "tool_output": corrected_tool_output,
            })

        return corrected_trajectory


class SimulationEnvironment:
    def __init__(self, student_llm, teacher_llm, available_tools_map):
        self.student_llm = student_llm
        self.teacher_llm = teacher_llm
        self.available_tools_map = available_tools_map
        self.knowledge_base = KnowledgeBase()
        self.tools = {
            "search_knowledge_base": Tool("search_knowledge_base", "Searches the customer knowledge base.", self.knowledge_base.search),
            # Add other tools here like 'initiate_refund', 'check_order_status'
        }

    def generate_customer_query(self):
        queries = [
            "I need to know your shipping times.",
            "What's your policy on returns?",
            "My package hasn't arrived, can you help?",
            "How do I reset my account password?",
            "I want a refund for a faulty item.",
            "I have a general question about your services."
        ]
        return random.choice(queries)

    def simulate_interaction(self, query, max_turns=5):
        conversation_history = []
        trajectory = []
        print(f"\n--- Simulating Interaction for: '{query}' ---")

        for turn in range(1, max_turns + 1):
            print(f"[Turn {turn}] Customer: {query}")
            agent_response, tool_call = self.student_llm.generate_response(query, conversation_history, self.tools)

            tool_output = None
            if tool_call:
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("args", {})
                if tool_name and tool_name in self.tools:
                    print(f"[Turn {turn}] Agent attempts tool call: {tool_name} with args {tool_args}")
                    tool_output = self.tools[tool_name](**tool_args)
                else:
                    tool_output = f"Error: Tool '{tool_name}' not recognized or available."

            trajectory.append({
                "turn": turn,
                "customer_query": query,
                "agent_response": agent_response,
                "tool_call": tool_call,
                "tool_output": tool_output,
            })

            conversation_history.append(f"Customer: {query}")
            conversation_history.append(f"Agent: {agent_response}")
            if tool_output:
                conversation_history.append(f"Tool Output: {tool_output}")

            print(f"[Turn {turn}] Agent: {agent_response}")
            if tool_output:
                print(f"[Turn {turn}] Tool Output: {tool_output}")

            # For simplicity, we stop after one meaningful exchange or max_turns
            if tool_call and tool_output and "Error" not in tool_output and "didn't understand" not in agent_response.lower():
                 # Simulate customer satisfaction or a follow-up if needed
                if "shipping policy" in str(tool_output).lower() or "return policy" in str(tool_output).lower():
                    query = "Thank you, that clarifies things." # End of a simple interaction
                else:
                    query = self.generate_customer_query() # For more complex interaction, generate new query
            else:
                query = self.generate_customer_query() # Continue with a new query or rephrase

            if turn == max_turns:
                print(f"--- Interaction Ended (Max Turns Reached) ---")
                break
            elif "thank you" in query.lower() or "goodbye" in query.lower():
                print(f"--- Interaction Ended (Customer Satisfied) ---")
                break

        return trajectory

class TrajectoryEvaluationModule:
    def evaluate(self, trajectory):
        is_valid = True
        problematic_segments = [] # Indices or descriptions of problematic turns

        for i, step in enumerate(trajectory):
            agent_response = step.get("agent_response", "").lower()
            tool_call = step.get("tool_call")
            tool_output = step.get("tool_output", "").lower()

            if "i'm sorry, i didn't understand" in agent_response or "apologies for the confusion" in agent_response:
                is_valid = False
                problematic_segments.append(f"Turn {step['turn']}: Generic understanding error.")
            if tool_call and "non_existent_policy" in str(tool_call.get("args", {}).get("query", "")).lower():
                is_valid = False
                problematic_segments.append(f"Turn {step['turn']}: Incorrect tool argument used.")
            if tool_output and "no relevant information found" in tool_output:
                is_valid = False
                problematic_segments.append(f"Turn {step['turn']}: Tool executed but found no info (possibly wrong query).")
            if tool_call and "error: tool" in tool_output.lower():
                is_valid = False
                problematic_segments.append(f"Turn {step['turn']}: Tool execution error.")

        return {"is_valid": is_valid, "problematic_segments": problematic_segments}

    def get_plausible_initial_segment(self, trajectory):
        initial_segment = []
        for step in trajectory:
            agent_response = step.get("agent_response", "").lower()
            tool_call = step.get("tool_call")
            tool_output = step.get("tool_output", "").lower()

            is_problematic = False
            if "i'm sorry, i didn't understand" in agent_response or "apologies for the confusion" in agent_response:
                is_problematic = True
            if tool_call and "non_existent_policy" in str(tool_call.get("args", {}).get("query", "")).lower():
                is_problematic = True
            if tool_output and "no relevant information found" in tool_output:
                is_problematic = True
            if tool_call and "error: tool" in tool_output.lower():
                is_problematic = True

            if is_problematic:
                break
            initial_segment.append(step)
        return initial_segment


# Main execution flow
if __name__ == "__main__":
    # 1. Initialize Components
    student_llm = StudentLLM()
    teacher_llm = TeacherLLM()
    kb = KnowledgeBase()

    available_tools_map = {
        "search_knowledge_base": kb.search,
        # Placeholder for other tools if they existed
    }

    sim_env = SimulationEnvironment(student_llm, teacher_llm, available_tools_map)
    eval_module = TrajectoryEvaluationModule()

    augmented_training_data = []
    num_simulations = 5

    print("\n--- Starting Data Augmentation Process ---")

    for i in range(num_simulations):
        print(f"\n=== Simulation {i+1}/{num_simulations} ===")
        initial_customer_query = sim_env.generate_customer_query()
        student_trajectory = sim_env.simulate_interaction(initial_customer_query, max_turns=3)

        evaluation_result = eval_module.evaluate(student_trajectory)
        print(f"\nEvaluation Result for Student Trajectory {i+1}: {evaluation_result}")

        if not evaluation_result["is_valid"]:
            print(f"Student trajectory {i+1} is invalid or incomplete. Engaging Teacher LLM...")
            initial_segment = eval_module.get_plausible_initial_segment(student_trajectory)
            corrected_trajectory = teacher_llm.correct_and_complete_trajectory(
                initial_segment, student_trajectory, sim_env.tools
            )
            augmented_training_data.append({"original": student_trajectory, "corrected": corrected_trajectory})
            print(f"Teacher corrected trajectory {i+1}:\n{json.dumps(corrected_trajectory, indent=2)}")
        else:
            print(f"Student trajectory {i+1} is valid. Adding to training data as is.")
            augmented_training_data.append({"original": student_trajectory, "corrected": student_trajectory}) # No correction needed

    print("\n--- Data Augmentation Process Complete ---")
    print(f"Total augmented data points: {len(augmented_training_data)}")

    # 6. Training & Fine-tuning Module (Placeholder)
    print("\n--- Placeholder for Training & Fine-tuning Student LLM ---")
    print("In a real scenario, 'augmented_training_data' would be used to fine-tune the Student LLM.")
    print("This would involve converting the trajectories into a suitable format (e.g., prompt-response pairs) and using an LLM fine-tuning library (e.g., Hugging Face's TRL or a custom PyTorch/TensorFlow loop).")

    # Example of how a corrected trajectory might look for training
    if augmented_training_data:
        print("\nExample of augmented data for training (first entry):")
        print(json.dumps(augmented_training_data[0]['corrected'], indent=2))

