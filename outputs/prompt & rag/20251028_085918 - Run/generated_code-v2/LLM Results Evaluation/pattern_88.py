import json
import random

class ConversationDataLoader:
    def load_conversations(self):
        return [
            {
                "conversation_id": "conv_001",
                "turns": [
                    {"speaker": "customer", "text": "I need help with my internet connection.", "turn_id": 1},
                    {"speaker": "chatbot", "text": "Certainly! Can you please describe the issue in more detail?", "turn_id": 2}
                ]
            },
            {
                "conversation_id": "conv_002",
                "turns": [
                    {"speaker": "customer", "text": "My bill seems too high this month.", "turn_id": 1},
                    {"speaker": "chatbot", "text": "I can help you with that. Could you please provide your account number?", "turn_id": 2}
                ]
            },
            {
                "conversation_id": "conv_003",
                "turns": [
                    {"speaker": "customer", "text": "How do I reset my password?", "turn_id": 1},
                    {"speaker": "chatbot", "text": "To reset your password, please visit our website and click on 'Forgot Password'.", "turn_id": 2}
                ]
            }
        ]

class EvaluationPromptGenerator:
    def __init__(self):
        self.evaluation_schema = {
            "Grammar and Fluency": "Score 1-5",
            "Relevance to Customer Query": "Score 1-5",
            "Helpfulness and Accuracy": "Score 1-5",
            "Empathy and Tone": "Score 1-5",
            "Conciseness": "Score 1-5"
        }

    def generate_prompt(self, conversation_context, chatbot_response):
        schema_str = "\n".join([f"- {k}: {v}" for k, v in self.evaluation_schema.items()])
        prompt = f"""Evaluate the following chatbot response based on the criteria below. 
Output a JSON object with scores for each criterion (1-5).

Conversation Context:
{conversation_context}

Chatbot Response:
{chatbot_response}

Evaluation Criteria:
{schema_str}

Output JSON scores:"""
        return prompt

class EvaluatorLLMInterface:
    def call_llm(self, prompt):
        # Simulate LLM response by generating random scores
        scores = {
            "Grammar and Fluency": random.randint(1, 5),
            "Relevance to Customer Query": random.randint(1, 5),
            "Helpfulness and Accuracy": random.randint(1, 5),
            "Empathy and Tone": random.randint(1, 5),
            "Conciseness": random.randint(1, 5)
        }
        return json.dumps(scores)

class EvaluationProcessor:
    def __init__(self, prompt_generator, llm_interface):
        self.prompt_generator = prompt_generator
        self.llm_interface = llm_interface
        self.all_evaluations = []

    def process_conversations(self, conversations):
        for conv in conversations:
            conversation_id = conv["conversation_id"]
            conversation_context = []
            for turn in conv["turns"]:
                conversation_context.append(f"{turn['speaker'].capitalize()}: {turn['text']}")
                if turn["speaker"] == "chatbot":
                    chatbot_response = turn["text"]
                    context_for_prompt = "\n".join(conversation_context[:-1]) # Exclude current chatbot response
                    
                    prompt = self.prompt_generator.generate_prompt(context_for_prompt, chatbot_response)
                    llm_output_json = self.llm_interface.call_llm(prompt)
                    
                    try:
                        evaluation_scores = json.loads(llm_output_json)
                        self.all_evaluations.append({
                            "conversation_id": conversation_id,
                            "turn_id": turn["turn_id"],
                            "chatbot_response": chatbot_response,
                            "scores": evaluation_scores
                        })
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from LLM for conv {conversation_id}, turn {turn['turn_id']}")
            
        return self.all_evaluations

class EvaluationReporter:
    def generate_report(self, evaluations):
        if not evaluations:
            return "No evaluations to report."

        total_scores = {k: [] for k in evaluations[0]["scores"].keys()}

        for eval_item in evaluations:
            for criterion, score in eval_item["scores"].items():
                total_scores[criterion].append(score)
        
        average_scores = {k: sum(v) / len(v) for k, v in total_scores.items()}

        report = "--- Chatbot Performance Evaluation Report ---\n"
        report += f"Total Chatbot Responses Evaluated: {len(evaluations)}\n\n"
        report += "Average Scores per Criterion:\n"
        for criterion, avg_score in average_scores.items():
            report += f"- {criterion}: {avg_score:.2f}\n"
        report += "\n--- End of Report ---"

        return report

if __name__ == "__main__":
    # 1. Data Ingestion
    data_loader = ConversationDataLoader()
    conversations = data_loader.load_conversations()

    # 2. LLMEVAL Framework Core Initialization
    prompt_generator = EvaluationPromptGenerator()
    llm_interface = EvaluatorLLMInterface()
    evaluation_processor = EvaluationProcessor(prompt_generator, llm_interface)

    # 3. Process Conversations and Generate Scores
    print("Processing conversations...")
    raw_evaluations = evaluation_processor.process_conversations(conversations)
    print(f"Completed processing {len(raw_evaluations)} chatbot responses.")

    # 4. Reporting and Analysis
    reporter = EvaluationReporter()
    performance_report = reporter.generate_report(raw_evaluations)

    print("\n" + performance_report)

    # Optional: Print raw evaluations for inspection
    # print("\nRaw Evaluations:")
    # for eval_item in raw_evaluations:
    #     print(eval_item)
