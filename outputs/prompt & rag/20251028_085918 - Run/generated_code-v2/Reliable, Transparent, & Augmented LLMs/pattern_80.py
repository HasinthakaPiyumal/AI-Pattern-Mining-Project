
import json
import random

def prepare_finetuning_dataset(output_path="finetuning_dataset.jsonl", num_examples=100, abstention_rate=0.2):
    """
    Prepares a synthetic finetuning dataset with a subset of answers replaced by 'I don't know'.

    Args:
        output_path (str): The path to save the dataset.
        num_examples (int): Total number of examples to generate.
        abstention_rate (float): The proportion of examples to replace with 'I don't know'.
    """
    dataset = []
    financial_questions = [
        "What is the interest rate on a savings account?",
        "How do I transfer money between my accounts?",
        "What are the requirements to open a checking account?",
        "Explain what a mortgage is.",
        "What is the difference between a credit card and a debit card?",
        "How can I report a lost or stolen card?",
        "What are the fees associated with an overdraft?",
        "Can I get a loan for a small business?",
        "What is the current exchange rate for USD to EUR?",
        "How do I set up direct deposit?"
    ]
    financial_answers = [
        "The interest rate on a standard savings account is currently 0.05% APY, but can vary based on account type.",
        "You can transfer money between your accounts through our online banking portal or mobile app by navigating to the 'Transfers' section.",
        "To open a checking account, you typically need a valid ID, proof of address, and an initial deposit. Specific requirements may vary.",
        "A mortgage is a loan used to purchase or maintain a home, land, or other types of real estate. The borrower agrees to pay the lender over time, typically in a series of regular payments.",
        "A credit card allows you to borrow money up to a certain limit and pay it back later, while a debit card uses funds directly from your bank account.",
        "To report a lost or stolen card, please call our 24/7 support line immediately at 1-800-555-0123.",
        "Overdraft fees can vary, but generally range from $25 to $35 per occurrence. Please refer to your account agreement for details.",
        "Yes, we offer various small business loan options. Please contact our business banking specialists for more information.",
        "The current exchange rate for USD to EUR fluctuates. For the latest rate, please check our currency converter tool online.",
        "You can set up direct deposit by providing your employer with your bank's routing number and your account number. These can be found on your checks or online banking portal."
    ]

    # Generate base dataset
    for i in range(num_examples):
        question_idx = random.randint(0, len(financial_questions) - 1)
        question = financial_questions[question_idx]
        answer = financial_answers[question_idx]
        dataset.append({"instruction": question, "response": answer})

    # Introduce abstention examples
    num_abstain_examples = int(num_examples * abstention_rate)
    abstain_indices = random.sample(range(num_examples), num_abstain_examples)

    for idx in abstain_indices:
        dataset[idx]["response"] = "I don't know or I cannot provide an answer to that, please contact a human agent."

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"Dataset prepared and saved to {output_path} with {num_abstain_examples} abstention examples.")

if __name__ == "__main__":
    prepare_finetuning_dataset()
