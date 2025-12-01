
import json
import random

def generate_synthetic_medical_data(num_samples=1000):
    symptoms_conditions = [
        ("fever, cough, sore throat", "Common Cold"),
        ("severe headache, stiff neck, fever", "Meningitis"),
        ("chest pain, shortness of breath, left arm pain", "Heart Attack"),
        ("sudden weakness on one side of body, difficulty speaking", "Stroke"),
        ("abdominal pain, nausea, vomiting, yellow skin", "Hepatitis"),
        ("fatigue, increased thirst, frequent urination", "Diabetes"),
        ("joint pain, swelling, stiffness in morning", "Arthritis"),
        ("rash, itching, hives after eating seafood", "Allergic Reaction"),
        ("persistent cough, weight loss, night sweats", "Tuberculosis"),
        ("muscle weakness, difficulty swallowing, drooping eyelids", "Myasthenia Gravis"),
        ("burning sensation during urination, frequent urge to urinate", "UTI"),
        ("severe back pain, radiating down leg", "Sciatica"),
        ("dizziness, spinning sensation, nausea", "Vertigo"),
        ("blurred vision, sensitivity to light, eye pain", "Glaucoma"),
        ("constant hunger, anxiety, rapid heart beat, weight loss", "Hyperthyroidism"),
        ("difficulty sleeping, irritability, lack of energy", "Insomnia"),
        ("headache, sensitivity to light and sound, nausea", "Migraine"),
        ("bloody stools, abdominal cramps, diarrhea", "Crohn's Disease"),
        ("unexplained weight gain, fatigue, cold sensitivity", "Hypothyroidism"),
        ("difficulty breathing, wheezing, chest tightness", "Asthma"),
    ]

    data = []
    for i in range(num_samples):
        # 80% confident answers
        if random.random() < 0.8:
            symptom, condition = random.choice(symptoms_conditions)
            data.append({"instruction": f"What conditions are associated with {symptom}?", "output": condition})
        # 20% 'I don't know' answers for abstention training
        else:
            ambiguous_symptom_starters = [
                "general discomfort", "feeling unwell", "some pain",
                "unusual sensation", "just not feeling right", "mild ache",
                "intermittent symptoms", "random tiredness", "slight headache",
                "vague symptoms", "non-specific pain", "just a little sick",
                "feeling off", "minor irritation", "occasional discomfort"
            ]
            symptom_description = random.choice(ambiguous_symptom_starters) + ", but nothing specific."
            data.append({"instruction": f"What conditions are associated with {symptom_description}?", "output": "I don't have enough information to suggest a potential condition. Please consult a healthcare professional for a proper diagnosis."})

    return data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_medical_data(num_samples=1000)
    with open("medical_finetuning_data.json", "w") as f:
        json.dump(synthetic_data, f, indent=4)
    print(f"Generated {len(synthetic_data)} samples and saved to medical_finetuning_data.json")

    # Generate a simple knowledge base for RAG
    knowledge_base = [
        "Common Cold: Viral infection of the nose and throat. Symptoms include runny nose, sore throat, cough, congestion, and sometimes body aches.",
        "Meningitis: Inflammation of the membranes surrounding the brain and spinal cord. Symptoms include sudden fever, severe headache, and stiff neck.",
        "Heart Attack: Occurs when blood flow to a part of the heart is blocked. Symptoms include chest pain, shortness of breath, pain in the left arm, and lightheadedness.",
        "Stroke: Occurs when blood supply to part of your brain is interrupted or reduced. Symptoms include sudden weakness or numbness on one side of the body, confusion, trouble speaking or understanding speech.",
        "Hepatitis: Inflammation of the liver. Symptoms can include fatigue, nausea, vomiting, abdominal pain, loss of appetite, and jaundice (yellowing of skin or eyes)."]
    with open("medical_knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f, indent=4)
    print(f"Generated a simple medical_knowledge_base.json")
