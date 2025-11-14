"""
Simulated Medical Knowledge Base
"""

medical_documents = {
    "doc_1": {
        "title": "Diagnosis and Treatment of Hypertension",
        "content": "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Diagnosis involves regular blood pressure readings. Treatment often includes lifestyle changes (diet, exercise) and medications (diuretics, ACE inhibitors, beta-blockers)."
    },
    "doc_2": {
        "title": "Managing Type 2 Diabetes Mellitus",
        "content": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists insulin. Management involves diet control, regular physical activity, and often medication, including oral agents or insulin injections. Regular monitoring of blood glucose is crucial."
    },
    "doc_3": {
        "title": "Symptoms and Management of Migraine Headaches",
        "content": "Migraine is a severe type of headache characterized by throbbing pain on one side of the head, sensitivity to light and sound, and sometimes nausea or vomiting. Triggers can include stress, certain foods, and hormonal changes. Treatment involves pain relievers, triptans, and preventative medications. Lifestyle adjustments can also help."
    },
    "doc_4": {
        "title": "Understanding Coronary Artery Disease",
        "content": "Coronary artery disease (CAD) is caused by plaque buildup in the walls of the arteries that supply blood to the heart. This narrows the arteries, reducing blood flow to the heart muscle. Symptoms include chest pain (angina), shortness of breath, and fatigue. Treatment options range from lifestyle changes and medications (statins, aspirin) to procedures like angioplasty or bypass surgery."
    },
    "doc_5": {
        "title": "Pediatric Asthma Guidelines",
        "content": "Asthma in children is a chronic inflammatory disease of the airways that causes recurrent episodes of wheezing, breathlessness, chest tightness, and coughing. Triggers can include allergens, respiratory infections, and exercise. Management involves long-term control medications (inhaled corticosteroids) and quick-relief medications (bronchodilators)."
    }
}

def get_document_by_id(doc_id):
    """Retrieves a document by its ID."""
    return medical_documents.get(doc_id, None)

def get_all_documents():
    """Returns all documents in the knowledge base."""
    return medical_documents
