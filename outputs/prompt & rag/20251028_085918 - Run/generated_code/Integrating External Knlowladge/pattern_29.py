import time
import random

def fetch_pubmed_articles(query: str, num_results: int = 3) -> list[dict]:
    """Simulates fetching medical articles from PubMed based on a query.
    Returns a list of dictionaries, each representing an article.
    """
    print(f"Simulating PubMed search for: '{query}'")
    # Dummy data for demonstration
    articles = [
        {
            "id": "pmid101",
            "title": f"Recent Advances in {query.title()} Treatment",
            "abstract": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam."
        },
        {
            "id": "pmid102",
            "title": f"Diagnostic Markers for {query.title()} Early Detection",
            "abstract": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident."
        },
        {
            "id": "pmid103",
            "title": f"Epidemiology of {query.title()} in Pediatric Population",
            "abstract": "Sunt in culpa qui officia deserunt mollit anim id est laborum. Ut aliquip ex ea commodo consequat. Sed ut perspiciatis unde omnis iste natus error sit voluptatem."
        },
    ]
    time.sleep(0.5) # Simulate API latency
    return random.sample(articles, min(num_results, len(articles)))

def get_ehr_data(patient_id: str) -> dict:
    """Simulates fetching Electronic Health Record (EHR) data for a given patient ID.
    Returns a dictionary with patient information.
    """
    print(f"Simulating EHR retrieval for patient ID: {patient_id}")
    # Dummy EHR data
    ehr_data = {
        "P001": {
            "name": "Alice Smith",
            "age": 45,
            "gender": "Female",
            "conditions": ["Hypertension", "Type 2 Diabetes"],
            "medications": ["Lisinopril", "Metformin"],
            "allergies": ["Penicillin"],
            "lab_results": {
                "glucose": "180 mg/dL",
                "blood_pressure": "140/90 mmHg"
            }
        },
        "P002": {
            "name": "Bob Johnson",
            "age": 60,
            "gender": "Male",
            "conditions": ["Coronary Artery Disease"],
            "medications": ["Aspirin", "Atorvastatin"],
            "allergies": [],
            "lab_results": {
                "cholesterol": "220 mg/dL",
                "blood_pressure": "130/85 mmHg"
            }
        }
    }
    time.sleep(0.3) # Simulate API latency
    return ehr_data.get(patient_id, {"error": "Patient not found"})

def check_drug_interactions(drug_list: list[str]) -> dict:
    """Simulates checking for drug-drug interactions.
    Returns a dictionary of potential interactions.
    """
    print(f"Simulating drug interaction check for: {', '.join(drug_list)}")
    interactions = {}
    if "Lisinopril" in drug_list and "Ibuprofen" in drug_list:
        interactions["Lisinopril-Ibuprofen"] = "Potential for reduced antihypertensive effect and renal impairment."
    if "Metformin" in drug_list and "Contrast Dye" in drug_list:
        interactions["Metformin-Contrast Dye"] = "Risk of lactic acidosis. Metformin should be withheld."
    time.sleep(0.4) # Simulate API latency
    return interactions

def controlled_web_search(query: str, trusted_sites: list[str] = None) -> list[dict]:
    """Simulates a controlled web search on trusted medical websites.
    In a real application, this would involve a robust browsing agent.
    """
    if trusted_sites is None:
        trusted_sites = ["who.int", "cdc.gov", "nejm.org"]
    
    print(f"Simulating controlled web search for '{query}' on sites like: {', '.join(trusted_sites)}")
    
    # Dummy results based on query
    results = [
        {
            "url": f"https://www.who.int/news/{query.lower().replace(' ', '-')}-overview",
            "title": f"WHO - Overview of {query.title()}",
            "snippet": "The World Health Organization provides comprehensive information on symptoms, diagnosis, and global impact of this condition."
        },
        {
            "url": f"https://www.cdc.gov/diseases/{query.lower().replace(' ', '-')}/index.html",
            "title": f"CDC - Information on {query.title()}",
            "snippet": "The Centers for Disease Control and Prevention offers guidelines for prevention, treatment, and public health advisories."
        }
    ]
    time.sleep(0.7)
    return results