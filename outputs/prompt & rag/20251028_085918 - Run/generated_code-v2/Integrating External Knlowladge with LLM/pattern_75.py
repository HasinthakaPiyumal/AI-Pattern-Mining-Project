
import networkx as nx

def build_medical_knowledge_graph():
    """
    Builds a simplified medical knowledge graph using NetworkX.
    In a real application, this would be populated from large datasets
    and potentially stored in a graph database like Neo4j.
    """
    kg = nx.DiGraph()

    # Entities: Diseases, Symptoms, Treatments, Drugs
    kg.add_nodes_from([
        ("Fever", {"type": "Symptom"}),
        ("Cough", {"type": "Symptom"}),
        ("Headache", {"type": "Symptom"}),
        ("Influenza", {"type": "Disease"}),
        ("Common Cold", {"type": "Disease"}),
        ("Pneumonia", {"type": "Disease"}),
        ("Paracetamol", {"type": "Drug"}),
        ("Antibiotics", {"type": "Drug"}),
        ("Rest", {"type": "Treatment"}),
        ("Flu Shot", {"type": "Treatment"}),
        ("Sore Throat", {"type": "Symptom"})
    ])

    # Relationships
    kg.add_edges_from([
        ("Fever", "indicates", "Influenza"),
        ("Cough", "indicates", "Influenza"),
        ("Headache", "indicates", "Influenza"),
        ("Fever", "indicates", "Common Cold"),
        ("Cough", "indicates", "Common Cold"),
        ("Sore Throat", "indicates", "Common Cold"),
        ("Fever", "indicates", "Pneumonia"),
        ("Cough", "indicates", "Pneumonia"),
        ("Influenza", "treated_by", "Paracetamol"),
        ("Influenza", "treated_by", "Rest"),
        ("Influenza", "prevented_by", "Flu Shot"),
        ("Common Cold", "treated_by", "Paracetamol"),
        ("Common Cold", "treated_by", "Rest"),
        ("Pneumonia", "treated_by", "Antibiotics"),
        ("Paracetamol", "alleviates", "Fever"),
        ("Paracetamol", "alleviates", "Headache")
    ])
    return kg

if __name__ == "__main__":
    kg = build_medical_knowledge_graph()
    print(f"Nodes in KG: {kg.number_of_nodes()}")
    print(f"Edges in KG: {kg.number_of_edges()}")
    print("Example path (Fever -> Influenza -> Paracetamol):")
    try:
        path = nx.shortest_path(kg, source="Fever", target="Paracetamol")
        print(path)
    except nx.NetworkXNoPath:
        print("No path found.")

