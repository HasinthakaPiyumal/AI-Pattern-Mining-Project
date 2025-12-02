from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

def setup_medical_knowledge_graph():
    kg = Graph()

    # Define namespaces
    EX = Namespace("http://example.org/medical#")
    DRUGBANK = Namespace("http://example.org/drugbank#")
    DISEASE = Namespace("http://example.org/disease#")
    SYMPTOM = Namespace("http://example.org/symptom#")
    TREATS = EX.treats
    HAS_SYMPTOM = EX.hasSymptom
    CAUSES = EX.causes

    # Add some sample medical data
    # Diseases
    diabetes = DISEASE.Diabetes
    hypertension = DISEASE.Hypertension
    flu = DISEASE.Flu
    migraine = DISEASE.Migraine

    kg.add((diabetes, RDF.type, EX.Disease))
    kg.add((diabetes, RDFS.label, Literal("Diabetes")))
    kg.add((hypertension, RDF.type, EX.Disease))
    kg.add((hypertension, RDFS.label, Literal("Hypertension")))
    kg.add((flu, RDF.type, EX.Disease))
    kg.add((flu, RDFS.label, Literal("Influenza")))
    kg.add((migraine, RDF.type, EX.Disease))
    kg.add((migraine, RDFS.label, Literal("Migraine")))

    # Drugs
    metformin = DRUGBANK.Metformin
    lisinopril = DRUGBANK.Lisinopril
    oseltamivir = DRUGBANK.Oseltamivir
    ibuprofen = DRUGBANK.Ibuprofen

    kg.add((metformin, RDF.type, EX.Drug))
    kg.add((metformin, RDFS.label, Literal("Metformin")))
    kg.add((lisinopril, RDF.type, EX.Drug))
    kg.add((lisinopril, RDFS.label, Literal("Lisinopril")))
    kg.add((oseltamivir, RDF.type, EX.Drug))
    kg.add((oseltamivir, RDFS.label, Literal("Oseltamivir")))
    kg.add((ibuprofen, RDF.type, EX.Drug))
    kg.add((ibuprofen, RDFS.label, Literal("Ibuprofen")))

    # Symptoms
    high_blood_sugar = SYMPTOM.HighBloodSugar
    high_blood_pressure = SYMPTOM.HighBloodPressure
    fever = SYMPTOM.Fever
    headache = SYMPTOM.Headache

    kg.add((high_blood_sugar, RDF.type, EX.Symptom))
    kg.add((high_blood_sugar, RDFS.label, Literal("High Blood Sugar")))
    kg.add((high_blood_pressure, RDF.type, EX.Symptom))
    kg.add((high_blood_pressure, RDFS.label, Literal("High Blood Pressure")))
    kg.add((fever, RDF.type, EX.Symptom))
    kg.add((fever, RDFS.label, Literal("Fever")))
    kg.add((headache, RDF.type, EX.Symptom))
    kg.add((headache, RDFS.label, Literal("Headache")))

    # Relationships
    kg.add((metformin, TREATS, diabetes))
    kg.add((lisinopril, TREATS, hypertension))
    kg.add((oseltamivir, TREATS, flu))
    kg.add((ibuprofen, TREATS, migraine))
    kg.add((ibuprofen, TREATS, fever)) # Ibuprofen can also treat fever related to flu or other conditions

    kg.add((diabetes, HAS_SYMPTOM, high_blood_sugar))
    kg.add((hypertension, HAS_SYMPTOM, high_blood_pressure))
    kg.add((flu, HAS_SYMPTOM, fever))
    kg.add((flu, HAS_SYMPTOM, headache))
    kg.add((migraine, HAS_SYMPTOM, headache))

    return kg
