class MedicalDictionary:
    def __init__(self):
        # A mock medical dictionary for demonstration purposes.
        # In a real application, this would connect to a comprehensive medical terminology database
        # (e.g., UMLS, SNOMED CT, ICD) or an external API.
        self.dictionary = {
            "en": {
                "hypertension": "A medical condition in which the blood pressure in the arteries is persistently elevated.",
                "diabetes": "A metabolic disease that causes high blood sugar levels.",
                "myocardial infarction": "Commonly known as a heart attack, it occurs when blood flow to a part of the heart is blocked.",
                "diagnosis": "The identification of the nature of an illness or other problem by examination of the symptoms."
            },
            "es": {
                "hypertension": "Una condición médica en la que la presión arterial en las arterias está persistentemente elevada.",
                "diabetes": "Una enfermedad metabólica que causa niveles altos de azúcar en la sangre.",
                "myocardial infarction": "Comúnmente conocido como ataque al corazón, ocurre cuando el flujo sanguíneo a una parte del corazón se bloquea.",
                "diagnosis": "La identificación de la naturaleza de una enfermedad u otro problema mediante el examen de los síntomas."
            },
            "fr": {
                "hypertension": "Une condition médicale dans laquelle la pression artérielle dans les artères est constamment élevée.",
                "diabetes": "Une maladie métabolique qui provoque des niveaux élevés de sucre dans le sang.",
                "myocardial infarction": "Communément appelé crise cardiaque, il se produit lorsque le flux sanguin vers une partie du cœur est bloqué.",
                "diagnosis": "L'identification de la nature d'une maladie ou d'un autre problème par l'examen des symptômes."
            }
        }

    def extract_medical_terms(self, text: str, source_lang: str = "en") -> list[str]:
        """Simulates extracting key medical terms from the input text.
        In a real scenario, this would involve NLP techniques (e.g., spaCy with a medical model).
        """
        # For simplicity, we'll just check for a few hardcoded terms in the text.
        # A more robust solution would use a tokenizer and NER.
        found_terms = []
        lower_text = text.lower()
        for term in self.dictionary[source_lang].keys():
            if term in lower_text:
                found_terms.append(term)
        return list(set(found_terms)) # Return unique terms

    def get_definitions(self, terms: list[str], source_lang: str, target_lang: str) -> dict:
        """Retrieves definitions for given terms in specified source and target languages.
        """
        definitions = {
            "source": {},
            "target": {}
        }
        for term in terms:
            definitions["source"][term] = self.dictionary.get(source_lang, {}).get(term, "No definition found.")
            definitions["target"][term] = self.dictionary.get(target_lang, {}).get(term, "No definition found.")
        return definitions