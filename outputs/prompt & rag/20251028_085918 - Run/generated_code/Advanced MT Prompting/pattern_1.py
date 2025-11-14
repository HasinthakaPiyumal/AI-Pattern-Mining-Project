class DictionaryService:
    def __init__(self):
        self.technical_terms = {
            "API": "Application Programming Interface: A set of defined rules that enable different applications to communicate with each other.",
            "SLA": "Service Level Agreement: A commitment between a service provider and a client. Particular aspects of the service – quality, availability, responsibilities – are agreed between the service provider and the service user.",
            "CRM": "Customer Relationship Management: A technology for managing all your company\'s relationships and interactions with customers and potential customers.",
            "FAQ": "Frequently Asked Questions: A list of common questions and answers on a particular topic."
        }

    def get_definition(self, term):
        return self.technical_terms.get(term.upper(), None)

    def add_term(self, term, definition):
        self.technical_terms[term.upper()] = definition

