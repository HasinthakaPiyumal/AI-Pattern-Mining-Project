class InCLTChatbot:
    def __init__(self):
        self.system_instruction = "You are a helpful multilingual customer support assistant. Provide concise and accurate answers."
        self.in_context_examples = [
            {
                "topic": "account_issue",
                "english_problem": "I can't log in to my account.",
                "english_solution": "Please ensure you are using the correct username and password. If you forgot your password, you can reset it via the 'Forgot Password' link.",
                "spanish_problem": "No puedo iniciar sesión en mi cuenta.",
                "spanish_solution": "Por favor, asegúrese de usar el nombre de usuario y la contraseña correctos. Si olvidó su contraseña, puede restablecerla a través del enlace 'Olvidé mi contraseña'."
            },
            {
                "topic": "billing_query",
                "english_problem": "What are the charges on my last bill?",
                "english_solution": "You can view a detailed breakdown of your charges in the 'Billing History' section of your account portal.",
                "spanish_problem": "¿Cuáles son los cargos en mi última factura?",
                "spanish_solution": "Puede ver un desglose detallado de sus cargos en la sección 'Historial de facturación' de su portal de cuenta."
            },
            {
                "topic": "product_info",
                "english_problem": "How do I use feature X?",
                "english_solution": "Feature X allows you to [explain feature X]. You can find a step-by-step guide in our help center under 'Product Features'.",
                "spanish_problem": "¿Cómo utilizo la función X?",
                "spanish_solution": "La función X le permite [explicar la función X]. Puede encontrar una guía paso a paso en nuestro centro de ayuda en 'Características del producto'."
            }
        ]

    def _select_relevant_examples(self, query, user_language):
        selected_topics = set()
        query_lower = query.lower()

        if "log in" in query_lower or "login" in query_lower or "cuenta" in query_lower or "sesión" in query_lower:
            selected_topics.add("account_issue")
        if "bill" in query_lower or "charges" in query_lower or "factura" in query_lower or "cargos" in query_lower:
            selected_topics.add("billing_query")
        if "feature" in query_lower or "función" in query_lower:
            selected_topics.add("product_info")

        relevant_examples = []
        for example in self.in_context_examples:
            if example["topic"] in selected_topics:
                relevant_examples.append(example)

        if not relevant_examples and self.in_context_examples:
            relevant_examples.append(self.in_context_examples[0])
            
        return relevant_examples

    def _build_prompt(self, user_query, user_language, selected_examples):
        prompt_parts = [self.system_instruction, "\n"]

        prompt_parts.append("Here are some examples of problems and solutions in both English and your language to help you:\n")
        for i, example in enumerate(selected_examples):
            prompt_parts.append(f"--- Example {i+1} ---\n")
            prompt_parts.append(f"English Problem: {example['english_problem']}\n")
            prompt_parts.append(f"English Solution: {example['english_solution']}\n")

            if user_language == "spanish" and "spanish_problem" in example:
                prompt_parts.append(f"Spanish Problem: {example['spanish_problem']}\n")
                prompt_parts.append(f"Spanish Solution: {example['spanish_solution']}\n")

            prompt_parts.append("\n")

        prompt_parts.append(f"Customer Query ({user_language.capitalize()}): {user_query}\n")
        prompt_parts.append("Assistant Response:")

        return "".join(prompt_parts)

    def _simulate_llm_response(self, prompt):
        if "Customer Query (Spanish):" in prompt:
            if "No puedo iniciar sesión" in prompt or "no puedo entrar" in prompt:
                if "Spanish Solution: Por favor, asegúrese de usar el nombre de usuario y la contraseña correctos." in prompt:
                    return "Para iniciar sesión, verifique su nombre de usuario y contraseña. Si los olvidó, puede restablecerlos. (Leveraged cross-lingual example)"
                return "Comprendo que tiene problemas para iniciar sesión. Por favor, asegúrese de que sus credenciales sean correctas."
            
            if "cargos" in prompt or "factura" in prompt:
                if "Spanish Solution: Puede ver un desglose detallado de sus cargos" in prompt:
                    return "Puede ver los detalles de su factura y cargos en la sección 'Historial de facturación' de su cuenta. (Leveraged cross-lingual example)"
                return "Para consultar los cargos de su factura, acceda a su historial de facturación."
            
            if "función X" in prompt:
                if "Spanish Solution: La función X le permite" in prompt:
                    return "La función X está diseñada para [explicar brevemente la función X]. Encuentre la guía completa en nuestro centro de ayuda. (Leveraged cross-lingual example)"
                return "Si desea saber cómo usar la función X, puede consultar nuestra guía en el centro de ayuda."

            return "Entiendo su consulta en español. Estoy procesando la información para proporcionarle una respuesta."

        elif "Customer Query (English):" in prompt:
            if "can't log in" in prompt or "login issue" in prompt:
                if "English Solution: Please ensure you are using the correct username and password." in prompt:
                    return "Please verify your username and password. If you forgot them, you can reset your password. (Leveraged cross-lingual example)"
                return "I understand you're having trouble logging in. Please check your credentials."
            
            if "charges" in prompt or "bill" in prompt:
                if "English Solution: You can view a detailed breakdown of your charges" in prompt:
                    return "You can find a detailed breakdown of your charges in the 'Billing History' section of your account portal. (Leveraged cross-lingual example)"
                return "To see your bill charges, please refer to your billing history."
            
            if "feature X" in prompt:
                if "English Solution: Feature X allows you to" in prompt:
                    return "Feature X is designed to [briefly explain feature X]. A step-by-step guide is available in our help center. (Leveraged cross-lingual example)"
                return "If you want to know how to use Feature X, please check our help center."

            return "I understand your query in English. I am processing the information to provide you with an answer."
        
        return "I am a multilingual assistant, ready to help you with your query, leveraging cross-lingual examples."

    def get_response(self, user_query, user_language="english"):
        if user_language not in ["english", "spanish"]:
            return "Sorry, I currently only support English and Spanish."

        selected_examples = self._select_relevant_examples(user_query, user_language)
        prompt = self._build_prompt(user_query, user_language, selected_examples)
        response = self._simulate_llm_response(prompt)
        return response