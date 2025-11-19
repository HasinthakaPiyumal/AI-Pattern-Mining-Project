import os
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage

class CustomerSupportAgent:
    def __init__(self, openai_api_key=None, knowledge_base_name="customer_support_kb"):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set as an environment variable.")

        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini") # Using gpt-4o-mini for cost-effectiveness
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.classifier = None
        self.train_classifier() # Train classifier on initialization

        self.short_term_memory = collections.deque(maxlen=5) # Stores last 5 turns (query-response pairs)
        self.vectorstore = self._initialize_chroma(knowledge_base_name)

    def _get_embeddings(self, texts):
        return self.embedding_model.encode(texts).tolist()

    def _initialize_chroma(self, collection_name):
        # For simplicity, ChromaDB will be in-memory. For persistence, add persist_directory.
        return Chroma(embedding_function=self._get_embeddings, collection_name=collection_name)

    def _generate_synthetic_training_data(self):
        simple_queries = [
            "What is your return policy?",
            "How do I track my order?",
            "What are your operating hours?",
            "How to reset my password?",
            "What payment methods do you accept?"
        ]
        complex_queries = [
            "My order was damaged during shipping, and I need to file a claim. Can you walk me through the process, including required documentation and timelines?",
            "I'm experiencing an intermittent error with the 'checkout' feature, specifically when applying a discount code that has previously worked. It sometimes processes, sometimes gives an error. What debugging steps can I take or what logs should I provide?",
            "Can you provide a detailed comparison between product X and product Y, focusing on their technical specifications, ideal use cases, and compatibility with third-party accessories?",
            "I need assistance integrating your API with our existing CRM system. We're encountering authentication issues despite following the documentation. What are common pitfalls, and can I get direct technical support for this?",
            "Explain the legal terms and conditions regarding data privacy for European customers under GDPR when using your services, particularly concerning data storage, transfer, and user rights to access/delete their data."
        ]
        X = simple_queries + complex_queries
        y = ["simple"] * len(simple_queries) + ["complex"] * len(complex_queries)
        return X, y

    def train_classifier(self):
        X_raw, y = self._generate_synthetic_training_data()
        X_embeddings = self._get_embeddings(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_train, y_train)

        # Optional: Print accuracy
        y_pred = self.classifier.predict(X_test)
        # print(f"Classifier accuracy: {accuracy_score(y_test, y_pred):.2f}")

    def classify_query(self, query):
        query_embedding = self._get_embeddings([query])
        prediction = self.classifier.predict(query_embedding)[0]
        return prediction

    def add_knowledge_article(self, article_content, metadata=None):
        self.vectorstore.add_documents([Document(page_content=article_content, metadata=metadata or {})])

    def retrieve_knowledge(self, query, k=3, complexity_level="simple"):
        if complexity_level == "complex":
            k = 5 # Retrieve more for complex queries
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def add_to_short_term_memory(self, query, response):
        self.short_term_memory.append((query, response))

    def get_short_term_memory(self):
        history = []
        for q, a in self.short_term_memory:
            history.append(HumanMessage(content=q))
            history.append(AIMessage(content=a))
        return history

    def hotswap_knowledge_base(self, new_knowledge_base_name):
        self.vectorstore = self._initialize_chroma(new_knowledge_base_name)
        print(f"Knowledge base hot-swapped to: {new_knowledge_base_name}")

    def _build_prompt(self, query, short_term_memory_history, relevant_knowledge, complexity_level):
        system_template = (
            "You are an AI customer support agent. "
            "Answer the user's question based on the provided context and conversation history. "
            "If you cannot find the answer, politely state that you don't know. "
            "Your response should be helpful, concise, and professional."
        )

        if complexity_level == "complex":
            system_template += (
                " For complex queries, provide detailed and comprehensive explanations, "
                "and ask clarifying questions if necessary to fully understand the user's need."
            )
        else:
            system_template += " For simple queries, provide direct and clear answers."

        context_template = "\n\nRelevant Knowledge:\n{relevant_knowledge}"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template + context_template),
            *short_term_memory_history,
            ("human", "{query}")
        ])

        return prompt_template.format_messages(
            relevant_knowledge=relevant_knowledge,
            query=query
        )

    def process_query(self, query):
        # 1. Classify query complexity
        complexity_level = self.classify_query(query)

        # 2. Retrieve memory
        short_term_memory_history = self.get_short_term_memory()
        relevant_knowledge = self.retrieve_knowledge(query, complexity_level=complexity_level)

        # 3. Build prompt
        prompt = self._build_prompt(query, short_term_memory_history, relevant_knowledge, complexity_level)

        # 4. Send to LLM and get response
        response = self.llm.invoke(prompt).content

        # 5. Add to short-term memory
        self.add_to_short_term_memory(query, response)

        return response

if __name__ == "__main__":
    # Example Usage:
    # Set your OpenAI API key as an environment variable or pass it directly
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    try:
        agent = CustomerSupportAgent()
    except ValueError as e:
        print(e)
        print("Please set the OPENAI_API_KEY environment variable or pass it to the CustomerSupportAgent constructor.")
        exit()

    # Add some initial knowledge articles
    agent.add_knowledge_article("Our return policy allows returns within 30 days of purchase with a valid receipt.")
    agent.add_knowledge_article("Orders can be tracked using the tracking number provided in your shipping confirmation email.")
    agent.add_knowledge_article("Our customer support hours are Monday to Friday, 9 AM to 5 PM EST.")
    agent.add_knowledge_article("To reset your password, visit the login page and click 'Forgot Password'.")
    agent.add_knowledge_article("We accept Visa, MasterCard, American Express, and PayPal.")
    agent.add_knowledge_article("Product X features a 15-inch display, 16GB RAM, and a 512GB SSD. It's ideal for graphic design. Product Y has a 13-inch display, 8GB RAM, and 256GB SSD, suited for everyday use.")
    agent.add_knowledge_article("Our API uses OAuth2 for authentication. Common issues include incorrect redirect URIs or missing scopes. Refer to our API documentation for details.")

    print("\n--- Starting Customer Support Chat ---")

    queries = [
        "What is your return policy?",
        "How do I track my order?",
        "I forgot my password, what should I do?",
        "My order #12345 was damaged during shipping last week. I need to file a claim; can you guide me through the required documentation and the exact steps to initiate the return and refund process, also specifying the timeline for resolution?"
    ]

    for i, user_query in enumerate(queries):
        print(f"\nUser: {user_query}")
        response = agent.process_query(user_query)
        print(f"Agent: {response}")
        if i == 0:
            # Demonstrate a follow-up query to check short-term memory
            print("\nUser (Follow-up): And how many days do I have?")
            follow_up_response = agent.process_query("And how many days do I have?")
            print(f"Agent: {follow_up_response}")

    print("\n--- Demonstrating Index Hotswapping ---")
    agent.hotswap_knowledge_base("new_product_faq_kb")
    agent.add_knowledge_article("New product Z is launching next month with advanced AI features.")
    print("\nUser: Tell me about new product Z.")
    response_new_kb = agent.process_query("Tell me about new product Z.")
    print(f"Agent: {response_new_kb}")

    print("\n--- End of Chat ---")
