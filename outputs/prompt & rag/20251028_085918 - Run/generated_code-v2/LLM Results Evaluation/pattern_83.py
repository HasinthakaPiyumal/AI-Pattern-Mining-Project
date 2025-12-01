
import math

class SmartCustomerSupportSystem:
    def __init__(self, historical_tickets):
        self.historical_tickets = historical_tickets
        self.vocabulary = self._build_vocabulary()
        self.historical_vectors = self._vectorize_historical_tickets()

    def _build_vocabulary(self):
        all_words = set()
        for ticket in self.historical_tickets:
            words = self._tokenize(ticket["query"])
            all_words.update(words)
        return sorted(list(all_words))

    def _tokenize(self, text):
        return text.lower().split()

    def _vectorize_text(self, text):
        word_counts = {}
        for word in self._tokenize(text):
            word_counts[word] = word_counts.get(word, 0) + 1

        vector = [0] * len(self.vocabulary)
        for i, vocab_word in enumerate(self.vocabulary):
            vector[i] = word_counts.get(vocab_word, 0)
        return vector

    def _vectorize_historical_tickets(self):
        vectors = []
        for ticket in self.historical_tickets:
            vectors.append(self._vectorize_text(ticket["query"]))
        return vectors

    def _euclidean_distance(self, vec1, vec2):
        distance = 0
        for i in range(len(vec1)):
            distance += (vec1[i] - vec2[i]) ** 2
        return math.sqrt(distance)

    def suggest_solutions(self, new_query, k=3):
        new_query_vector = self._vectorize_text(new_query)
        distances = []

        for i, historical_vector in enumerate(self.historical_vectors):
            dist = self._euclidean_distance(new_query_vector, historical_vector)
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        nearest_neighbors_indices = [idx for dist, idx in distances[:k]]

        suggested_solutions = []
        for idx in nearest_neighbors_indices:
            suggested_solutions.append({
                "query": self.historical_tickets[idx]["query"],
                "solution": self.historical_tickets[idx]["solution"]
            })
        return suggested_solutions

# Example Usage:
# Define some historical customer support tickets (Knowledge Base)
historical_tickets_data = [
    {"query": "My internet is not working.", "solution": "Troubleshoot router, check cables, restart modem."},
    {"query": "Cannot connect to wifi.", "solution": "Check Wi-Fi password, restart router, forget and reconnect network."},
    {"query": "Slow internet speed.", "solution": "Run speed test, check for background downloads, contact ISP."},
    {"query": "My laptop won't turn on.", "solution": "Check power adapter, battery, try a hard reset."},
    {"query": "Email not sending.", "solution": "Check internet connection, verify SMTP settings, check spam folder."},
    {"query": "Printer offline issue.", "solution": "Ensure printer is on, check cable connections, restart printer and computer."},
    {"query": "Forgot my password.", "solution": "Use password reset link on the login page."},
    {"query": "Computer freezing frequently.", "solution": "Check task manager for high resource usage, update drivers, scan for malware."},
]

# Initialize the customer support system
system = SmartCustomerSupportSystem(historical_tickets_data)

# Simulate an incoming customer query
new_customer_query = "My wifi is not connecting, I tried restarting."

# Get suggested solutions
k_suggestions = system.suggest_solutions(new_customer_query, k=2)

print("New Customer Query:", new_customer_query)
print("\nSuggested Solutions:")
for i, suggestion in enumerate(k_suggestions):
    print(f"  Suggestion {i+1}:")
    print(f"    Similar Query: {suggestion['query']}")
    print(f"    Proposed Solution: {suggestion['solution']}")

new_customer_query_2 = "I can't access internet on my device."
k_suggestions_2 = system.suggest_solutions(new_customer_query_2, k=1)
print("\nNew Customer Query:", new_customer_query_2)
print("\nSuggested Solutions:")
for i, suggestion in enumerate(k_suggestions_2):
    print(f"  Suggestion {i+1}:")
    print(f"    Similar Query: {suggestion['query']}")
    print(f"    Proposed Solution: {suggestion['solution']}")
