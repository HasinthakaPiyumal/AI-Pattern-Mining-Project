import requests
from bs4 import BeautifulSoup
from collections import deque
import time
import random

class MockLLM:
    def __init__(self):
        self.searched_topics = {}
        self.visited_urls = set()
        self.quotes = []

    def decide_action(self, current_state):
        question = current_state["question"]
        page_text = current_state.get("page_text", "")
        current_url = current_state.get("current_url", None)
        page_links = current_state.get("page_links", [])
        history = current_state["history"]

        if not history or history[-1]["command"] == "SEARCH":
            if question not in self.searched_topics:
                self.searched_topics[question] = True
                return {"command": "SEARCH", "argument": question}
            
        if page_text:
            # Simulate quoting relevant sections
            relevant_sentences = [s.strip() for s in page_text.split('.') if len(s.strip()) > 50 and any(keyword in s.lower() for keyword in question.lower().split()[:3])]
            if relevant_sentences and len(self.quotes) < 3:
                for sentence in random.sample(relevant_sentences, min(len(relevant_sentences), 1)):
                    if sentence not in [q['text'] for q in self.quotes]:
                        self.quotes.append({"text": sentence, "url": current_url})
                        return {"command": "QUOTE", "argument": sentence}

            # Simulate clicking a link
            if page_links:
                unvisited_links = [link for link in page_links if link not in self.visited_urls]
                if unvisited_links and len(self.visited_urls) < 5:
                    chosen_link = random.choice(unvisited_links)
                    self.visited_urls.add(chosen_link)
                    return {"command": "CLICK", "argument": chosen_link}
            
            # Simulate scrolling
            if len(page_text) > 1000 and current_state.get("scroll_position", 0) < 3:
                 return {"command": "SCROLL", "argument": "down"}

        return {"command": "END", "argument": None}

    def synthesize_answer(self, question, collected_quotes):
        if not collected_quotes:
            return f"MediSearch AI could not find specific information for: '{question}'."
        
        answer_parts = [f"Based on information gathered, here's a summary regarding '{question}':\n"]
        for i, quote_data in enumerate(collected_quotes):
            answer_parts.append(f"  - Quote {i+1}: \"{quote_data['text']}\" (Source: {quote_data['url']})")
        
        answer_parts.append("\nThis information is synthesized from various web sources.")
        return "\n".join(answer_parts)

class WebPageRetriever:
    def fetch_and_parse(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except requests.exceptions.RequestException as e:
            return None

    def get_visible_text(self, soup, scroll_position=0, chunk_size=2000):
        if not soup:
            return ""
        
        paragraphs = soup.find_all('p')
        full_text = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        
        start_index = min(scroll_position * chunk_size, len(full_text))
        end_index = min(start_index + chunk_size, len(full_text))
        
        return full_text[start_index:end_index]

    def get_links(self, soup, base_url):
        if not soup:
            return []
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('http') or href.startswith('https'):
                links.add(href)
            elif href.startswith('/') and base_url:
                links.add(requests.compat.urljoin(base_url, href))
        return list(links)

class BrowserEnvironment:
    def __init__(self):
        self.retriever = WebPageRetriever()
        self.current_url = None
        self.scroll_position = 0
        self.page_content = ""
        self.page_links = []

    def search_web(self, query):
        print(f"Performing web search for: {query}")
        # Mock search results for demonstration
        mock_results = {
            "diabetes": [
                "https://www.who.int/news-room/fact-sheets/detail/diabetes",
                "https://www.cdc.gov/diabetes/basics/index.html",
                "https://www.niddk.nih.gov/health-information/diabetes"
            ],
            "hypertension treatment": [
                "https://www.heart.org/en/health-topics/high-blood-pressure/treatment-of-high-blood-pressure",
                "https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/diagnosis-treatment/drc-20373417",
                "https://www.acc.org/latest-in-cardiology/articles/2021/03/17/14/08/hypertension-guidelines"
            ]
        }
        return mock_results.get(query.lower(), [])

    def navigate_to_url(self, url):
        print(f"Navigating to: {url}")
        self.current_url = url
        self.scroll_position = 0
        soup = self.retriever.fetch_and_parse(url)
        self.page_content = self.retriever.get_visible_text(soup, self.scroll_position)
        self.page_links = self.retriever.get_links(soup, url)
        return self.page_content, self.page_links

    def scroll_page(self, direction="down"):
        if direction == "down":
            self.scroll_position += 1
        elif direction == "up" and self.scroll_position > 0:
            self.scroll_position -= 1
        
        soup = self.retriever.fetch_and_parse(self.current_url) # Re-fetch to ensure consistent state for scrolling
        self.page_content = self.retriever.get_visible_text(soup, self.scroll_position)
        print(f"Scrolled {direction}. Current scroll position: {self.scroll_position}")
        return self.page_content

class MediSearchAI:
    def __init__(self):
        self.agent = MockLLM()
        self.browser = BrowserEnvironment()
        self.history = deque(maxlen=10) # Keep a history of actions
        self.collected_quotes = []

    def run_query(self, question):
        print(f"MediSearch AI initiated for query: '{question}'")
        self.history.clear()
        self.collected_quotes.clear()
        self.agent.visited_urls.clear() # Reset visited URLs for new query
        self.agent.quotes.clear() # Reset agent's internal quotes

        while True:
            current_state = {
                "question": question,
                "page_text": self.browser.page_content,
                "current_url": self.browser.current_url,
                "page_links": self.browser.page_links,
                "scroll_position": self.browser.scroll_position,
                "history": list(self.history)
            }
            
            action = self.agent.decide_action(current_state)
            command = action["command"]
            argument = action["argument"]

            self.history.append({"command": command, "argument": argument, "timestamp": time.time()})

            if command == "SEARCH":
                search_results = self.browser.search_web(argument)
                if search_results:
                    # Automatically click the first result for simplicity in this mock
                    self.browser.navigate_to_url(search_results[0])
                else:
                    print("No search results found.")
                    break
            elif command == "CLICK":
                self.browser.navigate_to_url(argument)
            elif command == "SCROLL":
                self.browser.scroll_page(argument)
            elif command == "QUOTE":
                print(f"Quoting: \"{argument}\"")
                self.collected_quotes.append({"text": argument, "url": self.browser.current_url})
                self.agent.quotes.append({"text": argument, "url": self.browser.current_url}) # Keep agent's internal quotes synced
            elif command == "END":
                print("Browsing session ended by LLM agent.")
                break
            
            # Prevent infinite loops in mock agent for demonstration
            if len(self.history) > 15: 
                print("Max browsing steps reached, ending session.")
                break

            time.sleep(0.5) # Simulate some processing time

        final_answer = self.agent.synthesize_answer(question, self.collected_quotes)
        return final_answer

if __name__ == "__main__":
    medisearch = MediSearchAI()
    
    query1 = "What are the latest treatments for diabetes?"
    answer1 = medisearch.run_query(query1)
    print("\n" + "="*50 + "\n")
    print(f"Final Answer for '{query1}':\n{answer1}")
    print("\n" + "="*50 + "\n")

    query2 = "Symptoms and prevention of hypertension."
    answer2 = medisearch.run_query(query2)
    print("\n" + "="*50 + "\n")
    print(f"Final Answer for '{query2}':\n{answer2}")
    print("\n" + "="*50 + "\n")

    query3 = "What is Lorem Ipsum?" # Query with no mock results
    answer3 = medisearch.run_query(query3)
    print("\n" + "="*50 + "\n")
    print(f"Final Answer for '{query3}':\n{answer3}")
    print("\n" + "="*50 + "\n")
