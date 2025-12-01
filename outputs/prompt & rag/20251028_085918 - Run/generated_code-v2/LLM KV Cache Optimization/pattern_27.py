
import collections
import random

PAGE_SIZE = 4  # Represents the number of KV pairs a page can hold
TOTAL_PAGES = 16 # Total available pages in our simulated GPU memory

class KVCachePageManager:
    def __init__(self, total_pages=TOTAL_PAGES, page_size=PAGE_SIZE):
        self.total_pages = total_pages
        self.page_size = page_size
        # Simulate physical memory as a list of pages
        self.physical_pages = [[] for _ in range(total_pages)]
        self.free_pages = set(range(total_pages))
        # Maps sequence_id to a list of (logical_block_index, physical_page_id) tuples
        self.sequence_page_map = collections.defaultdict(list)
        self.next_sequence_id = 0

    def _get_next_sequence_id(self):
        seq_id = self.next_sequence_id
        self.next_sequence_id += 1
        return seq_id

    def allocate_pages(self, sequence_id, num_pages_needed):
        if len(self.free_pages) < num_pages_needed:
            raise MemoryError(f"Not enough free pages to allocate {num_pages_needed} pages for sequence {sequence_id}")

        allocated_physical_page_ids = []
        for _ in range(num_pages_needed):
            physical_page_id = self.free_pages.pop()
            allocated_physical_page_ids.append(physical_page_id)
        
        # Map new pages to the end of the sequence's existing logical blocks
        current_logical_block_count = len(self.sequence_page_map[sequence_id])
        for i, phys_id in enumerate(allocated_physical_page_ids):
            self.sequence_page_map[sequence_id].append((current_logical_block_count + i, phys_id))

        return allocated_physical_page_ids

    def free_pages(self, sequence_id):
        if sequence_id in self.sequence_page_map:
            for _, physical_page_id in self.sequence_page_map[sequence_id]:
                self.free_pages.add(physical_page_id)
                self.physical_pages[physical_page_id] = [] # Clear page content
            del self.sequence_page_map[sequence_id]
            return True
        return False

    def get_physical_page_id(self, sequence_id, logical_block_index):
        for log_idx, phys_id in self.sequence_page_map[sequence_id]:
            if log_idx == logical_block_index:
                return phys_id
        return None # Page not found for this logical block

    def write_kv_to_page(self, physical_page_id, kv_data_list):
        if physical_page_id < 0 or physical_page_id >= self.total_pages:
            raise ValueError("Invalid physical page ID")
        # In a real scenario, this would write actual KV tensors.
        # Here, we just simulate storing data up to page_size.
        self.physical_pages[physical_page_id].extend(kv_data_list[:self.page_size - len(self.physical_pages[physical_page_id])])

    def read_kv_from_page(self, physical_page_id):
        if physical_page_id < 0 or physical_page_id >= self.total_pages:
            raise ValueError("Invalid physical page ID")
        return self.physical_pages[physical_page_id]

    def get_memory_usage(self):
        allocated_pages = self.total_pages - len(self.free_pages)
        return {"total_pages": self.total_pages, "allocated_pages": allocated_pages, "free_pages": len(self.free_pages)}

class LLMSimulator:
    def __init__(self, page_manager):
        self.page_manager = page_manager
        self.vocab = ["hello", "how", "are", "you", "customer", "support", "query", "response", "thank", "issue", "resolve", "good", "day", "help"]

    def _generate_token(self, prompt_tokens):
        # Simulate generating a new token based on prompt
        if not prompt_tokens:
            return random.choice(self.vocab)
        
        # Simple context-aware token generation simulation
        last_token = prompt_tokens[-1]
        if "customer" in last_token or "support" in last_token: return "service"
        if "issue" in last_token: return "resolved"
        if "hello" in last_token: return "how"
        if "thank" in last_token: return "you"
        return random.choice(self.vocab)


    def process_query(self, sequence_id, prompt_text, max_new_tokens=10):
        print(f"\n--- Processing Sequence ID: {sequence_id} ---")
        print(f"User Query: {prompt_text}")
        current_tokens = prompt_text.lower().split()
        generated_tokens = []

        # Simulate initial KV cache allocation for the prompt
        prompt_length = len(current_tokens)
        initial_pages_needed = (prompt_length + self.page_manager.page_size - 1) // self.page_manager.page_size
        try:
            self.page_manager.allocate_pages(sequence_id, initial_pages_needed)
            print(f"Allocated {initial_pages_needed} initial pages for prompt (length: {prompt_length})")
        except MemoryError as e:
            print(f"Error allocating initial pages: {e}")
            return "Sorry, the system is currently overloaded. Please try again later."

        # Simulate writing prompt's KV to pages
        for i, token in enumerate(current_tokens):
            logical_block_index = i // self.page_manager.page_size
            physical_page_id = self.page_manager.get_physical_page_id(sequence_id, logical_block_index)
            if physical_page_id is None:
                # This should ideally not happen if pages were pre-allocated correctly
                print(f"Error: No physical page for logical block {logical_block_index} of sequence {sequence_id}")
                continue
            # Simulate KV data for the token
            kv_data = f"KV_{token}_{i}"
            self.page_manager.write_kv_to_page(physical_page_id, [kv_data])

        # Simulate token generation loop
        for token_idx in range(max_new_tokens):
            # Check if more space is needed for the next token's KV cache
            current_sequence_length = len(current_tokens) + len(generated_tokens)
            # Next token would require space if it crosses a page boundary
            next_token_logical_block_index = (current_sequence_length) // self.page_manager.page_size
            
            # Check if we already have a page for this logical block index
            existing_phys_id = self.page_manager.get_physical_page_id(sequence_id, next_token_logical_block_index)
            
            if existing_phys_id is None: # Need to allocate a new page
                try:
                    self.page_manager.allocate_pages(sequence_id, 1)
                    print(f"Allocated 1 new page for sequence {sequence_id} (logical block {next_token_logical_block_index})")
                except MemoryError:
                    print("Memory limit reached during generation. Truncating response.")
                    break
            
            new_token = self._generate_token(current_tokens + generated_tokens)
            generated_tokens.append(new_token)

            # Simulate writing new token's KV to the allocated page
            physical_page_id_for_new_token = self.page_manager.get_physical_page_id(sequence_id, next_token_logical_block_index)
            if physical_page_id_for_new_token is not None:
                kv_data = f"KV_{new_token}_{current_sequence_length}"
                self.page_manager.write_kv_to_page(physical_page_id_for_new_token, [kv_data])

            if "<END>" in new_token: # Simulate end token
                break
        
        response = " ".join(generated_tokens)
        print(f"LLM Response: {response}")
        return response


class CustomerSupportAssistant:
    def __init__(self):
        self.page_manager = KVCachePageManager()
        self.llm_simulator = LLMSimulator(self.page_manager)
        self.active_sequences = {}

    def handle_query(self, query):
        sequence_id = self.page_manager._get_next_sequence_id()
        self.active_sequences[sequence_id] = query # Keep track of active queries
        
        print(f"Current Memory Usage: {self.page_manager.get_memory_usage()}")
        response = self.llm_simulator.process_query(sequence_id, query)
        print(f"Current Memory Usage AFTER processing: {self.page_manager.get_memory_usage()}")
        
        # For simplicity in this simulation, we free pages immediately after response.
        # In a real system, pages might be kept for a while for potential reuse or batched processing.
        self.page_manager.free_pages(sequence_id)
        print(f"Freed pages for sequence {sequence_id}. Memory Usage: {self.page_manager.get_memory_usage()}")
        return response

    def run(self):
        print("\n--- AI-Powered Customer Support Assistant (Simulated PagedAttention) ---")
        print(f"System initialized with {TOTAL_PAGES} pages, each holding {PAGE_SIZE} KV items.\n")
        print("Type your query or 'exit' to quit.")
        
        while True:
            user_query = input("\nUser: ")
            if user_query.lower() == 'exit':
                print("Exiting Customer Support Assistant. Goodbye!")
                break
            
            self.handle_query(user_query)

if __name__ == "__main__":
    app = CustomerSupportAssistant()
    app.run()
