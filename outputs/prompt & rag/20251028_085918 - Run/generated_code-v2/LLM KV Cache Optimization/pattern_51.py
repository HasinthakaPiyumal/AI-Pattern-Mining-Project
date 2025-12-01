import random

class KVCachePage:
    def __init__(self, page_id):
        self.page_id = page_id
        self.is_allocated = False
        self.sequence_id = None

class KVCacheManager:
    def __init__(self, total_pages):
        self.total_pages = total_pages
        self.pages = {i: KVCachePage(i) for i in range(total_pages)}
        self.allocated_sequences = {}

    def allocate_pages(self, sequence_id, num_pages):
        if sequence_id in self.allocated_sequences:
            # Already allocated, append more pages if needed or handle error
            return False

        available_pages = [page for page in self.pages.values() if not page.is_allocated]

        if len(available_pages) < num_pages:
            return False

        allocated_page_ids = []
        for _ in range(num_pages):
            page = available_pages.pop(random.randrange(len(available_pages)))
            page.is_allocated = True
            page.sequence_id = sequence_id
            allocated_page_ids.append(page.page_id)
        self.allocated_sequences[sequence_id] = allocated_page_ids
        return True

    def free_pages(self, sequence_id):
        if sequence_id not in self.allocated_sequences:
            return False

        for page_id in self.allocated_sequences[sequence_id]:
            page = self.pages[page_id]
            page.is_allocated = False
            page.sequence_id = None
        del self.allocated_sequences[sequence_id]
        return True

    def get_memory_utilization(self):
        allocated_count = sum(1 for page in self.pages.values() if page.is_allocated)
        return (allocated_count / self.total_pages) * 100

    def get_fragmentation_status(self):
        # A simplistic view of fragmentation: if there are free pages but not contiguous blocks of a certain size
        # For this simulation, we'll just report free pages vs. allocated pages
        free_count = sum(1 for page in self.pages.values() if not page.is_allocated)
        return f"{free_count} free pages, {len(self.allocated_sequences)} active sequences"

# Simulation Logic
if __name__ == "__main__":
    TOTAL_MEMORY_PAGES = 100
    MAX_QUERY_PAGES = 10
    NUM_SIMULATION_STEPS = 20

    kv_cache_manager = KVCacheManager(TOTAL_MEMORY_PAGES)
    active_queries = {}
    query_id_counter = 0

    print(f"--- PagedAttention Simulation with {TOTAL_MEMORY_PAGES} total pages ---")

    for step in range(NUM_SIMULATION_STEPS):
        print(f"\n--- Step {step + 1} ---")

        # Simulate new query arrival
        if random.random() < 0.7:  # 70% chance of a new query
            query_id_counter += 1
            new_query_id = f"query_{query_id_counter}"
            pages_needed = random.randint(1, MAX_QUERY_PAGES)
            if kv_cache_manager.allocate_pages(new_query_id, pages_needed):
                active_queries[new_query_id] = pages_needed
                print(f"  New query {new_query_id} arrived, requested {pages_needed} pages. Allocated successfully.")
            else:
                print(f"  New query {new_query_id} arrived, requested {pages_needed} pages. Allocation failed (out of memory).")

        # Simulate query completion (randomly select an active query to finish)
        if active_queries and random.random() < 0.5:  # 50% chance of a query finishing
            query_to_free = random.choice(list(active_queries.keys()))
            if kv_cache_manager.free_pages(query_to_free):
                print(f"  Query {query_to_free} completed, pages freed.")
                del active_queries[query_to_free]
            else:
                print(f"  Error freeing pages for query {query_to_free}.")

        print(f"  Memory Utilization: {kv_cache_manager.get_memory_utilization():.2f}%")
        print(f"  Fragmentation Status: {kv_cache_manager.get_fragmentation_status()}")

    print("\n--- Simulation End ---")
    print(f"Final Memory Utilization: {kv_cache_manager.get_memory_utilization():.2f}%")
    print(f"Final Fragmentation Status: {kv_cache_manager.get_fragmentation_status()}")