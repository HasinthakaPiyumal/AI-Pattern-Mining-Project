import collections

class KVPage:
    """
    Represents a single page in the Key-Value cache.
    In a real system, this would point to a specific memory block on a GPU.
    """
    def __init__(self, page_id: int, capacity: int):
        self.page_id = page_id
        self.capacity = capacity  # Number of tokens this page can hold
        self.sequence_ids = []  # To track which sequences are using parts of this page

    def __repr__(self):
        return f"KVPage(id={self.page_id}, capacity={self.capacity})"

class PagedAttentionKVManager:
    """
    Manages the allocation and deallocation of KV cache pages.
    Simulates the core idea of PagedAttention by handling pages like OS virtual memory.
    """
    def __init__(self, total_pages: int, page_size: int):
        if total_pages <= 0 or page_size <= 0:
            raise ValueError("total_pages and page_size must be positive integers.")

        self.total_pages = total_pages
        self.page_size = page_size  # Max tokens per page
        self.available_pages = collections.deque(KVPage(i, page_size) for i in range(total_pages))
        self.used_pages = {}
        self.sequence_page_map = collections.defaultdict(list) # Maps sequence_id to list of page_ids

        print(f"[KVManager] Initialized with {total_pages} pages, each with capacity {page_size} tokens.")

    def _get_page(self) -> KVPage or None:
        if not self.available_pages:
            return None # No pages left
        page = self.available_pages.popleft()
        self.used_pages[page.page_id] = page
        return page

    def allocate_pages_for_sequence(self, sequence_id: int, num_tokens_needed: int) -> list[KVPage]:
        """
        Allocates pages for a given sequence based on the number of tokens needed.
        If a sequence already has pages, it tries to extend them or allocate new ones.
        """
        if num_tokens_needed <= 0:
            return []

        print(f"[KVManager] Sequence {sequence_id} requesting {num_tokens_needed} tokens.")

        current_pages_for_sequence = self.sequence_page_map[sequence_id]
        current_allocated_capacity = sum(page.capacity for page in current_pages_for_sequence)

        pages_to_allocate = []

        # Determine how many *new* pages are needed beyond current allocation
        tokens_remaining_to_allocate = num_tokens_needed - current_allocated_capacity
        if tokens_remaining_to_allocate <= 0:
            # Already have enough capacity, no new pages needed
            print(f"[KVManager] Sequence {sequence_id} already has enough capacity ({current_allocated_capacity}) for {num_tokens_needed} tokens.")
            return current_pages_for_sequence

        # Calculate how many *additional* pages are required
        additional_pages_count = (tokens_remaining_to_allocate + self.page_size - 1) // self.page_size

        print(f"[KVManager] Sequence {sequence_id} needs {additional_pages_count} additional pages for {tokens_remaining_to_allocate} tokens.")

        for _ in range(additional_pages_count):
            page = self._get_page()
            if page is None:
                print(f"[KVManager] WARNING: No more pages available to allocate for sequence {sequence_id}!")
                # Revert any allocations in this call if not all pages could be allocated
                for allocated_page in pages_to_allocate:
                    self.free_pages([allocated_page.page_id]) # free them back
                return [] # Indicate failure to allocate
            pages_to_allocate.append(page)
            self.sequence_page_map[sequence_id].append(page)

        for page in pages_to_allocate:
            if sequence_id not in page.sequence_ids:
                page.sequence_ids.append(sequence_id)

        print(f"[KVManager] Allocated {len(pages_to_allocate)} new pages for sequence {sequence_id}. Total pages for sequence: {len(self.sequence_page_map[sequence_id])}")
        return self.sequence_page_map[sequence_id]

    def free_pages_for_sequence(self, sequence_id: int):
        """
        Frees all pages associated with a given sequence.
        """
        if sequence_id not in self.sequence_page_map:
            print(f"[KVManager] No pages found for sequence {sequence_id} to free.")
            return

        pages_to_free = self.sequence_page_map.pop(sequence_id)
        freed_page_ids = []
        for page in pages_to_free:
            if page.page_id in self.used_pages:
                # Remove sequence_id from page's tracking list
                if sequence_id in page.sequence_ids:
                    page.sequence_ids.remove(sequence_id)
                # Only free the page if no other sequence is using it
                if not page.sequence_ids:
                    freed_page_ids.append(page.page_id)
                    del self.used_pages[page.page_id]
                    self.available_pages.append(page)

        print(f"[KVManager] Freed {len(freed_page_ids)} actual pages for sequence {sequence_id}.")
        if freed_page_ids:
            print(f"[KVManager] Pages freed: {freed_page_ids}")

    def get_status(self) -> dict:
        """
        Returns the current status of the KV cache manager.
        """
        return {
            "total_pages": self.total_pages,
            "available_pages": len(self.available_pages),
            "used_pages": len(self.used_pages),
            "page_size": self.page_size,
            "sequences_active": len(self.sequence_page_map)
        }
