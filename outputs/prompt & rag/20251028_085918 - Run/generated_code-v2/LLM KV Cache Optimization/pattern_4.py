import collections

class KVNode:
    """Represents a Key-Value cache node for a conversational context.
    In a real LLM, this would hold actual tensor data for keys and values.
    """
    def __init__(self, node_id: str, data_size: int):
        self.node_id = node_id
        self.data_size = data_size # Simulate size for memory tracking
        self.has_host_copy = False # Flag for Swap-Out-Only-Once strategy

    def __repr__(self):
        return f"KVNode(ID={self.node_id}, Size={self.data_size} bytes, HasHostCopy={self.has_host_copy})"

class GPUMemory:
    """Simulates fast GPU memory with a fixed capacity, using an LRU eviction policy.
    Stores KVNode objects.
    """
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.current_usage = 0
        self.cache = collections.OrderedDict() # LRU cache: items at the end are most recently used

    def get_node(self, node_id: str) -> KVNode or None:
        if node_id in self.cache:
            self.cache.move_to_end(node_id) # Mark as recently used
            return self.cache[node_id]
        return None

    def add_node(self, node: KVNode) -> bool:
        """Attempts to add a node. Returns True if successful, False if GPU is full.
        If node is already present, it's marked as recently used.
        """
        if node.data_size > self.capacity:
            print(f"ERROR: Node {node.node_id} ({node.data_size} bytes) is too large for GPU capacity ({self.capacity} bytes).")
            return False

        if node.node_id in self.cache:
            self.cache.move_to_end(node.node_id)
            return True # Node already present and LRU updated

        if self.current_usage + node.data_size <= self.capacity:
            self.cache[node.node_id] = node
            self.current_usage += node.data_size
            self.cache.move_to_end(node.node_id) # Mark as recently used
            print(f"[GPU] Added node {node.node_id}. Usage: {self.current_usage}/{self.capacity}")
            return True
        else:
            return False # GPU is full, eviction needed

    def evict_lru(self) -> KVNode or None:
        """Evicts the Least Recently Used node from GPU memory.
        Returns the evicted KVNode object.
        """
        if not self.cache:
            return None

        lru_node_id, lru_node = self.cache.popitem(last=False) # Get and remove LRU item
        self.current_usage -= lru_node.data_size
        print(f"[GPU] Evicted LRU node {lru_node_id}. Usage: {self.current_usage}/{self.capacity}")
        return lru_node

    def remove_node(self, node_id: str) -> KVNode or None:
        """Removes a specific node from GPU memory (e.g., when promoted to host and then back).
        """
        if node_id in self.cache:
            node = self.cache.pop(node_id)
            self.current_usage -= node.data_size
            print(f"[GPU] Removed node {node_id} directly. Usage: {self.current_usage}/{self.capacity}")
            return node
        return None

class HostMemory:
    """Simulates slower Host memory. Stores copies of KVNode objects.
    For simplicity, capacity is not enforced, but usage is tracked.
    """
    def __init__(self):
        self.cache = {}
        self.current_usage = 0

    def get_node(self, node_id: str) -> KVNode or None:
        return self.cache.get(node_id)

    def add_node(self, node: KVNode) -> bool:
        """Adds a node to Host memory. Marks the node as having a host copy.
        Returns True if added, False if already present.
        """
        if node.node_id not in self.cache:
            self.cache[node.node_id] = node
            self.current_usage += node.data_size
            node.has_host_copy = True # Crucial: mark that host now has a copy
            print(f"[Host] Copied node {node.node_id} to Host. Usage: {self.current_usage}")
            return True
        print(f"[Host] Node {node.node_id} already in Host memory. No new copy.")
        return False

    def remove_node(self, node_id: str) -> KVNode or None:
        """Removes a node from Host memory (e.g., when it's re-promoted to GPU).
        """
        if node_id in self.cache:
            node = self.cache.pop(node_id)
            self.current_usage -= node.data_size
            print(f"[Host] Removed node {node_id} from Host. Usage: {self.current_usage}")
            return node
        return None

class SwapOutOnlyOnceCacheManager:
    """Manages a hierarchical GPU-Host cache with the Swap-Out-Only-Once strategy.
    This manager orchestrates node access, eviction, and promotion between memory levels.
    """
    def __init__(self, gpu_capacity_bytes: int):
        self.gpu_memory = GPUMemory(gpu_capacity_bytes)
        self.host_memory = HostMemory()
        self.all_known_nodes = {} # Tracks all nodes and their `has_host_copy` state

    def _get_or_create_node(self, node_id: str, data_size: int) -> KVNode:
        """Retrieves an existing KVNode or creates a new one if it doesn't exist.
        Ensures consistent `KVNode` instances across the cache system.
        """
        if node_id not in self.all_known_nodes:
            self.all_known_nodes[node_id] = KVNode(node_id, data_size)
        return self.all_known_nodes[node_id]

    def access_node(self, node_id: str, data_size: int = 100):
        """Accesses a node by its ID. Implements the Swap-Out-Only-Once logic.
        If the node is not in GPU, it's retrieved from Host (if available) or created,
        and then promoted to GPU, potentially causing an eviction.
        """
        print(f"\n>>> Accessing Node '{node_id}' (Data size for new node: {data_size} bytes) <<<")

        # 1. Check if node is already in GPU
        gpu_node = self.gpu_memory.get_node(node_id)
        if gpu_node:
            print(f"  Node '{node_id}' found in GPU. LRU state updated.")
            return gpu_node

        print(f"  Node '{node_id}' not in GPU. Checking host or creating.")
        node_to_add = self._get_or_create_node(node_id, data_size)

        # 2. If not in GPU, check Host memory
        if node_to_add.has_host_copy and self.host_memory.get_node(node_id):
            print(f"  Node '{node_id}' found in Host memory. Promoting to GPU.")
            self.host_memory.remove_node(node_id) # Remove from Host as it's moving to GPU
            # Note: `node_to_add.has_host_copy` remains True, which is key for strategy
        else:
            print(f"  Node '{node_id}' not in Host (or no host copy flag). Treating as new/first access.")
            # If it's truly new, node_to_add.has_host_copy is False initially.
            # If it was in host but removed, it should retain has_host_copy = True from `all_known_nodes`.

        # 3. Add to GPU, handling eviction if GPU is full
        while not self.gpu_memory.add_node(node_to_add):
            # GPU is full, an eviction is required
            evicted_node = self.gpu_memory.evict_lru()
            if evicted_node is None:
                print("CRITICAL ERROR: GPU is empty but cannot add node. This should not happen.")
                break

            print(f"  GPU full. Evicting LRU Node '{evicted_node.node_id}'.")
            # Apply the Swap-Out-Only-Once strategy
            if not evicted_node.has_host_copy:
                print(f"  Strategy: Node '{evicted_node.node_id}' has NO host copy. Transferring data to Host.")
                self.host_memory.add_node(evicted_node) # This also sets evicted_node.has_host_copy = True
            else:
                print(f"  Strategy: Node '{evicted_node.node_id}' ALREADY has a host copy. Freeing GPU memory (no host transfer).")
                # The host memory retains its copy until the node is explicitly removed from the entire cache
                # (e.g., conversation ends completely and is archived).

        print(f"<<< Node '{node_id}' access completed. >>>")
        return node_to_add

    def current_cache_state(self):
        """Prints the current state of both GPU and Host memory caches."""
        print("\n--- Current Cache State ---")
        print(f"GPU Memory (Capacity: {self.gpu_memory.capacity} bytes, Used: {self.gpu_memory.current_usage} bytes):")
        if not self.gpu_memory.cache:
            print("  <Empty>")
        for node_id, node in self.gpu_memory.cache.items():
            print(f"  - {node}")

        print(f"Host Memory (Used: {self.host_memory.current_usage} bytes):")
        if not self.host_memory.cache:
            print("  <Empty>")
        for node_id, node in self.host_memory.cache.items():
            print(f"  - {node}")
        print("---------------------------")

# --- Example Usage for a Customer Support Chatbot Simulation ---
if __name__ == "__main__":
    # Initialize the cache manager with a GPU capacity (e.g., 300 bytes for 3 small nodes)
    cache_manager = SwapOutOnlyOnceCacheManager(gpu_capacity_bytes=300)

    print("\n--- Scenario: Initial chat sessions (A, B, C) are active ---")
    # Simulate KV cache nodes for different conversation contexts
    cache_manager.access_node("chat_A", data_size=100)
    cache_manager.access_node("chat_B", data_size=100)
    cache_manager.access_node("chat_C", data_size=100)
    cache_manager.current_cache_state()

    print("\n--- Scenario: User A sends another message (accesses chat_A) ---")
    # chat_A is accessed, it's already in GPU, so its LRU position is updated.
    cache_manager.access_node("chat_A")
    cache_manager.current_cache_state()

    print("\n--- Scenario: A new chat session (D) starts. GPU is full.---")
    # chat_D needs to be added. chat_B is LRU and will be evicted.
    # chat_B has not been to host before, so it's copied.
    cache_manager.access_node("chat_D", data_size=100)
    cache_manager.current_cache_state()

    print("\n--- Scenario: User C sends another message (accesses chat_C) ---")
    # chat_C is in GPU, its LRU position is updated.
    cache_manager.access_node("chat_C")
    cache_manager.current_cache_state()

    print("\n--- Scenario: Another new chat session (E) starts. GPU is full.---")
    # chat_E needs to be added. chat_A is LRU and will be evicted.
    # chat_A has not been to host before, so it's copied.
    cache_manager.access_node("chat_E", data_size=100)
    cache_manager.current_cache_state()

    print("\n--- Scenario: User B re-engages (accesses chat_B). It was in Host.---")
    # chat_B is in Host. It will be promoted back to GPU.
    # This will cause chat_D (LRU in GPU) to be evicted.
    # chat_D has not been to host before, so it's copied.
    cache_manager.access_node("chat_B")
    cache_manager.current_cache_state()

    print("\n--- Scenario: User A re-engages (accesses chat_A). It was in Host.---")
    # chat_A is in Host. It will be promoted back to GPU.
    # This will cause chat_C (LRU in GPU) to be evicted.
    # chat_C has not been to host before, so it's copied.
    cache_manager.access_node("chat_A")
    cache_manager.current_cache_state()

    print("\n--- Scenario: Another new chat session (F) starts. GPU is full.---")
    # chat_F needs to be added. chat_E (LRU in GPU) will be evicted.
    # chat_E has not been to host before, so it's copied.
    cache_manager.access_node("chat_F", data_size=100)
    cache_manager.current_cache_state()

    print("\n--- Scenario: User B sends another message (accesses chat_B). Already in GPU.---")
    cache_manager.access_node("chat_B")
    cache_manager.current_cache_state()

    print("\n--- Scenario: Another new chat session (G) starts. GPU is full.---")
    # chat_G needs to be added. chat_A (LRU in GPU) will be evicted.
    # IMPORTANT: chat_A *already* has a host copy (from a previous eviction).
    # So, its GPU memory is freed, but NO new copy is made to Host.
    cache_manager.access_node("chat_G", data_size=100)
    cache_manager.current_cache_state()

    print("\n--- Scenario: User D re-engages (accesses chat_D). It was in Host.---")
    # chat_D is in Host. It will be promoted back to GPU.
    # This will cause chat_F (LRU in GPU) to be evicted.
    # chat_F has not been to host before, so it's copied.
    cache_manager.access_node("chat_D")
    cache_manager.current_cache_state()

    print("\n--- Final Check: Observe chat_A. It was evicted twice but only copied once.---")
    # chat_A should still have has_host_copy=True and be in Host memory.
    print(f"State of chat_A in host memory: {cache_manager.host_memory.get_node('chat_A')}")
    print(f"All known nodes for chat_A: {cache_manager.all_known_nodes.get('chat_A')}")