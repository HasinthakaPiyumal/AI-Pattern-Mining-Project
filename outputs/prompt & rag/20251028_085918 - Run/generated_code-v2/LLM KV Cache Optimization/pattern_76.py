class KVNode:
    def __init__(self, node_id, kv_data_size):
        self.node_id = node_id
        self.kv_data_size = kv_data_size
        self.on_gpu = False
        self.on_host = False
        self.has_host_copy = False

class GPUMemorySimulator:
    def __init__(self):
        self.allocated_memory = set()

    def allocate(self, node_id, data_size):
        self.allocated_memory.add(node_id)
        print(f"GPU: Allocated {data_size}MB for node {node_id}")

    def free(self, node_id):
        if node_id in self.allocated_memory:
            self.allocated_memory.remove(node_id)
            print(f"GPU: Freed memory for node {node_id}")

class HostMemorySimulator:
    def __init__(self):
        self.stored_data = {}

    def copy_to_host(self, node_id, data_size):
        if node_id not in self.stored_data:
            self.stored_data[node_id] = data_size
            print(f"Host: Copied {data_size}MB for node {node_id} to host memory")

    def free_from_host(self, node_id):
        if node_id in self.stored_data:
            del self.stored_data[node_id]
            print(f"Host: Freed memory for node {node_id} from host memory")

class CacheManager:
    def __init__(self):
        self.gpu_simulator = GPUMemorySimulator()
        self.host_simulator = HostMemorySimulator()
        self.nodes = {}

    def add_conversation(self, node_id, kv_data_size):
        if node_id not in self.nodes:
            self.nodes[node_id] = KVNode(node_id, kv_data_size)
            print(f"CacheManager: Added conversation node {node_id} with size {kv_data_size}MB")
        else:
            print(f"CacheManager: Conversation node {node_id} already exists.")

    def promote_to_gpu(self, node_id):
        if node_id not in self.nodes:
            print(f"Error: Node {node_id} not found.")
            return

        node = self.nodes[node_id]
        if not node.on_gpu:
            self.gpu_simulator.allocate(node_id, node.kv_data_size)
            node.on_gpu = True
            print(f"CacheManager: Promoted node {node_id} to GPU.")
        else:
            print(f"CacheManager: Node {node_id} already on GPU.")

    def evict_from_gpu(self, node_id):
        if node_id not in self.nodes:
            print(f"Error: Node {node_id} not found.")
            return

        node = self.nodes[node_id]
        if node.on_gpu:
            if not node.has_host_copy:
                self.host_simulator.copy_to_host(node_id, node.kv_data_size)
                node.on_host = True
                node.has_host_copy = True
                print(f"CacheManager: Evicted node {node_id} from GPU and copied to host (first time).")
            else:
                print(f"CacheManager: Evicted node {node_id} from GPU (host copy already exists).")
            
            self.gpu_simulator.free(node_id)
            node.on_gpu = False
        else:
            print(f"CacheManager: Node {node_id} is not on GPU.")

    def retire_conversation(self, node_id):
        if node_id not in self.nodes:
            print(f"Error: Node {node_id} not found.")
            return

        node = self.nodes[node_id]
        if node.on_gpu:
            self.gpu_simulator.free(node_id)
            node.on_gpu = False
        
        if node.on_host:
            self.host_simulator.free_from_host(node_id)
            node.on_host = False
            
        del self.nodes[node_id]
        print(f"CacheManager: Retired conversation node {node_id} from all caches.")


if __name__ == "__main__":
    cache_manager = CacheManager()

    # Simulate starting conversations
    cache_manager.add_conversation("conv_A", 100)
    cache_manager.add_conversation("conv_B", 150)
    cache_manager.add_conversation("conv_C", 80)

    print("\n--- Simulation Step 1: Promote to GPU ---")
    cache_manager.promote_to_gpu("conv_A")
    cache_manager.promote_to_gpu("conv_B")

    print("\n--- Simulation Step 2: Evict from GPU (first time for conv_A) ---")
    cache_manager.evict_from_gpu("conv_A")

    print("\n--- Simulation Step 3: Promote conv_A back to GPU ---")
    cache_manager.promote_to_gpu("conv_A")

    print("\n--- Simulation Step 4: Evict from GPU (second time for conv_A, no host copy needed) ---")
    cache_manager.evict_from_gpu("conv_A")

    print("\n--- Simulation Step 5: Evict conv_B (first time) ---")
    cache_manager.evict_from_gpu("conv_B")

    print("\n--- Simulation Step 6: Retire conversations ---")
    cache_manager.retire_conversation("conv_A")
    cache_manager.retire_conversation("conv_B")
    cache_manager.retire_conversation("conv_C")

    print("\n--- Final State Check ---")
    print(f"GPU allocated memory: {cache_manager.gpu_simulator.allocated_memory}")
    print(f"Host stored data: {cache_manager.host_simulator.stored_data}")
    print(f"Nodes in manager: {cache_manager.nodes.keys()}")