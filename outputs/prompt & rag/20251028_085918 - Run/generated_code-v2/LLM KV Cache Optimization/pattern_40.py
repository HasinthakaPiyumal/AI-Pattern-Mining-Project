import collections

class KVTensors:
    def __init__(self, conversation_id, size):
        self.conversation_id = conversation_id
        self.size = size
        self.data = [0] * size

class GPUManager:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.gpu_memory = {}
        self.current_memory_usage = 0

    def add(self, kv_tensors):
        if self.current_memory_usage + kv_tensors.size > self.max_memory:
            return False
        self.gpu_memory[kv_tensors.conversation_id] = kv_tensors
        self.current_memory_usage += kv_tensors.size
        return True

    def remove(self, conversation_id):
        if conversation_id in self.gpu_memory:
            kv_tensors = self.gpu_memory.pop(conversation_id)
            self.current_memory_usage -= kv_tensors.size
            return kv_tensors
        return None

    def get(self, conversation_id):
        return self.gpu_memory.get(conversation_id)

    def has(self, conversation_id):
        return conversation_id in self.gpu_memory

class HostManager:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.host_memory = {}
        self.current_memory_usage = 0

    def add(self, kv_tensors):
        if self.current_memory_usage + kv_tensors.size > self.max_memory:
            return False
        self.host_memory[kv_tensors.conversation_id] = kv_tensors
        self.current_memory_usage += kv_tensors.size
        return True

    def remove(self, conversation_id):
        if conversation_id in self.host_memory:
            kv_tensors = self.host_memory.pop(conversation_id)
            self.current_memory_usage -= kv_tensors.size
            return kv_tensors
        return None

    def get(self, conversation_id):
        return self.host_memory.get(conversation_id)

    def has(self, conversation_id):
        return conversation_id in self.host_memory

class SwapOutOnlyOnceCacheManager:
    def __init__(self, gpu_max_memory, host_max_memory):
        self.gpu_manager = GPUManager(gpu_max_memory)
        self.host_manager = HostManager(host_max_memory)
        self.evicted_once_nodes = set()
        self.gpu_lru = collections.deque()

    def _update_gpu_lru(self, conversation_id):
        if conversation_id in self.gpu_lru:
            self.gpu_lru.remove(conversation_id)
        self.gpu_lru.append(conversation_id)

    def _evict_from_gpu_for_space(self, required_size):
        while (self.gpu_manager.max_memory - self.gpu_manager.current_memory_usage < required_size and 
               self.gpu_lru):
            lru_conversation_id = self.gpu_lru.popleft()
            self.evict_from_gpu(lru_conversation_id)

    def evict_from_gpu(self, conversation_id):
        kv_tensors = self.gpu_manager.remove(conversation_id)
        if kv_tensors:
            if conversation_id in self.gpu_lru:
                self.gpu_lru.remove(conversation_id)

            if conversation_id not in self.evicted_once_nodes:
                if not self.host_manager.add(kv_tensors):
                    pass
                else:
                    self.evicted_once_nodes.add(conversation_id)

    def evict_from_host(self, conversation_id):
        if self.host_manager.remove(conversation_id):
            if conversation_id in self.evicted_once_nodes:
                self.evicted_once_nodes.remove(conversation_id)
            return True
        return False

    def get_kv_tensors(self, conversation_id, kv_size):
        if self.gpu_manager.has(conversation_id):
            self._update_gpu_lru(conversation_id)
            return self.gpu_manager.get(conversation_id)

        if self.host_manager.has(conversation_id):
            kv_tensors_from_host = self.host_manager.get(conversation_id)
            
            self._evict_from_gpu_for_space(kv_size)
            if self.gpu_manager.add(kv_tensors_from_host):
                self.host_manager.remove(conversation_id)
                if conversation_id in self.evicted_once_nodes:
                    self.evicted_once_nodes.remove(conversation_id)
                self._update_gpu_lru(conversation_id)
                return kv_tensors_from_host
            else:
                return kv_tensors_from_host

        new_kv_tensors = KVTensors(conversation_id, kv_size)
        
        self._evict_from_gpu_for_space(kv_size)
        if self.gpu_manager.add(new_kv_tensors):
            self._update_gpu_lru(conversation_id)
            return new_kv_tensors
        else:
            if self.host_manager.add(new_kv_tensors):
                self.evicted_once_nodes.add(conversation_id)
                return new_kv_tensors
            else:
                return None