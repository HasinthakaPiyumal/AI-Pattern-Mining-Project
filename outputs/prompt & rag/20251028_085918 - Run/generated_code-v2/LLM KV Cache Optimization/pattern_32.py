import torch
from collections import OrderedDict

class GPUMemorySimulator:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._memory = OrderedDict()

    def allocate(self, segment_id: str, tensors: torch.Tensor):
        if segment_id in self._memory:
            self._memory.move_to_end(segment_id)
        else:
            self._memory[segment_id] = tensors

    def free(self, segment_id: str):
        if segment_id in self._memory:
            del self._memory[segment_id]

    def get(self, segment_id: str) -> torch.Tensor or None:
        if segment_id in self._memory:
            self._memory.move_to_end(segment_id)
            return self._memory[segment_id]
        return None

    def is_full(self) -> bool:
        return len(self._memory) >= self._capacity

    def get_lru_segment_id(self) -> str or None:
        if not self._memory:
            return None
        return next(iter(self._memory))

class HostMemorySimulator:
    def __init__(self):
        self._memory = {}

    def store(self, segment_id: str, tensors: torch.Tensor):
        self._memory[segment_id] = tensors.cpu()

    def retrieve(self, segment_id: str) -> torch.Tensor or None:
        return self._memory.get(segment_id)

    def free(self, segment_id: str):
        if segment_id in self._memory:
            del self._memory[segment_id]

class SwapOutOnlyOnceCacheManager:
    def __init__(self, gpu_capacity: int):
        self.gpu_memory = GPUMemorySimulator(gpu_capacity)
        self.host_memory = HostMemorySimulator()
        self.copied_to_host_tracker = set()

    def put(self, segment_id: str, kv_tensors: torch.Tensor):
        if self.gpu_memory.is_full() and segment_id not in self.gpu_memory._memory:
            lru_segment_id = self.gpu_memory.get_lru_segment_id()
            if lru_segment_id:
                kv_to_evict = self.gpu_memory.get(lru_segment_id) # Get to remove it but also to get its value
                if lru_segment_id not in self.copied_to_host_tracker:
                    self.host_memory.store(lru_segment_id, kv_to_evict)
                    self.copied_to_host_tracker.add(lru_segment_id)
                self.gpu_memory.free(lru_segment_id)
        self.gpu_memory.allocate(segment_id, kv_tensors.cuda() if kv_tensors.device.type == 'cpu' else kv_tensors)

    def get(self, segment_id: str) -> torch.Tensor or None:
        tensors = self.gpu_memory.get(segment_id)
        if tensors is not None:
            return tensors
        else:
            tensors_from_host = self.host_memory.retrieve(segment_id)
            if tensors_from_host is not None:
                # Promote from host to GPU
                if self.gpu_memory.is_full():
                    lru_segment_id = self.gpu_memory.get_lru_segment_id()
                    if lru_segment_id:
                        kv_to_evict = self.gpu_memory.get(lru_segment_id)
                        if lru_segment_id not in self.copied_to_host_tracker:
                            self.host_memory.store(lru_segment_id, kv_to_evict)
                            self.copied_to_host_tracker.add(lru_segment_id)
                        self.gpu_memory.free(lru_segment_id)
                self.gpu_memory.allocate(segment_id, tensors_from_host.cuda())
                return tensors_from_host.cuda()
            return None