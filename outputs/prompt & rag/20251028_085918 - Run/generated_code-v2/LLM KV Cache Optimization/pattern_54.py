class KV_Cache:
    def __init__(self):
        self._cache = {}

    def set(self, key, value):
        self._cache[key] = value

    def get(self, key):
        return self._cache.get(key)

    def delete(self, key):
        if key in self._cache:
            del self._cache[key]

    def get_all_keys(self):
        return list(self._cache.keys())

    def clear(self):
        self._cache = {}

class GPUMemory:
    def __init__(self):
        self._cache = KV_Cache()
        self._failed = False

    def set(self, key, value):
        if not self._failed:
            self._cache.set(key, value)

    def get(self, key):
        if not self._failed:
            return self._cache.get(key)
        return None

    def delete(self, key):
        if not self._failed:
            self._cache.delete(key)

    def get_all_keys(self):
        if not self._failed:
            return self._cache.get_all_keys()
        return []

    def clear(self):
        self._cache.clear()

    def fail(self):
        self._failed = True
        self._cache.clear()

    def is_failed(self):
        return self._failed

    def restore(self):
        self._failed = False

class HostMemory:
    def __init__(self):
        self._cache = KV_Cache()

    def set(self, key, value):
        self._cache.set(key, value)

    def get(self, key):
        return self._cache.get(key)

    def delete(self, key):
        self._cache.delete(key)

    def get_all_keys(self):
        return self._cache.get_all_keys()

    def clear(self):
        self._cache.clear()

class MedAssistRAGSystem:
    def __init__(self, critical_keys=None):
        self.gpu_memory = GPUMemory()
        self.host_memory = HostMemory()
        self.critical_keys = set(critical_keys) if critical_keys else set()

    def _is_critical(self, key):
        return key in self.critical_keys

    def set_kv(self, key, value, is_critical_override=False):
        self.gpu_memory.set(key, value)
        if self._is_critical(key) or is_critical_override:
            self.host_memory.set(key, value)
            if is_critical_override and not self._is_critical(key):
                self.critical_keys.add(key)

    def get_kv(self, key):
        gpu_value = self.gpu_memory.get(key)
        if gpu_value is not None:
            return gpu_value
        elif self.gpu_memory.is_failed() and self._is_critical(key):
            return self.host_memory.get(key)
        return None

    def delete_kv(self, key):
        self.gpu_memory.delete(key)
        if self._is_critical(key):
            self.host_memory.delete(key)
            self.critical_keys.discard(key)

    def simulate_gpu_failure(self):
        print("Simulating GPU failure...")
        self.gpu_memory.fail()
        print("GPU memory cleared and marked as failed.")

    def recover_from_gpu_failure(self):
        if self.gpu_memory.is_failed():
            print("Recovering from GPU failure...")
            self.gpu_memory.restore()
            for key in self.critical_keys:
                critical_value = self.host_memory.get(key)
                if critical_value is not None:
                    self.gpu_memory.set(key, critical_value)
                    print(f"Restored critical KV: {key}")
            print("Recovery complete. Critical KV nodes restored to GPU.")
        else:
            print("GPU is not in a failed state. No recovery needed.")