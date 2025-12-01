import torch
from typing import Dict, Set, Optional, Tuple
from collections import OrderedDict

class KVCacheManager:
    def __init__(self, gpu_capacity: int):
        self.gpu_capacity = gpu_capacity
        self.gpu_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.host_cache: Dict[str, torch.Tensor] = {}
        self.swapped_out_once_tracker: Set[str] = set()

    def _make_space_in_gpu(self):
        if len(self.gpu_cache) >= self.gpu_capacity:
            lru_conversation_id, _ = self.gpu_cache.items().__iter__().__next__()
            self.evict_from_gpu(lru_conversation_id)

    def put_kv_cache(self, conversation_id: str, kv_tensors: torch.Tensor):
        if conversation_id in self.gpu_cache:
            self.gpu_cache.pop(conversation_id)
        elif conversation_id in self.host_cache:
            # If it's in host, we are promoting it, so remove from host first
            del self.host_cache[conversation_id]
            self.swapped_out_once_tracker.add(conversation_id) # Ensure it's tracked as swapped out once

        self._make_space_in_gpu()
        self.gpu_cache[conversation_id] = kv_tensors
        # Mark as recently used by moving to the end
        self.gpu_cache.move_to_end(conversation_id)

    def evict_from_gpu(self, conversation_id: str):
        if conversation_id in self.gpu_cache:
            kv_tensors = self.gpu_cache.pop(conversation_id)
            if conversation_id not in self.swapped_out_once_tracker:
                self.host_cache[conversation_id] = kv_tensors.clone() # Simulate copy to host memory
                self.swapped_out_once_tracker.add(conversation_id)
            print(f"Evicted {conversation_id} from GPU. Copied to host: {conversation_id in self.host_cache}")

    def promote_to_gpu(self, conversation_id: str):
        if conversation_id in self.host_cache:
            print(f"Promoting {conversation_id} from host to GPU.")
            kv_tensors = self.host_cache.pop(conversation_id)
            self.put_kv_cache(conversation_id, kv_tensors)
        elif conversation_id not in self.gpu_cache:
            print(f"Attempted to promote {conversation_id}, but not found in host or GPU.")

    def get_kv_cache(self, conversation_id: str) -> Optional[torch.Tensor]:
        if conversation_id in self.gpu_cache:
            self.gpu_cache.move_to_end(conversation_id) # Mark as recently used
            print(f"Retrieved {conversation_id} from GPU.")
            return self.gpu_cache[conversation_id]
        elif conversation_id in self.host_cache:
            print(f"Retrieved {conversation_id} from host. Promoting to GPU.")
            self.promote_to_gpu(conversation_id)
            # After promotion, it should be in GPU cache
            return self.gpu_cache.get(conversation_id)
        return None

    def remove_from_all_caches(self, conversation_id: str):
        if conversation_id in self.gpu_cache:
            del self.gpu_cache[conversation_id]
        if conversation_id in self.host_cache:
            del self.host_cache[conversation_id]
        if conversation_id in self.swapped_out_once_tracker:
            self.swapped_out_once_tracker.remove(conversation_id)
        print(f"Removed {conversation_id} from all caches.")

class LLMSimulator:
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager

    def process_turn(self, conversation_id: str, user_input: str, current_kv_cache: Optional[torch.Tensor] = None) -> Tuple[str, torch.Tensor]:
        # Simulate LLM processing and generating new KV tensors
        print(f"LLM processing for {conversation_id} with input: '{user_input}'")
        if current_kv_cache is None:
            # Simulate initial KV cache for a new conversation
            new_kv_tensors = torch.randn(1, 10, 512) # Example tensor
            response = f"Hello! How can I help you with '{user_input}'? (New conversation)"
        else:
            # Simulate updating existing KV cache
            new_kv_tensors = current_kv_cache + torch.randn(1, 1, 512) # Append or modify
            response = f"Continuing on '{user_input}'. (KV cache updated)"
        
        self.kv_cache_manager.put_kv_cache(conversation_id, new_kv_tensors)
        return response, new_kv_tensors

class Chatbot:
    def __init__(self, gpu_capacity: int):
        self.kv_cache_manager = KVCacheManager(gpu_capacity)
        self.llm_simulator = LLMSimulator(self.kv_cache_manager)
        self.active_conversations: Dict[str, torch.Tensor] = {}

    def start_conversation(self, conversation_id: str):
        print(f"\n--- Starting Conversation: {conversation_id} ---")
        initial_response, initial_kv = self.llm_simulator.process_turn(conversation_id, "Hi there!")
        self.active_conversations[conversation_id] = initial_kv
        print(f"Chatbot: {initial_response}")

    def send_message(self, conversation_id: str, message: str):
        print(f"\nUser ({conversation_id}): {message}")
        current_kv = self.kv_cache_manager.get_kv_cache(conversation_id)
        response, updated_kv = self.llm_simulator.process_turn(conversation_id, message, current_kv)
        self.active_conversations[conversation_id] = updated_kv
        print(f"Chatbot: {response}")

    def end_conversation(self, conversation_id: str):
        print(f"\n--- Ending Conversation: {conversation_id} ---")
        self.kv_cache_manager.remove_from_all_caches(conversation_id)
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]


if __name__ == "__main__":
    # Example Usage
    chatbot = Chatbot(gpu_capacity=2)

    # Conversation 1: Active
    chatbot.start_conversation("conv_1")
    chatbot.send_message("conv_1", "What are your capabilities?")

    # Conversation 2: Active
    chatbot.start_conversation("conv_2")
    chatbot.send_message("conv_2", "Tell me a joke.")

    # Conversation 3: Starts, forcing conv_1 eviction (first time)
    chatbot.start_conversation("conv_3")
    chatbot.send_message("conv_3", "How's the weather?")
    # conv_1 should be evicted and copied to host

    # Conversation 1: Resumes, promoting from host to GPU
    chatbot.send_message("conv_1", "Can you elaborate on that?")
    # conv_2 should be evicted and copied to host

    # Conversation 3: Continues
    chatbot.send_message("conv_3", "Thanks!")

    # Conversation 2: Resumes, promoting from host to GPU
    chatbot.send_message("conv_2", "Why was the math book sad?")
    # conv_1 should be evicted, but *not* copied again to host (already there)

    # Conversation 1: Resumes again
    chatbot.send_message("conv_1", "Okay, I understand.")
    # conv_3 should be evicted, but *not* copied again to host (already there)

    # End some conversations
    chatbot.end_conversation("conv_2")
    chatbot.end_conversation("conv_3")

    # Final check on conv_1
    chatbot.send_message("conv_1", "Goodbye!")
    chatbot.end_conversation("conv_1")

    print("\n--- Final Cache States ---")
    print(f"GPU Cache ({len(chatbot.kv_cache_manager.gpu_cache)} items): {list(chatbot.kv_cache_manager.gpu_cache.keys())}")
    print(f"Host Cache ({len(chatbot.kv_cache_manager.host_cache)} items): {list(chatbot.kv_cache_manager.host_cache.keys())}")
    print(f"Swapped Out Once Tracker ({len(chatbot.kv_cache_manager.swapped_out_once_tracker)} items): {list(chatbot.kv_cache_manager.swapped_out_once_tracker)}")
