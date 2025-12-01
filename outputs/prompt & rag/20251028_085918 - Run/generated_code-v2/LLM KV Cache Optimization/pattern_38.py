import streamlit as st
import os
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import uuid

# --- 1. Simulate KV Cache Management ---
class KVCacheManager:
    def __init__(self):
        self.gpu_memory_cache = {}
        self.host_memory_cache = {}
        self.cache_dir = "./kv_cache_backup"
        os.makedirs(self.cache_dir, exist_ok=True)
        st.session_state.gpu_active = True

    def add_to_gpu_cache(self, key, value, is_critical=False, tag=None):
        if st.session_state.gpu_active:
            self.gpu_memory_cache[key] = {"value": value, "is_critical": is_critical, "tag": tag}
            if is_critical:
                self.replicate_to_host(key, value, tag)
        else:
            st.warning("GPU is currently offline. Cannot add to GPU cache.")

    def get_from_gpu_cache(self, key):
        if st.session_state.gpu_active:
            return self.gpu_memory_cache.get(key, None)
        return None

    def replicate_to_host(self, key, value, tag):
        self.host_memory_cache[key] = {"value": value, "tag": tag}
        # Also persist to disk for more robust 