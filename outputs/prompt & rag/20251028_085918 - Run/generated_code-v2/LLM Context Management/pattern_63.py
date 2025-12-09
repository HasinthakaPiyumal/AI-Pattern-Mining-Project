import faiss
import os
import pickle
from news_indexer import NewsIndexer

class IndexManager:
    def __init__(self, index_filepath_prefix: str, news_indexer: NewsIndexer):
        self.index_filepath_prefix = index_filepath_prefix
        self.news_indexer = news_indexer
        self.current_index = None
        self.current_id_map = {}
        self.current_text_map = {}
        self._load_active_index()

    def _load_active_index(self):
        """Loads the initially active index from disk."""
        print(f"Attempting to load active index from {self.index_filepath_prefix}_active")
        index, id_map, text_map = self.news_indexer.load_index_data(f"{self.index_filepath_prefix}_active")
        if index is not None:
            self.current_index = index
            self.current_id_map = id_map
            self.current_text_map = text_map
            print("Active index loaded successfully.")
        else:
            print("No active index found. Starting with an empty index.")

    def hotswap_index(self, new_articles: list[dict]):
        """Builds a new index from updated articles and hotswaps it with the current one."""
        print("Initiating index hotswap...")
        # 1. Build a new index
        new_faiss_index, new_id_map, new_text_map = self.news_indexer.build_index(new_articles)

        if new_faiss_index is None:
            print("No new articles to build an index. Hotswap aborted.")
            return

        # 2. Save the new index (potentially as a 'staging' index first)
        new_filepath = f"{self.index_filepath_prefix}_new"
        self.news_indexer.save_index_data(new_faiss_index, new_id_map, new_text_map, new_filepath)

        # 3. Atomically replace the active index (simulated by renaming)
        # In a real-world scenario, this might involve more robust mechanisms (e.g., symlinks, distributed file systems)
        active_filepath = f"{self.index_filepath_prefix}_active"
        temp_filepath = f"{self.index_filepath_prefix}_temp_old"

        # Remove old temp files if they exist from a previous failed swap
        if os.path.exists(f"{temp_filepath}.faiss"): os.remove(f"{temp_filepath}.faiss")
        if os.path.exists(f"{temp_filepath}.meta"): os.remove(f"{temp_filepath}.meta")

        # Move current active to temp_old, then new to active
        if os.path.exists(f"{active_filepath}.faiss"):
            os.rename(f"{active_filepath}.faiss", f"{temp_filepath}.faiss")
        if os.path.exists(f"{active_filepath}.meta"):
            os.rename(f"{active_filepath}.meta", f"{temp_filepath}.meta")

        os.rename(f"{new_filepath}.faiss", f"{active_filepath}.faiss")
        os.rename(f"{new_filepath}.meta", f"{active_filepath}.meta")

        # Load the newly active index
        self.current_index = new_faiss_index
        self.current_id_map = new_id_map
        self.current_text_map = new_text_map

        print("Index hotswap completed successfully. New index is now active.")

        # Clean up old temp files
        if os.path.exists(f"{temp_filepath}.faiss"): os.remove(f"{temp_filepath}.faiss")
        if os.path.exists(f"{temp_filepath}.meta"): os.remove(f"{temp_filepath}.meta")

    def get_current_index_data(self) -> tuple[faiss.IndexFlatIP, dict, dict]:
        """Returns the currently active FAISS index and its associated mappings."""
        return self.current_index, self.current_id_map, self.current_text_map