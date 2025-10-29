import os
from datetime import datetime
import threading

class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, tag="", run_output=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, tag="",run_output=None):
        if getattr(self, "_initialized", False):
            return

        self.TAG = tag
        self.ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.OUTPUT_BASE = os.path.join(self.ROOT, "outputs")
        self.RUN_OUTPUT = os.path.join(
            self.OUTPUT_BASE, tag, run_output or datetime.now().strftime("%Y%m%d_%H%M%S - Run ")
        )
        os.makedirs(self.RUN_OUTPUT, exist_ok=True)

        self.PAPER_FOLDER = os.path.join(self.ROOT, "data/raw/papers", self.TAG)
        self.CLEANED_FOLDER = os.path.join(self.ROOT, "data/cleaned/papers", self.TAG)
        self.PATTERNS_FOLDER = os.path.join(self.RUN_OUTPUT, "extracted_patterns")
        self.PATTERNS_FILE = os.path.join(self.PATTERNS_FOLDER, "all_patterns.json")
        self.PATTERN_EMBEDDINGS_FILE = os.path.join(self.RUN_OUTPUT, "pattern_embeddings.csv")

        self.CLEANED = False
        self._initialized = True

    def change_tag(self, new_tag):
        """Reset the singleton with a new tag."""
        type(self)._instance = None
        return Config(tag=new_tag)
