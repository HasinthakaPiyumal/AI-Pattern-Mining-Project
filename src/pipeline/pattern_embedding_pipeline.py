from config.settings import Config
Config(tag="prompt & rag",run_output="20251028_085918 - Run ")

from stages.patterns_embedding import main as embed_patterns

embed_patterns()