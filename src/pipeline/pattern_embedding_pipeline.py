from config.settings import Config
Config(tag="microservices",run_output="ms-v1")

from stages.patterns_embedding import main as embed_patterns

embed_patterns()