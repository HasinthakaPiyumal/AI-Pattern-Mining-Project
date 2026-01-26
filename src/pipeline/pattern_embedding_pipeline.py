from config.settings import Config
Config(tag="all v2",run_output="20260126_014938 - Run ")

from stages.patterns_embedding import main as embed_patterns

embed_patterns()