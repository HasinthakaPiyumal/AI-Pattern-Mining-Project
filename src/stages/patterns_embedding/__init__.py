from .embedding_generator import generate_embeddings,save_embeddings,pattern_combiner
from utils.json_utils import load_json_from_file
from config.settings import Config
import time,os
from tqdm import trange as t

def main():
    patterns = load_json_from_file(Config().PATTERNS_FILE)
    skip_count = 50
    all_embeddings = []
    for i in t((len(patterns)//skip_count)+1, desc="Generating Embeddings", ncols=80):
        start_index = i * skip_count
        end_index = min(start_index + skip_count, len(patterns))
        combined_patterns = pattern_combiner(patterns[start_index:end_index])
        embeddings = generate_embeddings(combined_patterns)
        all_embeddings.extend(embeddings)
        if end_index != len(patterns):
            print()
            for _ in range(60):
                print(f'Waiting... {60-_} seconds remaining.', end='\r')
                time.sleep(1)
            os.system('clear')
    save_embeddings(all_embeddings,patterns)
