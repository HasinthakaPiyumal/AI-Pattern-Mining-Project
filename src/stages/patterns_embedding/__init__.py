from .embedding_generator import generate_embeddings,save_embeddings,pattern_combiner
from utils.json_utils import load_json_from_file
from config.settings import Config
import time,os
from tqdm import trange as t
import pandas as pd

def main():
    # patterns = load_json_from_file(Config().PATTERNS_FILE)
    patterns = pd.read_csv(Config().PATTERNS_FILE).to_dict(orient='records')
    skip_count = 3000
    all_embeddings = []
    if(Config().PATTERN_EMBEDDINGS_FILE and os.path.exists(Config().PATTERN_EMBEDDINGS_FILE)):
        if os.path.exists(Config().PATTERN_EMBEDDINGS_FILE+'.backup'):
            os.remove(Config().PATTERN_EMBEDDINGS_FILE+'.backup')
        os.rename(Config().PATTERN_EMBEDDINGS_FILE,Config().PATTERN_EMBEDDINGS_FILE+'.backup')
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
