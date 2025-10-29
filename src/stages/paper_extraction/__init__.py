from __future__ import annotations

import os

from tqdm import tqdm

from config.settings import Config
from utils.file_utils import get_all_text_files
from utils.json_utils import save_json_to_file

from .paper_cleaner import clean_all_pdfs
from .pattern_extractor import extract_patterns as _extract

__all__ = ["extract_patterns", "clean_all_pdfs"]

def _save_patterns(patterns, all_patterns, file_path):
    base_name = os.path.basename(file_path).replace("cleaned_", "").replace(".txt", "_patterns.json")
    output_path = os.path.join(Config().PATTERNS_FOLDER, base_name)
    save_json_to_file(patterns, output_path)
    save_json_to_file(all_patterns, Config().PATTERNS_FILE)


def extract_patterns(file_list):
    all_patterns = []
    for file_path in tqdm(file_list, desc="Extracting Patterns", ncols=80):
        patterns = _extract(file_path)
        all_patterns.extend(patterns)
        _save_patterns(patterns, all_patterns, file_path)
    return all_patterns

if __name__ == "__main__":
    cleaned_files = clean_all_pdfs(Config().PAPER_FOLDER, Config().CLEANED_FOLDER) # Uncomment if cleaning is needed
    # cleaned_files = get_all_text_files()
    # for i, file in enumerate(cleaned_files):
    #     print(f"File {i}: {file}")
    extract_patterns(cleaned_files)