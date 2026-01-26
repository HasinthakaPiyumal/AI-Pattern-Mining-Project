from __future__ import annotations

import os

from tqdm import tqdm

from config.settings import Config
from utils.file_utils import get_all_text_files
from utils.json_utils import load_json_from_file, save_json_to_file
from utils.logger import get_logger

from .paper_cleaner import clean_all_pdfs
from .pattern_extractor import extract_patterns as _extract

__all__ = ["extract_patterns", "clean_all_pdfs"]


logger = get_logger(__name__)


def _pattern_output_path(file_path: str) -> str:
    base_name = os.path.basename(file_path).replace("cleaned_", "").replace(".txt", "_patterns.json")
    return os.path.join(Config().PATTERNS_FOLDER, base_name)

def _save_patterns(patterns, all_patterns, file_path):
    output_path = _pattern_output_path(file_path)
    save_json_to_file(patterns, output_path)
    save_json_to_file(all_patterns, Config().PATTERNS_FILE)


def extract_patterns(file_list):
    all_patterns = []
    for file_path in tqdm(file_list, desc="Extracting Patterns", ncols=80):
        output_path = _pattern_output_path(file_path)
        if os.path.exists(output_path):
            try:
                patterns = load_json_from_file(output_path)
                logger.info("Skipping extraction; using existing patterns for %s", file_path)
            except Exception as exc:
                logger.warning("Failed to load existing patterns from %s. Re-extracting. (%s)", output_path, exc)
                patterns = _extract(file_path)
        else:
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