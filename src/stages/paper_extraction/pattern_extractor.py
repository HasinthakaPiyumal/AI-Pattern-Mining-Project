import os,time,json
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from .prompts import optimized_prompt,retry_prompt,summary_prompt
from config.settings import Config
from utils.json_utils import parse_json_safe, load_json_from_file, save_json_to_file
from utils.file_utils import read_text_file as read_
from utils.logger import get_logger
from tqdm import tqdm

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

logger = get_logger(__name__)

GEMINI_MAX_CHARS = int(os.getenv("GEMINI_MAX_CHARS", "24000"))  # approx 6k tokens, safe for gemini-2.5-flash
CHUNK_SIZE = 80000 #GEMINI_MAX_CHARS
CHUNK_OVERLAP = 800


def _chunk_cache_path(file_path, chunk_idx):
    base = os.path.basename(file_path).replace("cleaned_", "").replace(".txt", "")
    cache_dir = os.path.join(Config().PATTERNS_FOLDER, "chunk_cache", base)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"chunk_{chunk_idx}.json")


def _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks

def extract_patterns_from_text(text):
    stages = ["Extraction", "Embedding", "Validation"]
    pbar = tqdm(total=len(stages), desc="Pipeline Progress", ncols=80)

    prompt = PromptTemplate(
        template=optimized_prompt,
        input_variables=["text"]
    )

    prompt_2 = PromptTemplate(
        template=retry_prompt,
        input_variables=["text", "extracted_patterns"]
    )
    
    pbar.set_description("Stage 1: Pattern Extraction")
    iter_1 = llm.invoke(prompt.format(text=text))
    pbar.update(1)

    pbar.set_description("Stage 2: Pattern Refinement I")
    iter_2 = llm.invoke(prompt_2.format(text=text, extracted_patterns=iter_1.content))
    pbar.update(1)

    pbar.set_description("Stage 3: Pattern Refinement II")
    iter_3 = llm.invoke(prompt_2.format(text=text, extracted_patterns=iter_2.content))
    pbar.update(1)

    pbar.close()
    return parse_json_safe(iter_2.content)

def extract_patterns(file_path):
    text = read_(file_path)
    chunks = _chunk_text(text)

    all_patterns = []
    for idx, chunk in enumerate(chunks, start=1):
        cache_path = _chunk_cache_path(file_path, idx)

        if os.path.exists(cache_path):
            try:
                cached = load_json_from_file(cache_path)
                logger.info("Using cached chunk %d/%d for %s", idx, len(chunks), file_path)
                all_patterns.extend(cached)
                continue
            except Exception as exc:
                logger.warning("Failed to load cached chunk from %s (%s); re-extracting", cache_path, exc)

        logger.info("Extracting chunk %d/%d for %s", idx, len(chunks), file_path)
        patterns_chunk = extract_patterns_from_text(chunk)
        save_json_to_file(patterns_chunk, cache_path)
        all_patterns.extend(patterns_chunk)
    return all_patterns

def summarize_patterns(patterns):
    prompt = PromptTemplate(
        template=summary_prompt,
        input_variables=["patterns_text"]
    )

    chain = prompt | llm
    summary = chain.invoke({"patterns_text": patterns})
    return parse_json_safe(summary.content)