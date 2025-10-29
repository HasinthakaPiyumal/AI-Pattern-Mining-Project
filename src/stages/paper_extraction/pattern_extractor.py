import os,time,json
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from .prompts import optimized_prompt,retry_prompt,summary_prompt
from utils.json_utils import parse_json_safe
from utils.file_utils import read_text_file as read_
from utils.logger import get_logger
from tqdm import tqdm

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

logger = get_logger(__name__)

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
    return parse_json_safe(iter_3.content)

def extract_patterns(file_path):
    text = read_(file_path)
    patterns = extract_patterns_from_text(text)
    return patterns

def summarize_patterns(patterns):
    prompt = PromptTemplate(
        template=summary_prompt,
        input_variables=["patterns_text"]
    )

    chain = prompt | llm
    summary = chain.invoke({"patterns_text": patterns})
    return parse_json_safe(summary.content)