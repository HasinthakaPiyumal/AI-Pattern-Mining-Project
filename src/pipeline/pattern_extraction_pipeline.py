from stages.paper_extraction import extract_patterns as extract_,clean_all_pdfs
from utils.file_utils import get_all_text_files
from config import settings

co = settings.Config(tag="all v2",run_output="20260126_014938 - Run ")

# Clean the PDFs if not already cleaned
if co.CLEANED == False:        
    cleaned_files = clean_all_pdfs(co.PAPER_FOLDER,co.CLEANED_FOLDER)
else:
    cleaned_files = get_all_text_files(co.CLEANED_FOLDER)

# Extract patterns from cleaned files
extract_(cleaned_files)