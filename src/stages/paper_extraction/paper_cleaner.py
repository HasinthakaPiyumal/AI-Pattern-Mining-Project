
from utils.file_utils import clean_and_save_file,get_all_pdf_files,get_all_text_files
import os,tqdm
from utils.logger import get_logger


logger = get_logger(__name__)

def clean_all_pdfs(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for pdf_path in tqdm.tqdm(get_all_pdf_files(folder_path), desc="Cleaning PDFs",ncols=80):
        filename = os.path.basename(pdf_path)
        output_path = os.path.join(output_folder, f"cleaned_{filename}.txt")
        clean_and_save_file(pdf_path, output_path)
    return get_all_text_files(output_folder)

if __name__ == "__main__":
    # paper_folder = "papers"
    # output_folder = "cleaned_papers"
    # clean_all_pdfs_in_folder(paper_folder, output_folder)
    file_path = "papers/Software-Engineering_Design_Patterns_for_Machine_Learning_Applications.pdf"
    print('Cleaning file:', file_path)
    output_path = "cleaned_papers/cleaned_Software-Engineering_Design_Patterns_for_Machine_Learning_Applications.pdf.txt"
    clean_all_pdfs(file_path, output_path)