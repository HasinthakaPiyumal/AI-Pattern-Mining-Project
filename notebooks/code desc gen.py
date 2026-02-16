import pandas as pd
import os
from langchain.chat_models import init_chat_model
from time import time
from dotenv import load_dotenv
load_dotenv()

labeled_data = pd.read_csv('/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/data/datasets/labeled_data.csv')
labeled_data = labeled_data[labeled_data['verified_pattern'].notna()]
labeled_data['path'] = labeled_data['file'].apply(lambda x: f'/home/hasinthaka/Documents/Projects/AI/Pattern Mining/pipeline/notebooks/result/repo_callgraph_clusters/{x.split("/")[-2]}/{x.split("/")[-1]}')

def load_code(path):
    with open(path, 'r') as file:
        return file.read()
labeled_data['code'] = labeled_data['path'].apply(load_code)

gem = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.7)
gpt = init_chat_model("gpt-5-nano", model_provider="openai", temperature=0.7)

prompt = """\
You are an expert code description generator. Generate a concise and informative description of the community in 2-3 sentences. Focus on the code AI patterns. Strictly generate the description about patterns represented in the code. Avoid generic statements and ensure the description is specific to a AI patterns.

code:
{code}
"""

def generate_community_description(code: str,llm,max_retries=3) -> str:
    for _ in range(max_retries):
        try:
            print(f"--------- Generating code summary",end="\r")
            response = llm.invoke(prompt.format(code=code))
            print(f"Generated --------------",end="\r")
            return response.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

from concurrent.futures import ThreadPoolExecutor

def generate_code_summaries(code: str) -> str:
    def task(llm):
        return generate_community_description(code, llm)

    print("Generating code summaries (6 in parallel)...", end="\r")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(task, gem),  # 01
            executor.submit(task, gpt),  # 02
            executor.submit(task, gem),  # 03
            executor.submit(task, gpt),  # 04
            executor.submit(task, gem),  # 05
            executor.submit(task, gpt),  # 06
        ]
        results = [f.result() for f in futures]
    return tuple(results)


from torch._dynamo.pgo import code_state_path


description_file = "result/community_description/feb-10-2026-community-descriptions.csv"
if os.path.exists(description_file):
    descriptions = pd.read_csv(description_file)
else:
    descriptions = pd.DataFrame(columns=['file', 'code','verified_pattern','code_summary_01','code_summary_02','code_summary_03','code_summary_04','code_summary_05','code_summary_06'])


def main(file,code,verified_pattern):
    print(f"{len(descriptions)} - Generating code summary for {file}")
    if file in descriptions['file'].values:
        return 0
    description = generate_code_summaries(code)
    new_dict = {}
    new_dict['code_summary_01'] = description[0]
    new_dict['code_summary_02'] = description[1]
    new_dict['code_summary_03'] = description[2]
    new_dict['code_summary_04'] = description[3]
    new_dict['code_summary_05'] = description[4]
    new_dict['code_summary_06'] = description[5]
    new_dict['verified_pattern'] = verified_pattern
    new_dict['file'] = file
    new_dict['code'] = code
    descriptions.loc[len(descriptions)] = new_dict
    descriptions.to_csv(description_file,index=False)

for index,row in labeled_data.iterrows():
    main(row['file'],row['code'],row['verified_pattern'])
