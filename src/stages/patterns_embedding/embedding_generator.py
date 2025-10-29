from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd,os,json,time
from utils.json_utils import load_json_from_file
from config.settings import Config
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def generate_embeddings(texts):
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(texts)
    return embeddings

def pattern_combiner(patterns):
    combined_patterns = []

    for pattern in patterns:
        pattern_text = f"<PatternName>{pattern.get('Pattern Name', 'Unnamed Pattern')}</PatternName>\n"
        pattern_text += f"<Problem>{pattern.get('Problem', '')}</Problem>\n"
        pattern_text += f"<Context>{pattern.get('Context', '')}</Context>\n"
        pattern_text += f"<Solution>{pattern.get('Solution', '')}</Solution>\n"
        pattern_text += f"<Result>{str(pattern.get('Result', ''))}</Result>\n"
        pattern_text += f"<Uses>\n"
        pattern_text += f"{str(pattern.get('Uses', []))}\n"
        pattern_text += f"</Uses>\n"
        combined_patterns.append(pattern_text)
    return combined_patterns

def save_embeddings(embeddings,patterns):
    file_path = Config().PATTERN_EMBEDDINGS_FILE
    if os.path.exists(file_path):   
        df1 = pd.read_csv(file_path)
        df2 = pd.DataFrame(embeddings)
        df2['Pattern Name'] = [pattern.get('Pattern Name', 'Unnamed Pattern') for pattern in patterns]
        df2['Problem'] = [pattern.get('Problem', '') for pattern in patterns]
        df2['Context'] = [pattern.get('Context', '') for pattern in patterns]
        df2['Solution'] = [pattern.get('Solution', '') for pattern in patterns]
        df2['Result'] = [str(pattern.get('Result', '')) for pattern in patterns]
        df2['Uses'] = [str(pattern.get('Uses', [])) for pattern in patterns]
        df = pd.concat([df1, df2], ignore_index=True)
        df.to_csv(file_path, index=False)
    else:
        df = pd.DataFrame(embeddings)
        df['Pattern Name'] = [pattern.get('Pattern Name', 'Unnamed Pattern') for pattern in patterns]
        df['Problem'] = [pattern.get('Problem', '') for pattern in patterns]
        df['Context'] = [pattern.get('Context', '') for pattern in patterns]
        df['Solution'] = [pattern.get('Solution', '') for pattern in patterns]
        df['Result'] = [str(pattern.get('Result', '')) for pattern in patterns]
        df['Uses'] = [str(pattern.get('Uses', [])) for pattern in patterns]
        df.to_csv(file_path, index=False)
