import json,os

def parse_json_safe(text,delimiter='[]'):
    start, end = text.find(delimiter[0]), text.rfind(delimiter[1]) + 1
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end])
        except:
            return [] if delimiter == '[]' else {}
    return [] if delimiter == '[]' else {}

def save_json_to_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def load_json_from_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)