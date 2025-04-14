import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def load_jsonl_indent(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.loads(f)
    return data