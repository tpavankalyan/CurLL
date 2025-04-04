import json
from tqdm import tqdm
import unicodedata
import re
from ftfy import fix_text
from datasets import Dataset, DatasetDict
import random

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u201c": '"', "\u201d": '"',  # Fancy quotes → Regular quotes
        "\u2018": "'", "\u2019": "'",  # Fancy apostrophes → Regular apostrophes
        "\u2013": "-", "\u2014": "-",  # En dash & Em dash → Hyphen
        "\u2026": "...",  # Ellipsis → Three dots
        "\u00A0": " "  # Non-breaking space → Regular space
    }  
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = fix_text(text) 
    return text

base_path = "/datadrive/pavan/ContinualLearning/SkillData/data/age_0_5/questions/"
text_types = ["conv", "story", "poem"]

texts = []
for text_type in text_types:
    file_path = base_path + f"{text_type}/" + "parsed.json"
    with open(file_path, "r") as f:
        l = json.load(f)

    if text_type == "conv":
        tt = "conversation"
    else:
        tt = text_type

    for i in tqdm(range(len(l))):
        conversation = ""
        for in_q,q in enumerate(l[i]["output"]["ir"]):
            if in_q == 0:
                conversation = conversation + f"Here is a {tt}:\n\n{clean_text(l[i]["content"])}\n\nAnswer some questions based on the context above:\n\nQuestion: {clean_text(q["instruction"])}\n"
            else:
                conversation = conversation + f"Question: {clean_text(q["instruction"])}\n"
            conversation = conversation + f"Answer: {clean_text(q["response"])}\n\n"
        texts.append(conversation)


random.seed(42)
random.shuffle(texts)

n_total = len(texts)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_texts = texts[:n_train]
val_texts = texts[n_train:n_train + n_val]
test_texts = texts[n_train + n_val:]

# --- Create DatasetDict ---
dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train_texts}),
    "val": Dataset.from_dict({"text": val_texts}),
    "test": Dataset.from_dict({"text": test_texts}),
})

# --- Push to Hugging Face ---
dataset_name = "Pavankalyan/age_0_5_text"  # Replace with your info

# Upload to the hub
dataset.push_to_hub(dataset_name)
