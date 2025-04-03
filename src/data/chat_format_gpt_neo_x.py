import json
from tqdm import tqdm
import unicodedata
import re
from ftfy import fix_text

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
output_base_path = "/datadrive/pavan/CurLL/data/age_0_5/"

for text_type in text_types:
    output_path = output_base_path + f"{text_type}_" + "chat_format_neox.jsonl"
    file_path = base_path + f"{text_type}/" + "parsed.json"
    with open(file_path, "r") as f:
        l = json.load(f)

    if text_type == "conv":
        tt = "conversation"
    else:
        tt = text_type

    for i in tqdm(range(len(l))):
        conversation = {}
        conversation["chat"] = []
        for in_q,q in enumerate(l[i]["output"]["ir"]):
            if in_q == 0:
                conversation["chat"].append({
                    "role": "user",
                    "content": f"Here is a {tt}:\n\n" + 
                    clean_text(l[i]["content"]) + 
                    f"Answer some questions based on the context above:\n\n" + 
                    clean_text(q["instruction"])
                })
            else:
                conversation["chat"].append({
                    "role": "user",
                    "content": clean_text(q["instruction"])
                })
            conversation["chat"].append({
                "role": "assistant",
                "content": clean_text(q["response"])
            })

        with open(output_path, "a") as f:
            f.write(json.dumps(conversation) + "\n")

    print(f"Writing {text_type} to {output_path}")

