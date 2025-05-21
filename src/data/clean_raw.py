import json
import re
from tqdm import tqdm
import unicodedata
from ftfy import fix_text as ft

def load_data(metadata_path, seed=False):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    num_files = len(metadata)
    data = []
    for i in tqdm(range(num_files)):
        file_name = f"chunk_{i}.jsonl"
        chunk_path = "/".join(metadata_path.split("/")[:-1] + [file_name]).replace("seed", "raw")
        # print(chunk_path)
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
            # print(len(chunk_data))
        data.extend(chunk_data)
    print("Total samples: ",len(data))
    return data

def output_dict_check(data_list):
    good_list = [o for o in data_list if isinstance(o['output'], dict)]
    bad_list = [o for o in data_list if not isinstance(o['output'], dict)]
    print("Total samples: ",len(data_list))
    print("Good samples: ",len(good_list))
    print("Bad samples: ",len(bad_list))
    return good_list, bad_list

def extract_json_from_string(text):
    # Use regex to extract the JSON block between ```json and ```
    match = re.search(r"```json\n({.*?})\n```", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON block found in the input string.")
    
    json_str = match.group(1)
    json_str = json_str.replace("{{", "{")
    json_str = json_str.replace("}}", "}")

    # Parse the JSON string
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def validate_instruction_response(dict_instance, output_format):
    required_keys = {}
    for k in output_format:
        required_keys[k] = type(output_format[k])
    for key, expected_type in required_keys.items():
        if key not in dict_instance:
            return False
    return True

def check_json_struct(data_list, prompt_path):
    with open(prompt_path, "r") as f:
        prompt = json.load(f)
    output_format = extract_json_from_string(prompt['user'])
    print(output_format)
    good_list = [o for o in data_list if validate_instruction_response(o['output'], output_format)]
    bad_list = [o for o in data_list if not validate_instruction_response(o['output'], output_format)]
    print("Total samples: ",len(data_list))
    print("Good samples: ",len(good_list))
    print("Bad samples: ",len(bad_list))
    return good_list, bad_list

def extract_key_values(s, keys):
    entry = {}
    for i, key in enumerate(keys):
        # Regex pattern: look for "key": and capture everything until the next key or end
        pattern = rf'"{key}"\s*:\s*(.*?)\s*(?="{keys[i+1]}"\s*:|$)' if i + 1 < len(keys) else rf'"{key}"\s*:\s*(.*)'
        match = re.search(pattern, s, re.DOTALL)
        if match:
            value = match.group(1).strip().strip('",')
            entry[key] = value.strip("```").strip("\n").strip("}").strip("\n")
        else:
            entry[key] = s
            # print(s)
            # print("-----------------------------------------------")
    return entry

def fix_text(bad_list, prompt_path):
    with open(prompt_path, "r") as f:
        prompt = json.load(f)
    output_format = extract_json_from_string(prompt['user'])
    for i in range(len(bad_list)):
        bad_list[i]['output'] = extract_key_values(bad_list[i]['output'], list(output_format.keys()))
    return bad_list

def fix_keys(bad_list):
    for i in range(len(bad_list)):
        output_new = {}
        for k in bad_list[i]['output'].keys():
            key_name = k.lower()
            if ("expanded" in key_name or "topic" in key_name) and bad_list[i]['output'][k] is not None:
                output_new["expanded_topic"] = bad_list[i]['output'][k]
            if ("generated" in key_name or "text" in key_name) and bad_list[i]['output'][k] is not None:
                output_new["generated_text"] = bad_list[i]['output'][k]
        bad_list[i]['output'] = output_new
    return bad_list

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
    text = ft(text) 
    return text

def clean_text_list(text_list):
    for i in tqdm(range(len(text_list))):
        for k in text_list[i]['output'].keys():
            if isinstance(text_list[i]['output'][k], str):
                text_list[i]['output'][k] = clean_text(text_list[i]['output'][k])
    return text_list

def hf_format(data_list, text_type="context"):
    if text_type=="context":
        for i in range(len(data_list)):
            data_list[i]['topic'] = data_list[i]['output']['expanded_topic']
            data_list[i]['output'] = data_list[i]['output']['generated_text']
            data_list[i]['POS'] = data_list[i]['word_list'][1]
            data_list[i]['word_list'] = data_list[i]['word_list'][0]
    elif text_type=="instruct":
        for i in range(len(data_list)):
            data_list[i]['instruction'] = data_list[i]['output']['instruction']
            data_list[i]['response'] = data_list[i]['output']['response']
            data_list[i]['POS'] = data_list[i]['word_list'][1]
            data_list[i]['word_list'] = data_list[i]['word_list'][0]
    return data_list

stage = 1
text_type = "instruct"  # or "context"
metadata_path = f"/datadrive/pavan/az_storage/data_unorganized/stages/stage{stage}/seed/{text_type}/metadata_chunks.jsonl"
prompt_path = f"/datadrive/pavan/az_storage/data_unorganized/stages/stage{stage}/seed/{text_type}/prompt_v1.json"
save_path = f"/datadrive/pavan/az_storage/data_unorganized/stages/stage{stage}/raw/{text_type}/all_data.jsonl"

data = load_data(metadata_path)
print("Total number of text snippets: ", len(data))

g0, b0 = output_dict_check(data)

print(b0[0])

g1, b1 = check_json_struct(g0, prompt_path)

print("Total number of bad samples: ", len(b0))

b0_fixed = fix_text(b0, prompt_path)

b1_fixed = fix_keys(b1)

all_samples = g1 + b0_fixed + b1_fixed

all_correct, all_incorrect = output_dict_check(all_samples)
all_correct, all_incorrect = check_json_struct(all_correct, prompt_path)

if len(all_incorrect) > 0:
    print("There are still some incorrect samples after fixing.")
    exit()


cleaned_samples = clean_text_list(all_samples)

with open(save_path, "w") as f:
    json.dump(all_samples, f, indent=4)

#upload to huggingface
from datasets import DatasetDict, Dataset
dataset = DatasetDict()
cleaned_samples = hf_format(cleaned_samples, text_type="instruct")
dataset['train'] = Dataset.from_list(cleaned_samples)
dataset_name = f"Pavankalyan/stage{stage}_instruct_cleaned"  # Replace with your info
dataset.push_to_hub(dataset_name)
print("Dataset uploaded to Hugging Face.")













