import json
from tqdm import tqdm
from datasets import DatasetDict, Dataset

with open("/datadrive/pavan/az_storage/data_unorganized/stages/stage0/raw/context/all_data.jsonl", "r") as f:
    data = json.load(f)

print("Total number of text snippets: ",len(data))

#flatten the output dictinonary
for k in tqdm(data):
    k['topic'] = k['output']['expanded_topic']
    k['output'] = k['output']['generated_text']
    k['POS'] = k['word_list'][1]
    k['word_list'] = k['word_list'][0]

# Create a DatasetDict
dataset = DatasetDict()
dataset['train'] = Dataset.from_list(data)

#print the first 2 samples
# print("First 2 samples: ", dataset['all'][:2])

# --- Push to Hugging Face ---
dataset_name = "Pavankalyan/stage0_context_raw"  # Replace with your info

# Upload to the hub
dataset.push_to_hub(dataset_name)