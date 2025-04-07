import os
import json
from datasets import Dataset, DatasetDict, load_dataset

def save_dataset_as_jsonl(dataset, output_dir="/datadrive/pavan/CurLL/data/age_0_5/texts/"):
    if isinstance(dataset, DatasetDict):
        for split_name, split_data in dataset.items():
            file_path = output_dir + f"{split_name}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {split_name} split to {file_path}")
    elif isinstance(dataset, Dataset):
        file_path = output_dir +  "data.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved dataset to {file_path}")
    else:
        raise TypeError("Input must be a Hugging Face Dataset or DatasetDict.")

dataset_name = "Pavankalyan/age_0_5_text"  # change this
dataset = load_dataset(dataset_name)  # returns a DatasetDict

# Save the dataset as JSONL files
save_dataset_as_jsonl(dataset)