from datasets import load_dataset, concatenate_datasets, DatasetDict

# Load datasets
ds = load_dataset("Pavankalyan/stage0_instruct_simple_split")
dr = load_dataset("Pavankalyan/stage0_context_cleaned")
print("Dataset loaded successfully.")

# Create new text-only examples
def merge_instructions(row):
    return {"text": "<|user|> " + row["instruction"] + "\n<|assistant|> " + row["response"]}

def prepare_context(row):
    return {"text": "<|user|> " + row["output"]}

# Apply transformations, keeping only 'text'
ds_text = ds['train'].map(merge_instructions, remove_columns=ds['train'].column_names)
dr_text = dr['train'].map(prepare_context, remove_columns=dr['train'].column_names)

# Combine datasets
combined_text = concatenate_datasets([ds_text, dr_text])

# Wrap in a new DatasetDict with just a 'train' split
final_dataset = DatasetDict({
    "train": combined_text
})

# Push to a new Hugging Face dataset
final_dataset.push_to_hub("Pavankalyan/stage0_all_text_only")
print("New dataset pushed successfully.")
