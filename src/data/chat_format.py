from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets

# Load all datasets
# d0 = load_dataset("Pavankalyan/stage0_context_cleaned")["train"]
d1 = load_dataset("Pavankalyan/stage1_context_cleaned")["train"]
# i0 = load_dataset("Pavankalyan/stage0_instruct_cleaned")["train"]
i1 = load_dataset("Pavankalyan/stage1_instruct_cleaned")["train"]

# Token formatting functions
def add_ins_chat_tokens(example):
    return {
        "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"
    }

def add_con_chat_tokens(example):
    return {
        "text": f"<|user|>\n{example['output']}"
    }

# Apply the formatting
# d0 = d0.map(add_con_chat_tokens, remove_columns=d0.column_names)
d1 = d1.map(add_con_chat_tokens, remove_columns=d1.column_names)
# i0 = i0.map(add_ins_chat_tokens, remove_columns=i0.column_names)
i1 = i1.map(add_ins_chat_tokens, remove_columns=i1.column_names)

# Combine all into a single dataset
# combined_dataset = concatenate_datasets([d0, d1, i0, i1])
# combined_dataset = concatenate_datasets([d0, i0])
combined_dataset = concatenate_datasets([d1, i1])


dataset = DatasetDict({"train": combined_dataset})

# Push to the Hugging Face Hub
dataset_name = "Pavankalyan/stages_1_mix"
dataset.push_to_hub(dataset_name)
print("Dataset uploaded to Hugging Face.")
