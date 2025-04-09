from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset and tokenizer
dataset = load_dataset("Pavankalyan/age_0_5_text", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and count tokens
def count_tokens(batch):
    tokens = tokenizer(batch["text"], return_attention_mask=False, return_token_type_ids=False)
    return {"num_tokens": [len(input_ids) for input_ids in tokens["input_ids"]]}

# Apply token counting
tokenized_dataset = dataset.map(count_tokens, batched=True, batch_size=1000)

# Sum total tokens
total_tokens = sum(tokenized_dataset["num_tokens"])
print(f"Total tokens: {total_tokens}")
