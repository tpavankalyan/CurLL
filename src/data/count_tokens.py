from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Load the dataset and tokenizer
dataset = load_dataset("Pavankalyan/stage0_context_cleaned", split="train")
# tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer = AutoTokenizer.from_pretrained("Pavankalyan/TinyStoriesInstruct-tokenizer")

#print vocab size
print(f"Tokenizer vocab size: {len(tokenizer)}")

#print special tokens and ids
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
# Print special tokens and their corresponding IDs
for token_name, token in tokenizer.special_tokens_map.items():
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token_name}: {token} -> {token_id}")


# Tokenize and count tokens
def count_tokens(batch):
    texts = [text + "<|endoftext|>" for text in batch["output"]]
    tokens = tokenizer(texts, return_attention_mask=False, return_token_type_ids=False)
    return {"num_tokens": [len(input_ids) for input_ids in tokens["input_ids"]]}

# Apply token counting
tokenized_dataset = dataset.map(count_tokens, batched=True, batch_size=1000)

# Sum total tokens
total_tokens = sum(tokenized_dataset["num_tokens"])
print(f"Total tokens: {total_tokens}")
