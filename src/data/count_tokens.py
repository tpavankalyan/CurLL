from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Load the dataset and tokenizer
dataset = load_dataset("roneneldan/TinyStories", split="validation")
# tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
tokenizer = AutoTokenizer.from_pretrained("Pavankalyan/TinyStoriesInstruct-tokenizer")

#load the model and print the number of parameters
# model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
# Print the number of parameters in the model
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters in the model: {num_params}")
# print non embedded parameters
# non_embedded_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.dim() > 1)
# print(f"Number of non-embedded parameters in the model: {non_embedded_params}")
# Print the model architecture
# print(model)

#print number of hidden layers
# num_hidden_layers = model.config.num_hidden_layers
# print(f"Number of hidden layers in the model: {num_hidden_layers}")
# print number of attention heads
# num_attention_heads = model.config.num_attention_heads
# print(f"Number of attention heads in the model: {num_attention_heads}")



#print vocab size
print(f"Tokenizer vocab size: {len(tokenizer)}")

#print special tokens and ids
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
# Print special tokens and their corresponding IDs
for token_name, token in tokenizer.special_tokens_map.items():
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token_name}: {token} -> {token_id}")
exit()

# Tokenize and count tokens
def count_tokens(batch):
    tokens = tokenizer(batch["text"], return_attention_mask=False, return_token_type_ids=False)
    return {"num_tokens": [len(input_ids) for input_ids in tokens["input_ids"]]}

# Apply token counting
tokenized_dataset = dataset.map(count_tokens, batched=True, batch_size=1000)

# Sum total tokens
total_tokens = sum(tokenized_dataset["num_tokens"])
print(f"Total tokens: {total_tokens}")
