from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Check and print EOS token
if tokenizer.eos_token:
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
else:
    print("EOS token does NOT exist.")

# Check and print BOS token
if tokenizer.bos_token:
    print(f"BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
else:
    print("BOS token does NOT exist.")

# Print vocabulary size
print(f"Vocabulary size: {tokenizer.vocab_size}")
