import sys
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.nn import Embedding

def count_parameters(model):
    embedding_params = 0
    non_embedding_params = 0

    for name, module in model.named_modules():
        if isinstance(module, Embedding):
            for param in module.parameters():
                embedding_params += param.numel()

    all_params = sum(p.numel() for p in model.parameters())
    non_embedding_params = all_params - embedding_params
    return embedding_params, non_embedding_params, all_params

def show_tokenizer_info(tokenizer):
    print("\n🔤 Tokenizer Info")
    print("-----------------")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    special_tokens = {
        "pad_token": tokenizer.pad_token,
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.unk_token,
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "mask_token": tokenizer.mask_token,
    }

    print("Special tokens:")
    for k, v in special_tokens.items():
        print(f"  {k:<12}: {repr(v)}")

def show_model_info(config, model):
    print("\n⚙️ Model Config Info")
    print("--------------------")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Architectures: {config.architectures}")
    print(f"Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"Num hidden layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"Num attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
    print(f"Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"Tied embeddings: {getattr(config, 'tie_word_embeddings', 'N/A')}")
    print(f"Is encoder: {getattr(config, 'is_encoder_decoder', False)}")
    print(f"Is decoder-only: {getattr(config, 'is_decoder', False)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"\n🔍 Loading model and tokenizer for '{model_name}'...\n")

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embedding, non_embedding, total = count_parameters(model)

    print("🔢 Parameter Breakdown")
    print("----------------------")
    print(f"Embedding parameters:     {embedding:,}")
    print(f"Non-embedding parameters: {non_embedding:,}")
    print(f"Total parameters:         {total:,}")

    show_model_info(config, model)
    show_tokenizer_info(tokenizer)
