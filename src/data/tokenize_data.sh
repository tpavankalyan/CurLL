python3 /datadrive/pavan/CurLL/nanotron/tools/preprocess_data.py --tokenizer-name-or-path HuggingFaceTB/SmolLM-360M --output-folder /datadrive/pavan/CurLL/data/age_0_5/age_0_5_text --n-tasks 16 hf --dataset Pavankalyan/age_0_5_text


# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 /datadrive/pavan/CurLL/nanotron/run_train.py --config-file /datadrive/pavan/CurLL/nanotron/examples/config_tiny_llama.yaml > output.txt
