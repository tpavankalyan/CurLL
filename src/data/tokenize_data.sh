python /datadrive/pavan/lm/gpt-neox/tools/datasets/preprocess_data_with_chat_template.py \
            --input "/datadrive/pavan/CurLL/data/age_0_5/chat_template_neo_x/conv_chat_format_neox.jsonl","/datadrive/pavan/CurLL/data/age_0_5/chat_template_neo_x/poem_chat_format_neox.jsonl","/datadrive/pavan/CurLL/data/age_0_5/chat_template_neo_x/story_chat_format_neox.jsonl"\
            --jsonl-keys chat \
            --no-mask \
            --output /datadrive/pavan/CurLL/data/age_0_5/tokenized/ \
            --tokenizer-path "deepseek-ai/DeepSeek-V3-0324" \
            --dataset-impl mmap