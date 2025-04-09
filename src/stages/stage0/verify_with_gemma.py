import json
import argparse
from pathlib import Path
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams
from vllm import LLM, SamplingParams
from huggingface_hub import login
from tqdm import tqdm
import re
import os

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def _process_story(seed, new_seed):
    output = seed["output"]
    if isinstance(output, dict):
        new_seed["content"] = output.get("story", "")
    else:
        new_seed["content"] = output


def _process_poem(seed, new_seed):
    output = seed["output"]
    if isinstance(output, dict):
        new_seed["content"] = output.get("content", "")
    else:
        new_seed["content"] = output


def _process_conversation(seed, new_seed):
    output = seed["output"]
    if isinstance(output, dict) and "dialogue" in output:
        dialogue_text = []
        for entry in output["dialogue"]:
            speaker = entry.get("speaker", "Unknown")
            line = entry.get("line", "")
            dialogue_text.append(f"{speaker}: {line}")
        new_seed["content"] = "\n".join(dialogue_text)
    else:
        new_seed["content"] = str(output)

def convert_to_string(data):
    """
    Convert a list of dictionaries into a formatted string.
    """
    result = []
    for item in data:
        instruction = item.get('instruction', '').strip()
        response = item.get('response', '').strip()
        result.append(f"Instruction: {instruction}\nResponse: {response}\n")
    return '\n'.join(result)

def get_seed_data(pre_seed_data, text_type, qa=False):
    if qa:
        for i, seed in enumerate(pre_seed_data):
            pre_seed_data[i]['ir'] = convert_to_string(seed['output']['ir'])
        return pre_seed_data
    else:
        seed_list = []
        valid_text_types = ['story', 'poem', 'conv']
        if text_type not in valid_text_types:
            raise ValueError(f"Invalid text_type. Must be one of {valid_text_types}")   
        for seed in pre_seed_data:
            try:
                required_fields = ['skill', 'sub_skill', 'goal', 'indicator', 'topic', 'output']
                for field in required_fields:
                    if field not in seed:
                        raise KeyError(f"Missing required field: {field}")
                new_seed = {
                    "skill": seed["skill"],
                    "sub_skill": seed["sub_skill"],
                    "goal": seed["goal"],
                    "indicator": seed["indicator"],
                    "topic": seed["topic"]
                }
                if text_type == "story":
                    _process_story(seed, new_seed)
                elif text_type == "poem":
                    _process_poem(seed, new_seed)
                elif text_type == "conv":
                    _process_conversation(seed, new_seed)
                
                seed_list.append(new_seed)
                
            except Exception as e:
                print(f"Error processing seed: {e}")
                continue
        return seed_list



def generate_texts(text_type, batch_size=1024, qa=False):
    login(token='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')
    prompts_path = "/datadrive/pavan/ContinualLearning/SkillData/metadata/prompts_gpt4_5.json"
    if qa:
        format_type = "questions"
    else:
        format_type = "texts"
    base_path = f"/datadrive/pavan/az_storage/data_unorganized/age_0_5/{format_type}/"

    seed_data = load_json(base_path + text_type + "/random_5000.json")
    seed_data = get_seed_data(seed_data, text_type, qa=qa)
    prompts = load_json(prompts_path)
    system_prompt = prompts[format_type][text_type]["system"]
    user_prompt = prompts[format_type][text_type]["user"]

    system_msg = {"role": "system", "content": system_prompt}
    conversations = [
        [system_msg, {"role": "user", "content": user_prompt.format(**seed)}]
        for seed in seed_data
    ]

    print("-"*50)
    print("Total instances to generate: ", len(conversations))

    sampling_params = SamplingParams(
        max_tokens=8192, temperature=0
    )
    model_name = "google/gemma-3-27b-it"
    llm = LLM(model=model_name, tensor_parallel_size=2, 
              enable_prefix_caching=True, gpu_memory_utilization=0.97,
              enable_chunked_prefill=True, max_num_batched_tokens=256,
              max_num_seqs=256, max_model_len=2048)
    
    output_path = base_path + text_type + "/random_5000_verified_gemma.json"
    total_tokens = 0
    c = 0
    
    for i in tqdm(range(0, len(conversations), batch_size)):
        batch_conversations = conversations[i:i+batch_size]
        batch_seed_data = seed_data[i:i+batch_size]

        batch_outputs = llm.chat(messages=batch_conversations, 
                                 sampling_params=sampling_params, 
                                 use_tqdm=True)
        
        parsed_outputs = []
        for idx, output in enumerate(batch_outputs):
            s = output.outputs[0].text
            total_tokens+=len(output.outputs[0].token_ids)
            o=s
            try:
                match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
                if match:
                    s1 = match.group(1)
                    try:
                        o = json.loads(s1)
                    except json.JSONDecodeError:
                        o = s
            except json.JSONDecodeError:
                o = s
            
            parsed_outputs.append({"seed": batch_seed_data[idx], "response": o, "index_num": c})
            c+=1

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing_data = load_json(output_path)
        else:
            existing_data = []
        existing_data.extend(parsed_outputs)
        with open(output_path, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Total tokens generated: {total_tokens}")
        print("-"*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--qa", action='store_true', help="Set this flag for QA generation")
    
    args = parser.parse_args()
    
    generate_texts(
        args.text_type, args.batch_size, args.qa
    )

if __name__ == "__main__":
    main()