#load a huggingface dataset
from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("roneneldan/TinyStoriesInstruct")

train_samples = [[]]

for i in tqdm(range(len(ds['train']))):
    train_samples[-1].extend(ds['train'][i]['text'])
    if ds['train'][i]['text'] == '<|endoftext|>' or ds['train'][i]['text'].endswith('<|endoftext|>'):
        train_samples.append([])

print(len(train_samples))
print(train_samples[0])

#save the list
import json
with open('train_samples.json', 'w') as f:
    json.dump(train_samples, f)


# import json

# with open('train_samples.json', 'r') as f:
#     train_samples = json.load(f)
# print(len(train_samples))