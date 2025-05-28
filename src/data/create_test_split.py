import os
import json
import numpy as np
import pandas as pd
import random
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def concat_instruction_response(row):
    return row["instruction"] + "\n\n" + row["response"]


def process_indicator(indicator_id, df, num_templates=2, num_test_samples=50):
    tempdf1 = df[df['id'] == indicator_id].reset_index(drop=True)
    if tempdf1.empty:
        return []  # Return empty list instead of df
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    # Select the topics
    topics = list(df['context_template'].unique())
    Z_topics = list(set(df[df['id']==indicator_id]['context_template'].values))
    Z_indices = [topics.index(t) for t in Z_topics]

    topic_embeddings = model.encode(topics, normalize_embeddings=True)
    topic_dist_matrix = cosine_distances(topic_embeddings)

    # Greedy selection from Z_indices
    topic_selected = []
    topic_candidates = list(Z_indices)
    topic_indices = set(range(len(topics)))

    for _ in range(min(num_templates, len(topic_candidates))):  # Use num_templates parameter
        best_idx = None
        best_score = -np.inf

        for idx in topic_candidates:
            temp_selected = topic_selected + [idx]
            temp_rest = list(topic_indices - set(temp_selected))
            if temp_rest:  # Ensure temp_rest is not empty
                avg_dist = topic_dist_matrix[np.ix_(temp_selected, temp_rest)].mean()
                if avg_dist > best_score:
                    best_score = avg_dist
                    best_idx = idx

        if best_idx is not None:
            topic_selected.append(best_idx)
            topic_candidates.remove(best_idx)

    selected_topics = [topics[i] for i in topic_selected]

    # Find closest indicators
    all_inds = list(df['indicator'].unique())
    main_ind = tempdf1['indicator'].values[0]
    index_ind = all_inds.index(main_ind)

    embeddings_inds = model.encode(all_inds, normalize_embeddings=True)
    dist_matrix = cosine_distances(embeddings_inds)
    closest_indices = np.argsort(dist_matrix[index_ind])
    closest_indices = closest_indices[closest_indices != index_ind][:2]
    closest_ids = [df[df['indicator'] == all_inds[i]]['id'].values[0] for i in closest_indices]

    tempdf2 = df[df['id'].isin(closest_ids)].reset_index(drop=True)
    tempdf1['ir_output'] = tempdf1.apply(concat_instruction_response, axis=1)
    tempdf2['ir_output'] = tempdf2.apply(concat_instruction_response, axis=1)

    tempdf = pd.concat([tempdf1, tempdf2]).reset_index(drop=True)

    # Store results for all topics
    results = []

    for topic in selected_topics:
        topic_df = tempdf1[tempdf1['context_template'] == topic].reset_index(drop=True)
        if len(topic_df) < num_test_samples:
            continue

        main_irs = topic_df['ir_output'].tolist()
        all_irs = tempdf['ir_output'].tolist()

        Z_indices = [all_irs.index(m) for m in main_irs if m in all_irs]
        if not Z_indices:
            continue

        embeddings = model.encode(all_irs, normalize_embeddings=True)
        dist_matrix = cosine_distances(embeddings)

        selected, candidates = [], list(Z_indices)
        Y_indices = set(range(len(all_irs)))

        for _ in range(min(num_test_samples, len(candidates))):
            best_idx, best_score = None, -np.inf
            for idx in candidates:
                temp_selected = selected + [idx]
                temp_rest = list(Y_indices - set(temp_selected))
                if temp_rest:  # Ensure temp_rest is not empty
                    avg_dist = dist_matrix[np.ix_(temp_selected, temp_rest)].mean()
                    if avg_dist > best_score:
                        best_score, best_idx = avg_dist, idx
            
            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)

        if selected:
            selected_rows = tempdf.iloc[selected]
            selected_instructions = selected_rows['instruction'].tolist()
            selected_responses = selected_rows['response'].tolist()

            test_result = ('test', indicator_id, topic, selected_instructions, selected_responses)
            
            # Get remaining instructions for validation set
            all_topic_instructions = set(tempdf1[tempdf1['context_template'] == topic]['instruction'])
            all_topic_responses = set(tempdf1[tempdf1['context_template'] == topic]['response'])
            remaining = list(all_topic_instructions - set(selected_instructions))
            remaining_responses = list(all_topic_responses - set(selected_responses))
            val_result = ('val', indicator_id, topic, remaining, remaining_responses)
            
            results.extend([test_result, val_result])

    return results


def main(stage):
    ds = load_dataset(f"Pavankalyan/stage{stage}_instruct_cleaned")
    df = ds['train'].to_pandas()
    df['split'] = 'train'

    indicators = df['id'].unique().tolist()

    with Pool(min(cpu_count(), 50)) as pool:
        worker = partial(process_indicator, df=df)
        all_results = list(tqdm(pool.imap(worker, indicators), total=len(indicators)))
        
        # Flatten results and apply splits
        for result_list in tqdm(all_results):
            if result_list:
                for result in result_list:
                    split_type, indicator_id, topic, instructions, responses_s = result
                    df.loc[
                        (df['id'] == indicator_id) &
                        (df['context_template'] == topic) &
                        (df['instruction'].isin(instructions)) &
                        (df['response'].isin(responses_s)),
                        'split'
                    ] = split_type

    print(df['split'].value_counts())
    # df.to_csv(f"stage{stage}_split.csv", index=False)
    # print(f"Saved split dataset for stage {stage}.")
    #save df as hf dataset
    import pandas as pd
    from datasets import Dataset, DatasetDict
    # Create a dictionary to hold each split
    split_dfs = {split: df[df['split'] == split].drop(columns=['split']) for split in df['split'].unique()}

    # Convert each split DataFrame to a Hugging Face Dataset
    dataset_dict = {split: Dataset.from_pandas(split_df, preserve_index=False) for split, split_df in split_dfs.items()}

    # Create a DatasetDict
    hf_dataset = DatasetDict(dataset_dict)

    hf_dataset.push_to_hub(f"Pavankalyan/stage{stage}_instruct_split")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()
    main(args.stage)