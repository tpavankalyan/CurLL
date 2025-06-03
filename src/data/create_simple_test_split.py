import os
import json
import numpy as np
import pandas as pd
import random
import torch
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def concat_instruction_response(row):
    """Concatenates 'instruction' and 'response' fields into a single string."""
    return row["instruction"] + "\n\n" + row["response"]

def process_single_indicator_optimized(tempdf, model_name="all-MiniLM-L6-v2", num_test_samples=100):
    """
    Processes a single indicator's data to determine 'test' and 'val' splits based on diversity.
    This function is designed to be run in parallel for different indicators.

    Args:
        tempdf (pd.DataFrame): A DataFrame containing rows for a single indicator,
                               preserving the original index from the main DataFrame.
        model_name (str): The name of the SentenceTransformer model to use for embeddings.
        num_test_samples (int): The number of samples to assign to the 'test' split.

    Returns:
        list: A list of tuples, where each tuple is (original_df_index, 'test'/'val').
    """
    
    # Store original indices to map back to the main DataFrame later
    # tempdf already preserves the original index from the main DataFrame
    original_corr_indices = tempdf.index.tolist() 

    # Create the combined instruction-response output
    tempdf['ir_output'] = tempdf.apply(concat_instruction_response, axis=1)
    all_irs = tempdf['ir_output'].tolist()
    
    # Load the SentenceTransformer model. This is done inside the process
    # to avoid pickling issues and to manage GPU memory per process if available.
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = model.encode(all_irs, normalize_embeddings=True)
    
    # Calculate cosine distances between all instruction-response pairs
    dist_matrix = cosine_distances(embeddings)

    N = len(all_irs) # Total number of samples for this indicator
    x = 200 # Number of diverse samples to select based on the criteria

    # Initialize masks to track selected and remaining samples efficiently
    selected_mask = np.zeros(N, dtype=bool) # True for selected, False for not
    remaining_mask = np.ones(N, dtype=bool) # True for remaining, False for not

    selected_indices = [] # Stores the indices of selected samples in order of selection

    # Greedy selection loop to pick 'x' diverse samples
    for _ in range(x):
        best_idx, best_score = None, -np.inf
        
        # Get current selected and remaining indices using masks
        current_selected_indices_np = np.where(selected_mask)[0]
        current_remaining_indices_np = np.where(remaining_mask)[0] 

        # If there are no more candidates to select from, break the loop
        if len(current_remaining_indices_np) == 0:
            break

        # Calculate the sum of distances between currently selected and currently remaining samples.
        # This sum is updated incrementally for each candidate in the inner loop.
        sum_dist_S_R_current = 0
        if len(current_selected_indices_np) > 0 and len(current_remaining_indices_np) > 0:
            sum_dist_S_R_current = dist_matrix[np.ix_(current_selected_indices_np, current_remaining_indices_np)].sum()

        # Iterate through all currently remaining candidates to find the best one to add
        for idx_candidate in current_remaining_indices_np:
            new_num_selected = len(current_selected_indices_np) + 1
            new_num_remaining = len(current_remaining_indices_np) - 1

            # If adding this candidate makes the remaining set empty AND we still need more items,
            # this candidate cannot be chosen as the metric requires a non-empty remaining set for future steps.
            if new_num_remaining == 0 and new_num_selected < x:
                continue 
            
            # If new_num_remaining is 0, the average distance to remaining is undefined.
            # In this case, we skip the candidate as it cannot be evaluated by the metric.
            if new_num_remaining == 0:
                continue

            # Calculate sum of distances from current selected items to this candidate
            sum_dist_selected_to_candidate = 0
            if len(current_selected_indices_np) > 0:
                sum_dist_selected_to_candidate = dist_matrix[current_selected_indices_np, idx_candidate].sum()

            # Calculate sum of distances from this candidate to the *new* remaining set
            # (which is the current remaining set minus this candidate itself)
            temp_remaining_indices_excluding_candidate_np = current_remaining_indices_np[current_remaining_indices_np != idx_candidate]
            
            sum_dist_candidate_to_new_remaining = 0
            if len(temp_remaining_indices_excluding_candidate_np) > 0:
                sum_dist_candidate_to_new_remaining = dist_matrix[idx_candidate, temp_remaining_indices_excluding_candidate_np].sum()

            # Calculate the score for this candidate using the optimized formula
            # Score = (Sum_dist(S_current, R_current) - Sum_dist(S_current, candidate) + Sum_dist(candidate, R_new)) / (new_num_selected * new_num_remaining)
            score_numerator = sum_dist_S_R_current - sum_dist_selected_to_candidate + sum_dist_candidate_to_new_remaining
            avg_dist = score_numerator / (new_num_selected * new_num_remaining)

            # Update best candidate if current one has a higher score
            if avg_dist > best_score:
                best_score, best_idx = avg_dist, idx_candidate
        
        # If a best candidate was found, add it to the selected set and update masks
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            remaining_mask[best_idx] = False # Remove from remaining candidates
        else:
            # If no best_idx found (e.g., no more candidates satisfy the criteria), break
            break

    # After selecting 'x' items, sort them by their average distance to the *final* remaining set
    final_remaining_indices = np.where(remaining_mask)[0]
    
    if len(final_remaining_indices) > 0:
        selected_with_scores = []
        for idx in selected_indices:
            # Calculate average distance from this selected item to the final remaining items
            avg_dist_to_remaining = dist_matrix[idx, final_remaining_indices].mean()
            selected_with_scores.append((idx, avg_dist_to_remaining))
        
        # Sort by distance score (highest first - most different)
        selected_with_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_selected_indices = [idx for idx, _ in selected_with_scores]
    else:
        # If no final remaining indices, all selected items are "equally" good
        # Just use the order they were selected in.
        sorted_selected_indices = selected_indices 

    # Assign the top 'num_test_samples' to 'test' split, and the rest to 'val' split
    test_indices = sorted_selected_indices[:num_test_samples]
    val_indices = sorted_selected_indices[num_test_samples:]
    
    # Prepare results to be returned: (original_df_index, split_value)
    results = []
    for i in test_indices:
        results.append((original_corr_indices[i], 'test'))
    for i in val_indices:
        results.append((original_corr_indices[i], 'val'))
    
    return results

def main(stage):
    """
    Main function to load data, process it in parallel, and push the split dataset to Hugging Face Hub.

    Args:
        stage (int): The stage number for the dataset (e.g., 1, 2).
    """
    # Load the dataset from Hugging Face Hub
    ds = load_dataset(f"Pavankalyan/stage{stage}_instruct_cleaned")
    df = ds['train'].to_pandas()
    df['split'] = 'train' # Initialize 'split' column for all rows

    # Get unique indicator IDs
    indicators = df['id'].unique().tolist()

    # Create a list of DataFrames, each containing data for a single indicator.
    # This ensures that the original DataFrame indices are preserved within each chunk.
    indicator_dfs = [df[df['id'] == indicator_id] for indicator_id in indicators]

    # Determine the number of processes to use (defaults to all available CPU cores)
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel processing.")

    all_results = []
    # Use multiprocessing Pool to process each indicator DataFrame in parallel
    with Pool(processes=num_processes) as pool:
        # imap_unordered is used with tqdm for better progress reporting
        for result_chunk in tqdm(pool.imap_unordered(process_single_indicator_optimized, indicator_dfs), 
                                 total=len(indicator_dfs), 
                                 desc="Processing indicators"):
            all_results.extend(result_chunk)

    # Apply the collected split results back to the original DataFrame efficiently
    # Create a pandas Series where index is original_df_index and value is 'test'/'val'
    split_updates = pd.Series({idx: split for idx, split in all_results})
    # Use .loc for setting values based on index
    df.loc[split_updates.index, 'split'] = split_updates.values

    # Print value counts for the 'split' column to verify distribution
    print(df['split'].value_counts())

    # Convert the pandas DataFrame back to a Hugging Face DatasetDict
    # Drop the temporary 'ir_output' column before creating the dataset
    split_dfs = {split: df[df['split'] == split].drop(columns=['split', 'ir_output'], errors='ignore') 
                 for split in df['split'].unique()}

    dataset_dict = {split: Dataset.from_pandas(split_df, preserve_index=False) 
                    for split, split_df in split_dfs.items()}
    hf_dataset = DatasetDict(dataset_dict)

    # Push the newly split dataset to Hugging Face Hub
    hf_dataset.push_to_hub(f"Pavankalyan/stage{stage}_instruct_simple_split")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset based on indicator diversity.")
    parser.add_argument("--stage", type=int, required=True, help="The stage number of the dataset to process.")
    args = parser.parse_args()
    main(args.stage)

