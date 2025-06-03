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
import gc

def concat_instruction_response(row):
    """Concatenates 'instruction' and 'response' fields into a single string."""
    return row["instruction"] + "\n\n" + row["response"]

def process_single_indicator_cpu_only(args):
    """
    Processes a single indicator's data using pre-computed embeddings.
    This avoids loading the model in each process.
    
    Args:
        args: tuple of (tempdf, embeddings_for_indicator, num_test_samples)
        
    Returns:
        list: A list of tuples, where each tuple is (original_df_index, 'test'/'val').
    """
    tempdf, embeddings, num_test_samples = args
    
    # Store original indices to map back to the main DataFrame later
    original_corr_indices = tempdf.index.tolist() 
    
    # Calculate cosine distances between all instruction-response pairs
    dist_matrix = cosine_distances(embeddings)

    N = len(embeddings) # Total number of samples for this indicator
    x = min(200, N)  # Don't select more than available samples

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
            if new_num_remaining == 0:
                continue

            # Calculate sum of distances from current selected items to this candidate
            sum_dist_selected_to_candidate = 0
            if len(current_selected_indices_np) > 0:
                sum_dist_selected_to_candidate = dist_matrix[current_selected_indices_np, idx_candidate].sum()

            # Calculate sum of distances from this candidate to the *new* remaining set
            temp_remaining_indices_excluding_candidate_np = current_remaining_indices_np[current_remaining_indices_np != idx_candidate]
            
            sum_dist_candidate_to_new_remaining = 0
            if len(temp_remaining_indices_excluding_candidate_np) > 0:
                sum_dist_candidate_to_new_remaining = dist_matrix[idx_candidate, temp_remaining_indices_excluding_candidate_np].sum()

            # Calculate the score for this candidate using the optimized formula
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
            # If no best_idx found, break
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
        # If no final remaining indices, use selection order
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

def compute_embeddings_batch(df, model_name="all-MiniLM-L6-v2", batch_size=1000):
    """
    Compute embeddings for all indicators in batches to manage memory efficiently.
    
    Args:
        df: DataFrame with the data
        model_name: SentenceTransformer model name
        batch_size: Number of samples to process at once
        
    Returns:
        dict: indicator_id -> embeddings array
    """
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create instruction-response concatenations
    df['ir_output'] = df.apply(concat_instruction_response, axis=1)
    
    embeddings_by_indicator = {}
    indicators = df['id'].unique()
    
    print("Computing embeddings for each indicator...")
    for indicator_id in tqdm(indicators, desc="Processing indicators"):
        indicator_df = df[df['id'] == indicator_id]
        texts = indicator_df['ir_output'].tolist()
        
        # Process in batches to manage memory
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
            all_embeddings.append(batch_embeddings)
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batch embeddings
        embeddings_by_indicator[indicator_id] = np.vstack(all_embeddings)
        
        # Force garbage collection
        gc.collect()
    
    # Clear model from memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return embeddings_by_indicator

def main(stage, max_processes=None):
    """
    Main function to load data, process it with controlled parallelism, and push the split dataset to Hugging Face Hub.

    Args:
        stage (int): The stage number for the dataset (e.g., 1, 2).
        max_processes (int): Maximum number of processes to use (default: min(4, cpu_count()))
    """
    # Load the dataset from Hugging Face Hub
    print(f"Loading dataset for stage {stage}...")
    ds = load_dataset(f"Pavankalyan/stage{stage}_instruct_cleaned")
    df = ds['train'].to_pandas()
    df['split'] = 'train' # Initialize 'split' column for all rows
    
    print(f"Dataset loaded with {len(df)} samples and {df['id'].nunique()} unique indicators")

    # Compute embeddings for all indicators
    embeddings_by_indicator = compute_embeddings_batch(df)
    
    # Prepare data for multiprocessing
    indicators = df['id'].unique().tolist()
    process_args = []
    
    for indicator_id in indicators:
        indicator_df = df[df['id'] == indicator_id]
        embeddings = embeddings_by_indicator[indicator_id]
        process_args.append((indicator_df, embeddings, 100))  # num_test_samples = 100
    
    # Use fewer processes to avoid memory issues
    if max_processes is None:
        max_processes = min(4, cpu_count())  # Limit to 4 processes max
    
    print(f"Using {max_processes} processes for parallel processing.")

    all_results = []
    # Use multiprocessing Pool with limited processes
    with Pool(processes=max_processes) as pool:
        for result_chunk in tqdm(pool.imap_unordered(process_single_indicator_cpu_only, process_args), 
                                 total=len(process_args), 
                                 desc="Processing indicators"):
            all_results.extend(result_chunk)

    # Apply the collected split results back to the original DataFrame
    split_updates = pd.Series({idx: split for idx, split in all_results})
    df.loc[split_updates.index, 'split'] = split_updates.values

    # Print value counts for the 'split' column to verify distribution
    print("Split distribution:")
    print(df['split'].value_counts())

    # Convert the pandas DataFrame back to a Hugging Face DatasetDict
    split_dfs = {split: df[df['split'] == split].drop(columns=['split', 'ir_output'], errors='ignore') 
                 for split in df['split'].unique()}

    dataset_dict = {split: Dataset.from_pandas(split_df, preserve_index=False) 
                    for split, split_df in split_dfs.items()}
    hf_dataset = DatasetDict(dataset_dict)

    # Push the newly split dataset to Hugging Face Hub
    print(f"Pushing dataset to Hugging Face Hub...")
    hf_dataset.push_to_hub(f"Pavankalyan/stage{stage}_instruct_simple_split")
    print("Dataset successfully pushed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset based on indicator diversity.")
    parser.add_argument("--stage", type=int, required=True, help="The stage number of the dataset to process.")
    parser.add_argument("--max-processes", type=int, default=None, help="Maximum number of processes to use (default: min(4, cpu_count()))")
    args = parser.parse_args()
    main(args.stage, args.max_processes)