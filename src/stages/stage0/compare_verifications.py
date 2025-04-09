import pandas as pd
import json

# Function to load a jsonl file into a DataFrame
def load_jsonl_to_df(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load both files
df1 = load_jsonl_to_df('/datadrive/pavan/az_storage/data_unorganized/age_0_5/questions/poem/random_5000_verified_gemma.json')
df2 = load_jsonl_to_df('/datadrive/pavan/az_storage/data_unorganized/age_0_5/questions/poem/random_5000_verified_gpt4_5.json')

# Keep only entries where 'response' is a dict in both files
df1_valid = df1[df1['response'].apply(lambda x: isinstance(x, dict))].copy()
df2_valid = df2[df2['response'].apply(lambda x: isinstance(x, dict))].copy()

# Merge only on index_num where both are valid
valid_indices = set(df1_valid['index_num']) & set(df2_valid['index_num'])

df1_valid = df1_valid[df1_valid['index_num'].isin(valid_indices)]
df2_valid = df2_valid[df2_valid['index_num'].isin(valid_indices)]

# Flatten response ratings
def flatten_response(df, suffix):
    #response_fields = ['rhythm_and_flow', 'relevance_to_topic', 'skill_reinforcement', 'age_appropriateness', 'overall_quality']
    response_fields = ['skill_alignment', 'ambiguity', 'content_relevance', 'open_endedness', 'age_appropriateness', 'readability', 'skill_coverage', 'response_quality', 'poetic_context_integration', 'overall_quality']
    for field in response_fields:
        df[f'{field}_rating_{suffix}'] = df['response'].apply(lambda x: float(x[field]['rating']))
    return df[['index_num'] + [f'{field}_rating_{suffix}' for field in response_fields]]



df1_flat = flatten_response(df1_valid, 'file1')
df2_flat = flatten_response(df2_valid, 'file2')

# Merge on index_num
merged = pd.merge(df1_flat, df2_flat, on='index_num')

# Calculate agreement
for field in ['rhythm_and_flow', 'relevance_to_topic', 'skill_reinforcement', 'age_appropriateness', 'overall_quality']:
    merged[f'{field}_agreement'] = merged[f'{field}_rating_file1'] == merged[f'{field}_rating_file2']

print(len(merged))

# Agreement summary
agreement_summary = merged[[col for col in merged.columns if col.endswith('_agreement')]].mean()

print("✅ Agreement Summary (fraction of exact matches):")
print(agreement_summary)

# Optional: preview mismatches
print("\n❌ Mismatched examples:")
print(merged[merged[[col for col in merged.columns if col.endswith('_agreement')]].any(axis=1) == False])