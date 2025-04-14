import pandas as pd
import json

# Function to load a jsonl file into a DataFrame
def load_jsonl_to_df(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load both files
df1 = load_jsonl_to_df('/datadrive/pavan/az_storage/data_unorganized/age_0_5/texts/poem/random_5000_verified_gemma.json')
df2 = load_jsonl_to_df('/datadrive/pavan/az_storage/data_unorganized/age_0_5/texts/poem/random_5000_verified_gpt4_5.json')

# Keep only entries where 'response' is a dict in both files
df1_valid = df1[df1['response'].apply(lambda x: isinstance(x, dict))].copy()
df2_valid = df2[df2['response'].apply(lambda x: isinstance(x, dict))].copy()

# Merge only on index_num where both are valid
valid_indices = set(df1_valid['index_num']) & set(df2_valid['index_num'])

df1_valid = df1_valid[df1_valid['index_num'].isin(valid_indices)]
df2_valid = df2_valid[df2_valid['index_num'].isin(valid_indices)]

# Flatten response ratings
def flatten_response(df, suffix):
    response_fields = ['rhythm_and_flow', 'relevance_to_topic', 'skill_reinforcement', 'age_appropriateness', 'overall_quality']
    # response_fields = ['skill_alignment', 'ambiguity', 'content_relevance', 'open_endedness', 'age_appropriateness', 'readability', 'skill_coverage', 'response_quality', 'poetic_context_integration', 'overall_quality']
    for field in response_fields:
        df[f'{field}_rating_{suffix}'] = df['response'].apply(lambda x: float(x[field]['rating']))
    # return df[['index_num'] + [f'{field}_rating_{suffix}' for field in response_fields]]
    return df

s1 = 'gemma'
s2 = 'gpt4_5'
df1_flat = flatten_response(df1_valid, s1)
df2_flat = flatten_response(df2_valid, s2)

# Merge on index_num
merged = pd.merge(df1_flat, df2_flat, on='index_num')

# Calculate agreement
for field in ['rhythm_and_flow', 'relevance_to_topic', 'skill_reinforcement', 'age_appropriateness', 'overall_quality']:
    merged[f'{field}_agreement'] = merged[f'{field}_rating_{s1}'] == merged[f'{field}_rating_{s2}']

print(len(merged))

print(merged.columns)

for i in range(len(merged)):
    print(i)
    for field in ['rhythm_and_flow', 'relevance_to_topic', 'skill_reinforcement', 'age_appropriateness', 'overall_quality']:
        print(field)
        print(f"Gemma: {merged.iloc[i][f'{field}_rating_{s1}']}, GPT-4.5: {merged.iloc[i][f'{field}_rating_{s2}']}")
        print(f"Agreement: {merged.iloc[i][f'{field}_agreement']}")
        print("")
        print("--------------------------------------------------")
    if i == 10:
        break

print(merged['overall_quality_rating_gpt4_5'].value_counts())

print(merged['overall_quality_rating_gemma'].value_counts())
c = 0
d=0
v = 0
k = 0
s = 0
for i in range(len(merged)):
    if merged['overall_quality_rating_gemma'][i]>merged['overall_quality_rating_gpt4_5'][i]:
        if merged['overall_quality_rating_gemma'][i] - merged['overall_quality_rating_gpt4_5'][i]>1:
            d+=1
        c+=1
    elif merged['overall_quality_rating_gemma'][i]<merged['overall_quality_rating_gpt4_5'][i]:
        if merged['overall_quality_rating_gpt4_5'][i] - merged['overall_quality_rating_gemma'][i]>1:
            v+=1
            if merged['overall_quality_rating_gpt4_5'][i] - merged['overall_quality_rating_gemma'][i]>2:
                s+=1
        k+=1
print(len(merged))
print(f"Total: {c}")
print(f"Total with difference > 1: {d}")
print(f"Total: {k}")
print(f"Total with difference > 1: {v}")
print(f"Total with difference > 2: {s}")
# Agreement summary
agreement_summary = merged[[col for col in merged.columns if col.endswith('_agreement')]].mean()

print("✅ Agreement Summary (fraction of exact matches):")
print(agreement_summary)

# Optional: preview mismatches
print("\n❌ Mismatched examples:")
print(merged[merged[[col for col in merged.columns if col.endswith('_agreement')]].any(axis=1) == False])