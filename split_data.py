import os
import pandas as pd
import numpy as np

data_dir = 'dataset'
player_columns = [
    "OFF_PLAYER1_ID", "OFF_PLAYER2_ID", "OFF_PLAYER3_ID", "OFF_PLAYER4_ID", "OFF_PLAYER5_ID",
    "DEF_PLAYER1_ID", "DEF_PLAYER2_ID", "DEF_PLAYER3_ID", "DEF_PLAYER4_ID", "DEF_PLAYER5_ID"
]

player_counts = {}
total_possessions = 0

# First pass: count how many times each player appears and total possessions
for filename in os.listdir(data_dir):
    if filename.lower().endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        
        # Ensure all player_columns are present
        missing_cols = [col for col in player_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in file {filename}, skipping this file.")
            continue
        
        # Remove rows with NaNs in player columns
        df = df.dropna(subset=player_columns, how='any')

        # Each row represents one possession
        total_possessions += len(df)

        for col in player_columns:
            # Convert to numeric and drop any non-numeric values
            players = pd.to_numeric(df[col], errors='coerce').dropna()

            # Filter out invalid IDs like 0 if known invalid
            players = players[players > 0]

            # Convert to int
            players = players.astype(int)

            # Count appearances
            for p in players:
                player_counts[p] = player_counts.get(p, 0) + 1

# Calculate the threshold
# average_team_possessions = total_possessions / 30.0
# threshold = 0.2 * average_team_possessions  # 10% of average team possessions
threshold = 4000

# Identify rare players
rare_players = {p for p, count in player_counts.items() if count < threshold}

# Define a placeholder for rare players
RARE_PLAYER_ID = 999999

# Second pass: replace rare players
output_dir = 'dataset_cleaned'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.lower().endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        
        # Verify columns again just in case
        missing_cols = [col for col in player_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in file {filename}, skipping this file.")
            continue

        # Convert to numeric before replacement to avoid issues if there are non-numeric values
        for col in player_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Filter out invalid (if needed), though here we can just replace them if rare
            df[col] = df[col].apply(lambda x: RARE_PLAYER_ID if pd.notnull(x) and x in rare_players else x)

        # Drop rows that became NaN entirely if any (optional)
        df = df.dropna(subset=player_columns, how='any')

        # Convert back to int where possible (for uniformity)
        for col in player_columns:
            # Ensure no NaNs remain
            df[col] = df[col].fillna(RARE_PLAYER_ID)
            df[col] = df[col].astype(int)

        df.to_csv(os.path.join(output_dir, filename), index=False)

print("Processing complete.")

import os
import pandas as pd

data_dir = 'dataset_cleaned'  # Directory where all the CSV files are located
player_id_set = set()

# Explicitly define player ID columns to avoid including non-player columns
player_columns = [
    "OFF_PLAYER1_ID", "OFF_PLAYER2_ID", "OFF_PLAYER3_ID", "OFF_PLAYER4_ID", "OFF_PLAYER5_ID",
    "DEF_PLAYER1_ID", "DEF_PLAYER2_ID", "DEF_PLAYER3_ID", "DEF_PLAYER4_ID", "DEF_PLAYER5_ID"
]

for filename in os.listdir(data_dir):
    if filename.lower().endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        
        # Check if all required columns are present
        missing_cols = [col for col in player_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing expected player columns {missing_cols} in file {filename}")
            continue
        
        # Process each player column
        for col in player_columns:
            # Drop NaN
            col_values = df[col].dropna()
            
            # Ensure values are numeric
            col_values = pd.to_numeric(col_values, errors='coerce').dropna()
            
            # Filter out invalid IDs (e.g., 0 if not a real player ID)
            col_values = col_values[col_values > 0]
            
            # Convert to int
            col_values = col_values.astype(int)
            
            # Update the set of player IDs
            player_id_set.update(col_values.unique())

# After processing all files, player_id_set contains all unique player IDs
print(f"Total unique player IDs found: {len(player_id_set)}")
print(player_id_set)

import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = 'dataset_cleaned'  # Directory with all your cleaned CSVs
output_dir = 'split_data_parquet'
os.makedirs(output_dir, exist_ok=True)

# 1. Combine all CSVs into one DataFrame
all_dfs = []
for filename in os.listdir(data_dir):
    if filename.lower().endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
del all_dfs  # free memory

# 2. Shuffle the entire combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Split into train, val, test (80/10/10 example)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_df, temp_df = train_test_split(combined_df, test_size=(1 - train_ratio), random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=(test_ratio/(test_ratio+val_ratio)), random_state=42)

del combined_df, temp_df  # free memory

# Define how many shards you want per split
num_shards_train = 10
num_shards_val = 2
num_shards_test = 2

player_columns = [
    "OFF_PLAYER1_ID", "OFF_PLAYER2_ID", "OFF_PLAYER3_ID", "OFF_PLAYER4_ID", "OFF_PLAYER5_ID",
    "DEF_PLAYER1_ID", "DEF_PLAYER2_ID", "DEF_PLAYER3_ID", "DEF_PLAYER4_ID", "DEF_PLAYER5_ID"
]

# Before writing shards, inspect player IDs
all_ids = pd.concat([train_df[player_columns], val_df[player_columns], test_df[player_columns]], ignore_index=True)
max_id = all_ids.max().max()
min_id = all_ids.min().min()

print("Max player ID:", max_id)
print("Min player ID:", min_id)

def write_parquet_shards(df, split_name, num_shards):
    # Calculate shard sizes
    num_rows = len(df)
    shard_size = num_rows // num_shards
    remainder = num_rows % num_shards

    start = 0
    for i in range(num_shards):
        end = start + shard_size + (1 if i < remainder else 0)
        shard_df = df.iloc[start:end]
        shard_path = os.path.join(output_dir, f"{split_name}_{i:03d}.parquet")
        shard_df.to_parquet(shard_path, index=False)
        start = end

# After any adjustments, write out the shards
write_parquet_shards(train_df, 'train', num_shards_train)
write_parquet_shards(val_df, 'val', num_shards_val)
write_parquet_shards(test_df, 'test', num_shards_test)

print("Data has been successfully split into Parquet shards.")
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")