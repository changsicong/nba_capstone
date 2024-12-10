import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD

##############################################
# Parameters & Setup
##############################################
seq_len = 10
embedding_dim = 8
batch_size = 64
data_dir = 'split_data_parquet'

player_columns = [
    "OFF_PLAYER1_ID", "OFF_PLAYER2_ID", "OFF_PLAYER3_ID", "OFF_PLAYER4_ID", "OFF_PLAYER5_ID",
    "DEF_PLAYER1_ID", "DEF_PLAYER2_ID", "DEF_PLAYER3_ID", "DEF_PLAYER4_ID", "DEF_PLAYER5_ID"
]

main_out_column = "OUTCOME"
second_chance_column = "SECOND_CHANCE"
unwanted_cols = ["SHOOTER_ID","ASSISTER_ID","BLOCKER_ID","STEALER_ID","REBOUNDER_ID","TURNOVER_ID"]

# Identify shard files
train_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('train_') and f.endswith('.parquet')])
val_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('val_') and f.endswith('.parquet')])
test_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('test_') and f.endswith('.parquet')])

##############################################
# Build Player ID Mapping
##############################################
def gather_unique_player_ids(files, player_cols):
    unique_ids = set()
    for fpath in files:
        print(f"Scanning file for unique IDs: {fpath}")
        df = pd.read_parquet(fpath, columns=player_cols)  # load only player columns
        df = df.dropna(subset=player_cols)
        for col in player_cols:
            unique_ids.update(df[col].dropna().astype(int).unique())
    return unique_ids

all_files = train_files + val_files + test_files
all_unique_ids = gather_unique_player_ids(all_files, player_columns)

unique_players = np.sort(list(all_unique_ids))
player_to_index = {p: i for i, p in enumerate(unique_players)}
v = len(unique_players)
print(f"Number of unique players: {v}")
print("Example mapping:", list(player_to_index.items())[:10])

##############################################
# Model Definition Using Embeddings (with updated v)
##############################################
def slice_offense(t):
    return t[:, :5, :]

def slice_defense(t):
    return t[:, 5:, :]

def mean_axis_1(t):
    return tf.reduce_mean(t, axis=1)

input_players = Input(shape=(seq_len,), dtype='int32', name='players_input')
player_embedding = layers.Embedding(input_dim=v, output_dim=embedding_dim, name='player_embedding')(input_players)

offense_emb = layers.Lambda(slice_offense, name='offense_slice')(player_embedding)
defense_emb = layers.Lambda(slice_defense, name='defense_slice')(player_embedding)
off_mean = layers.Lambda(mean_axis_1, name='off_mean')(offense_emb)
def_mean = layers.Lambda(mean_axis_1, name='def_mean')(defense_emb)

concat = layers.Concatenate(name='concat')([off_mean, def_mean])
hidden = layers.Dense(128, activation='relu', name='hidden')(concat)

main_out = layers.Dense(14, activation='softmax', name='main_out')(hidden)
second_chance_out = layers.Dense(1, activation='sigmoid', name='second_chance_out')(hidden)

# Set a custom learning rate
# optimizer = Adam(learning_rate=0.0001)
optimizer = SGD(learning_rate=0.01, momentum=0.9)

model = Model(inputs=input_players, outputs=[main_out, second_chance_out])
model.compile(
    optimizer=optimizer,
    loss={
        'main_out': 'categorical_crossentropy',
        'second_chance_out': 'binary_crossentropy'
    },
    metrics={
        'main_out': 'accuracy',
        'second_chance_out': 'accuracy'
    }
)

model.summary()

##############################################
# Utility Functions for tf.data Pipeline
##############################################
def shard_generator(file_list, main_col, sc_col, mapping):
    """
    Yields individual samples (X, (y_main, y_sc)) from shard files.
    Applies player_to_index mapping to ensure IDs are in [0, v-1].
    """
    for fpath in file_list:
        print(f"Loading shard: {fpath}")
        df = pd.read_parquet(fpath)

        # Drop rows with NaNs in player columns
        df = df.dropna(subset=player_columns)

        # Convert players to int
        for col in player_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Drop unwanted columns
        for c in unwanted_cols:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        # One-hot the outcome column
        categories = sorted(df[main_col].unique())
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        num_samples = len(df)
        y_main = np.zeros((num_samples, 14), dtype='float32')
        for i, val in enumerate(df[main_col]):
            class_idx = cat_to_idx[val]
            y_main[i, class_idx] = 1.0

        y_sc = df[sc_col].astype(int).values.reshape(-1, 1)

        # Drop target columns now
        df.drop(columns=[main_col, sc_col], inplace=True)

        # Map player IDs
        for c in player_columns:
            df[c] = df[c].map(mapping)

        X = df[player_columns].values.astype(np.int32)

        # Yield each sample
        for i in range(num_samples):
            yield X[i], (y_main[i], y_sc[i])

def create_dataset(file_list, main_col, sc_col, batch_size, mapping, shuffle_buffer=10000):
    ds = tf.data.Dataset.from_generator(
        lambda: shard_generator(file_list, main_col, sc_col, mapping),
        output_types=(tf.int32, (tf.float32, tf.float32)),
        output_shapes=((seq_len,), ((14,), (1,)))
    )

    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

##############################################
# Create Datasets using mapping
##############################################
train_ds = create_dataset(train_files, main_out_column, second_chance_column, batch_size, player_to_index)
val_ds = create_dataset(val_files, main_out_column, second_chance_column, batch_size, player_to_index)
test_ds = create_dataset(test_files, main_out_column, second_chance_column, batch_size, player_to_index)

##############################################
# Training with tf.data
##############################################
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

##############################################
# Evaluation
##############################################
model.evaluate(test_ds)

model.save('model_one.keras')