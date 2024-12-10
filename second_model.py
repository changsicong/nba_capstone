import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam

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

# These are the role ID columns given as actual player IDs
role_id_cols = ["SHOOTER_ID", "ASSISTER_ID", "BLOCKER_ID", "STEALER_ID", "REBOUNDER_ID", "TURNOVER_ID"]

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
# Transformer Encoder Block Definition
##############################################
def transformer_encoder(inputs, num_heads=2, ff_dim=32, dropout_rate=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(inputs, inputs)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = layers.Dense(ff_dim, activation='relu')(out1)
    ffn = layers.Dense(embedding_dim)(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2

##############################################
# Define Named Functions for Lambdas
##############################################
def slice_offense(t):
    return t[:, :5, :]

def slice_defense(t):
    return t[:, 5:, :]

def mean_axis_1(t):
    return tf.reduce_mean(t, axis=1)

##############################################
# Model Definition Using Transformer
##############################################
input_players = Input(shape=(seq_len,), dtype='int32', name='players_input')
player_embedding = layers.Embedding(input_dim=v, output_dim=embedding_dim, name='player_embedding')(input_players)

x = transformer_encoder(player_embedding, num_heads=2, ff_dim=32, dropout_rate=0.1)

offense_emb = layers.Lambda(slice_offense, name='offense_slice')(x)
defense_emb = layers.Lambda(slice_defense, name='defense_slice')(x)

off_mean = layers.Lambda(mean_axis_1, name='off_mean')(offense_emb)
def_mean = layers.Lambda(mean_axis_1, name='def_mean')(defense_emb)

concat = layers.Concatenate(name='concat')([off_mean, def_mean])
hidden = layers.Dense(128, activation='relu', name='hidden')(concat)

main_out = layers.Dense(14, activation='softmax', name='main_out')(hidden)
second_chance_out = layers.Dense(1, activation='sigmoid', name='second_chance_out')(hidden)

# For roles: shooter, assister, blocker, stealer, rebounder, turnover
# mean-pool all players for roles
all_mean = layers.Lambda(mean_axis_1, name='all_mean')(x)
roles_hidden = layers.Dense(64, activation='relu', name='roles_hidden')(all_mean)

# Each role output: 10 classes (one for each of the 10 players on the floor)
shooter_out = layers.Dense(seq_len, activation='softmax', name='shooter_out')(roles_hidden)
assister_out = layers.Dense(seq_len, activation='softmax', name='assister_out')(roles_hidden)
blocker_out = layers.Dense(seq_len, activation='softmax', name='blocker_out')(roles_hidden)
stealer_out = layers.Dense(seq_len, activation='softmax', name='stealer_out')(roles_hidden)
rebounder_out = layers.Dense(seq_len, activation='softmax', name='rebounder_out')(roles_hidden)
turnover_out = layers.Dense(seq_len, activation='softmax', name='turnover_out')(roles_hidden)

model = Model(inputs=input_players, outputs=[
    main_out, second_chance_out,
    shooter_out, assister_out, blocker_out, stealer_out, rebounder_out, turnover_out
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss={
        'main_out': 'categorical_crossentropy',
        'second_chance_out': 'binary_crossentropy',
        'shooter_out': 'categorical_crossentropy',
        'assister_out': 'categorical_crossentropy',
        'blocker_out': 'categorical_crossentropy',
        'stealer_out': 'categorical_crossentropy',
        'rebounder_out': 'categorical_crossentropy',
        'turnover_out': 'categorical_crossentropy'
    },
    metrics={
        'main_out': 'accuracy',
        'second_chance_out': 'accuracy',
        'shooter_out': 'accuracy',
        'assister_out': 'accuracy',
        'blocker_out': 'accuracy',
        'stealer_out': 'accuracy',
        'rebounder_out': 'accuracy',
        'turnover_out': 'accuracy'
    }
)

model.summary()

##############################################
# Utility Functions for tf.data Pipeline
##############################################
def one_hot_role_from_id(role_id_array, row_players):
    num_samples = len(role_id_array)
    arr = np.zeros((num_samples, 10), dtype='float32')
    for i, rid in enumerate(role_id_array):
        if pd.isnull(rid):
            continue
        rid = int(rid)
        idx_pos = np.where(row_players[i] == rid)[0]
        if len(idx_pos) > 0:
            arr[i, idx_pos[0]] = 1.0
    return arr

def shard_generator(file_list, main_col, sc_col, mapping):
    for fpath in file_list:
        print(f"Loading shard: {fpath}")
        df = pd.read_parquet(fpath)
        df = df.dropna(subset=player_columns)
        df_players_original = df[player_columns].copy()

        # Convert to int
        for col in player_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Outcome one-hot
        categories = sorted(df[main_col].unique())
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        num_samples = len(df)
        y_main = np.zeros((num_samples, 14), dtype='float32')
        for i, val in enumerate(df[main_col]):
            class_idx = cat_to_idx[val]
            y_main[i, class_idx] = 1.0

        y_sc = df[sc_col].astype(int).values.reshape(-1, 1)

        # Role IDs
        shooter_ids = df["SHOOTER_ID"]
        assister_ids = df["ASSISTER_ID"]
        blocker_ids = df["BLOCKER_ID"]
        stealer_ids = df["STEALER_ID"]
        rebounder_ids = df["REBOUNDER_ID"]
        turnover_ids = df["TURNOVER_ID"]

        # Drop target columns
        df.drop(columns=[main_col, sc_col, "SHOOTER_ID", "ASSISTER_ID", "BLOCKER_ID", "STEALER_ID", "REBOUNDER_ID", "TURNOVER_ID"], inplace=True)

        row_players_original = df_players_original[player_columns].values
        y_shooter = one_hot_role_from_id(shooter_ids, row_players_original)
        y_assister = one_hot_role_from_id(assister_ids, row_players_original)
        y_blocker = one_hot_role_from_id(blocker_ids, row_players_original)
        y_stealer = one_hot_role_from_id(stealer_ids, row_players_original)
        y_rebounder = one_hot_role_from_id(rebounder_ids, row_players_original)
        y_turnover = one_hot_role_from_id(turnover_ids, row_players_original)

        # Map player IDs now
        for c in player_columns:
            df[c] = df[c].map(mapping)
        X = df[player_columns].values.astype(np.int32)

        for i in range(num_samples):
            yield X[i], (y_main[i], y_sc[i], y_shooter[i], y_assister[i], y_blocker[i], y_stealer[i], y_rebounder[i], y_turnover[i])

def create_dataset(file_list, main_col, sc_col, batch_size, mapping, shuffle_buffer=10000):
    ds = tf.data.Dataset.from_generator(
        lambda: shard_generator(file_list, main_col, sc_col, mapping),
        output_types=(tf.int32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)),
        output_shapes=((seq_len,), ((14,), (1,), (10,), (10,), (10,), (10,), (10,), (10,)))
    )

    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_files, main_out_column, second_chance_column, batch_size, player_to_index)
val_ds = create_dataset(val_files, main_out_column, second_chance_column, batch_size, player_to_index)
test_ds = create_dataset(test_files, main_out_column, second_chance_column, batch_size, player_to_index)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.evaluate(test_ds)