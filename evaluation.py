import pandas as pd
from nba_api.stats.static import players
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import tensorflow as tf
import os

def generate_compare_data():
    def get_player_id(name):
        res = players.find_players_by_full_name(name)
        if len(res) > 0:
            return res[0]['id']
        return None

    dpm_df = pd.read_csv('dpm.csv')
    dpm_df['player_id'] = dpm_df['player_name'].apply(get_player_id)
    dpm_df = dpm_df.dropna(subset=['player_id'])
    dpm_df['player_id'] = dpm_df['player_id'].astype(int)

    epm_df = pd.read_csv('epm.csv')
    epm_df['player_id'] = epm_df['player_name'].apply(get_player_id)
    epm_df = epm_df.dropna(subset=['player_id'])
    epm_df['player_id'] = epm_df['player_id'].astype(int)

    lebron_df = pd.read_csv('lebron.csv')
    lebron_df['player_id'] = lebron_df['player_name'].apply(get_player_id)
    lebron_df = lebron_df.dropna(subset=['player_id'])
    lebron_df['player_id'] = lebron_df['player_id'].astype(int)

    raptor_df = pd.read_csv('raptor.csv')
    raptor_df['player_id'] = raptor_df['player_name'].apply(get_player_id)
    raptor_df = raptor_df.dropna(subset=['player_id'])
    raptor_df['player_id'] = raptor_df['player_id'].astype(int)

    bpm_df = pd.read_csv('bpm.csv')
    bpm_df['player_id'] = bpm_df['player_name'].apply(get_player_id)
    bpm_df = bpm_df.dropna(subset=['player_id'])
    bpm_df['player_id'] = bpm_df['player_id'].astype(int)

    rapm_df = pd.read_csv('rapm.csv')
    rapm_df['player_id'] = rapm_df['player_name'].apply(get_player_id)
    rapm_df = rapm_df.dropna(subset=['player_id'])
    rapm_df['player_id'] = rapm_df['player_id'].astype(int)

    merged_df = dpm_df.merge(epm_df, on='player_id', how='outer', suffixes=('_dpm','_epm'))
    merged_df = merged_df.merge(lebron_df, on='player_id', how='outer', suffixes=('','_lebron'))
    merged_df = merged_df.merge(raptor_df, on='player_id', how='outer', suffixes=('_final','_raptor'))
    merged_df = merged_df.merge(bpm_df, on='player_id', how='outer', suffixes=('_done','_bpm'))
    merged_df = merged_df.merge(rapm_df, on='player_id', how='outer', suffixes=('_x','_rapm'))

    if 'player_name_final' in merged_df.columns:
        merged_df['player_name'] = merged_df['player_name_final'].combine_first(merged_df['player_name'])
    elif 'player_name' not in merged_df.columns:
        cols_with_name = [c for c in merged_df.columns if 'player_name' in c]
        if cols_with_name:
            merged_df['player_name'] = merged_df[cols_with_name[0]]

    name_cols = [c for c in merged_df.columns if 'player_name' in c and c != 'player_name']
    merged_df = merged_df.drop(columns=name_cols, errors='ignore')

    cols = ['player_id', 'player_name'] + [c for c in merged_df.columns if c not in ['player_id', 'player_name']]
    merged_df = merged_df[cols]

    merged_df.to_csv('merged_metrics.csv', index=False)

generate_compare_data()

def predict_lineup_points(model, player_ids, player_to_index, num_simulations=100):
    lineup_indices = [player_to_index[pid] for pid in player_ids]
    lineup_array = np.array([lineup_indices])
    predictions = model.predict(lineup_array)
    main_out_prob = predictions[0][0]
    second_chance_p = predictions[1][0][0]

    def get_points_and_extras(outcome):
        if outcome in [0,1]: return 2,0
        elif outcome in [2,3]: return 3,0
        elif 4<=outcome<=8: return 0,0
        elif outcome==9: return 1,0
        elif outcome==10: return 2,0
        elif outcome==11: return 3,0
        elif outcome==12: return 1,1
        elif outcome==13: return 0,1
        return 0,0

    total_points=0
    for _ in range(num_simulations):
        points=0
        base_possessions=0
        while base_possessions<100:
            outcome = np.random.choice(np.arange(14), p=main_out_prob)
            second_chance = np.random.rand()<second_chance_p
            pts, extra_pos = get_points_and_extras(outcome)
            points += pts
            base_possessions+=1
            base_possessions+=extra_pos
            if second_chance:
                base_possessions+=1
        total_points+=points
    return total_points/num_simulations

def generate_comparison_sheet(
    games_csv,
    merged_csv,
    player_to_index,
    model_one_path,
    model_two_path
):
    games = pd.read_csv(games_csv)
    merged = pd.read_csv(merged_csv)

    merged.set_index('player_id', inplace=True)

    # Load models
    model_one = tf.keras.models.load_model(model_one_path, compile=False)
    model_two = tf.keras.models.load_model(model_two_path, compile=False)

    def parse_lineup(x):
        return list(map(int,x.split(',')))
    games['lineup_ids'] = games['lineup_player_ids'].apply(parse_lineup)

    metrics = ['dpm','epm','lebron','raptor','bpm','LA-RAPM','RAPM','multi-RAPM','multi-LA-RAPM','My model one','My model two']
    preds = []
    for _, row in games.iterrows():
        actual = row['actual_off_points_per_100']
        pids = row['lineup_ids']
        df_line = merged.loc[merged.index.intersection(pids)]
        if len(df_line)<len(pids):
            continue

        dpm_pred = df_line['dpm'].mean()
        epm_pred = df_line['epm'].mean()
        lebron_pred = df_line['lebron'].mean()
        raptor_pred = df_line['raptor'].mean()
        bpm_pred = df_line['bpm'].mean()
        la_rapm_pred = df_line['LA-RAPM'].mean()
        rapm_pred = df_line['RAPM'].mean()
        multi_rapm_pred = df_line['multi-RAPM'].mean()
        multi_la_rapm_pred = df_line['multi-LA-RAPM'].mean()

        model_one_pred = predict_lineup_points(model_one, pids, player_to_index, num_simulations=100)
        model_two_pred = predict_lineup_points(model_two, pids, player_to_index, num_simulations=100)

        preds.append({
            'actual': actual,
            'dpm': dpm_pred,
            'epm': epm_pred,
            'lebron': lebron_pred,
            'raptor': raptor_pred,
            'bpm': bpm_pred,
            'LA-RAPM': la_rapm_pred,
            'RAPM': rapm_pred,
            'multi-RAPM': multi_rapm_pred,
            'multi-LA-RAPM': multi_la_rapm_pred,
            'My model one': model_one_pred,
            'My model two': model_two_pred
        })

    pred_df = pd.DataFrame(preds)
    metrics_for_eval = ['dpm','epm','lebron','raptor','bpm','LA-RAPM','RAPM','multi-RAPM','multi-LA-RAPM','My model one','My model two']

    rows = ['PCC','MSE','MAE']
    data = []
    for metric in metrics_for_eval:
        x = pred_df[metric].dropna()
        y = pred_df['actual'].loc[x.index]
        if len(x)<2:
            pcc = np.nan
        else:
            pcc, _ = pearsonr(x, y)
        mse = root_mean_squared_error(pred_df['actual'], pred_df[metric], squared=True)
        mae = mean_absolute_error(pred_df['actual'], pred_df[metric])
        data.append([pcc,mse,mae])

    result_df = pd.DataFrame(data, index=metrics_for_eval, columns=rows).T
    result_df.to_csv('comparison_sheet.csv')
    return result_df

games_csv = 'games_2023_2024.csv'
merged_csv = 'merged_metrics.csv'
model_one_path = 'model_one.keras'
model_two_path = 'model_two.keras'

def gather_unique_player_ids(files, player_cols):
    unique_ids = set()
    for fpath in files:
        print(f"Scanning file for unique IDs: {fpath}")
        df = pd.read_parquet(fpath, columns=player_cols)
        df = df.dropna(subset=player_cols)
        for col in player_cols:
            unique_ids.update(df[col].dropna().astype(int).unique())
    return unique_ids

data_dir = 'split_data_parquet'
train_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('train_') and f.endswith('.parquet')])
val_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('val_') and f.endswith('.parquet')])
test_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('test_') and f.endswith('.parquet')])

player_columns = [
    "OFF_PLAYER1_ID", "OFF_PLAYER2_ID", "OFF_PLAYER3_ID", "OFF_PLAYER4_ID", "OFF_PLAYER5_ID",
    "DEF_PLAYER1_ID", "DEF_PLAYER2_ID", "DEF_PLAYER3_ID", "DEF_PLAYER4_ID", "DEF_PLAYER5_ID"
]
all_files = train_files + val_files + test_files
all_unique_ids = gather_unique_player_ids(all_files, player_columns)

unique_players = np.sort(list(all_unique_ids))
player_to_index = {p: i for i, p in enumerate(unique_players)}

result_df = generate_comparison_sheet(
    games_csv=games_csv,
    merged_csv=merged_csv,
    player_to_index=player_to_index,
    model_one_path=model_one_path,
    model_two_path=model_two_path
)

print(result_df)