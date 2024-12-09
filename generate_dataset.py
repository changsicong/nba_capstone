from nba_api.stats.endpoints import leaguegamefinder
import nba_on_court as noc
from nba_api.stats.endpoints import playbyplayv2
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the event message types that matter
FIELD_GOAL_MADE = 1
FIELD_GOAL_MISSED = 2
FREE_THROW_ATTEMPT = 3
REBOUND = 4
TURNOVER = 5

valid_event_types = {FIELD_GOAL_MADE, FIELD_GOAL_MISSED, FREE_THROW_ATTEMPT, REBOUND, TURNOVER}

columns = [
    "OFF_PLAYER1_ID", "OFF_PLAYER2_ID", "OFF_PLAYER3_ID", "OFF_PLAYER4_ID", "OFF_PLAYER5_ID",
    "DEF_PLAYER1_ID", "DEF_PLAYER2_ID", "DEF_PLAYER3_ID", "DEF_PLAYER4_ID", "DEF_PLAYER5_ID",
    "OUTCOME",         # One of the 12 categories described (e.g., "2PT_ASSIST", "MISSED_BLOCKED", etc.)
    "SECOND_CHANCE",    # 0 or 1
    "SHOOTER_ID",       # ID of the player who took the shot (if applicable)
    "ASSISTER_ID",      # ID of the player who assisted (if applicable)
    "BLOCKER_ID",       # ID of the player who blocked the shot (if applicable)
    "STEALER_ID",       # ID of the player who stole the ball (if applicable)
    "REBOUNDER_ID" ,    # ID of the player who rebound the ball (if applicable)
    "TURNOVER_ID"      # ID of the player who turnover the ball (if applicable)
]

def generate_single_game_data(game_id):
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id).play_by_play.get_data_frame()
    pbp = noc.players_on_court(pbp)

    filtered_pbp = pbp[pbp['EVENTMSGTYPE'].isin(valid_event_types)].copy()
    filtered_pbp = filtered_pbp.reset_index(drop=True)

    # player_columns = [
    #     "AWAY_PLAYER1", "AWAY_PLAYER2", "AWAY_PLAYER3", "AWAY_PLAYER4", "AWAY_PLAYER5",
    #     "HOME_PLAYER1", "HOME_PLAYER2", "HOME_PLAYER3", "HOME_PLAYER4", "HOME_PLAYER5"
    # ]
    # # Filter rows so that all on-court players are in player_id_set
    # mask = filtered_pbp[player_columns].isin(all_players).all(axis=1)
    # filtered_pbp = filtered_pbp[mask].copy()

    # Now we have only the rows we care about.
    # We can iterate through these rows and handle each event type as needed.

    processed_pbp = pd.DataFrame(columns=columns)
    rows = []

    def get_off_def_players(row):
        away_players = [row['AWAY_PLAYER1'], row['AWAY_PLAYER2'], row['AWAY_PLAYER3'], row['AWAY_PLAYER4'], row['AWAY_PLAYER5']]
        home_players = [row['HOME_PLAYER1'], row['HOME_PLAYER2'], row['HOME_PLAYER3'], row['HOME_PLAYER4'], row['HOME_PLAYER5']]
        
        if row['PERSON1TYPE'] == 5:
            # Away is offense
            off_players = away_players
            def_players = home_players
        elif row['PERSON1TYPE'] == 4:
            # Home is offense
            off_players = home_players
            def_players = away_players
        else:
            # Unexpected case
            off_players = [np.nan]*5
            def_players = [np.nan]*5
        
        return off_players, def_players

    def check_and_one(pbp, current_idx, shooter_id):
        # Look ahead for next free throw attempt by the same shooter
        next_events = pbp.iloc[current_idx+1:]
        for i, r in next_events.iterrows():
            if r['EVENTMSGTYPE'] == FREE_THROW_ATTEMPT and r['PLAYER1_ID'] == shooter_id:
                # Check if made free throw
                desc = (r['HOMEDESCRIPTION'] if pd.notnull(r['HOMEDESCRIPTION']) else '') + \
                    (r['VISITORDESCRIPTION'] if pd.notnull(r['VISITORDESCRIPTION']) else '')
                if "MISS" not in desc.upper():
                    # FT made => and-one
                    return True
                else:
                    return False
        return False

    for idx, row in filtered_pbp.iterrows():
        event_type = row['EVENTMSGTYPE']
        off_players, def_players = get_off_def_players(row)
        
        # Switch-like structure to handle cases
        if event_type == FIELD_GOAL_MADE:
            # Determine if 2pt or 3pt
            desc = (row['HOMEDESCRIPTION'] if pd.notnull(row['HOMEDESCRIPTION']) else '') + \
                (row['VISITORDESCRIPTION'] if pd.notnull(row['VISITORDESCRIPTION']) else '')
            desc_upper = desc.upper()
            is_3pt = "3PT" in desc_upper

            # Determine assisted or unassisted
            assisted = (row['PERSON2TYPE'] == 4 or row['PERSON2TYPE'] == 5)

            # Base outcome mapping
            # 0: 2-point unassisted
            # 1: 2-point assisted
            # 2: 3-point unassisted
            # 3: 3-point assisted
            if not is_3pt:
                base_outcome = 1 if assisted else 0
            else:
                base_outcome = 3 if assisted else 2

            shooter_id = row['PLAYER1_ID']
            assister_id = row['PLAYER2_ID'] if assisted else np.nan

            # Check for and-one
            if check_and_one(filtered_pbp, idx, shooter_id):
                # 4-point scenarios for both 2pt+1 or 3pt+1
                # If base_outcome in [0,2] => unassisted => outcome=4
                # If base_outcome in [1,3] => assisted => outcome=5
                if base_outcome in [0, 2]:
                    final_outcome = 4
                else:
                    final_outcome = 5
            else:
                final_outcome = base_outcome

            data = {
                "OFF_PLAYER1_ID": off_players[0],
                "OFF_PLAYER2_ID": off_players[1],
                "OFF_PLAYER3_ID": off_players[2],
                "OFF_PLAYER4_ID": off_players[3],
                "OFF_PLAYER5_ID": off_players[4],
                "DEF_PLAYER1_ID": def_players[0],
                "DEF_PLAYER2_ID": def_players[1],
                "DEF_PLAYER3_ID": def_players[2],
                "DEF_PLAYER4_ID": def_players[3],
                "DEF_PLAYER5_ID": def_players[4],
                "OUTCOME": final_outcome,
                "SECOND_CHANCE": 0, # Not computed here
                "SHOOTER_ID": shooter_id,
                "ASSISTER_ID": assister_id,
                "BLOCKER_ID": np.nan,
                "STEALER_ID": np.nan,
                "REBOUNDER_ID": np.nan,
                "TURNOVER_ID": np.nan
            }
            rows.append(data)

        elif event_type == FIELD_GOAL_MISSED:       
            # Check if blocked
            # If PERSON3TYPE and PLAYER3_ID indicate a block, outcome=6 (blocked), else 7 (unblocked)
            blocked = False
            blocker_id = np.nan
            if row['PERSON3TYPE'] in [4,5] and pd.notnull(row['PLAYER3_ID']) and row['PLAYER3_ID'] != 0:
                # ADDITION: Check if PLAYER3_NAME is present and not empty
                if pd.isnull(row['PLAYER3_NAME']) or row['PLAYER3_NAME'].strip() == '':
                    # If no player name, treat as not blocked
                    blocked = False
                    blocker_id = np.nan
                else:
                    blocked = True
                    blocker_id = row['PLAYER3_ID']
            
            if blocked:
                final_outcome = 6  # Missed (blocked)
            else:
                final_outcome = 7  # Missed (unblocked)
            
            shooter_id = row['PLAYER1_ID']
            
            # Now find the rebound after this miss
            # Scan subsequent events for the next rebound (EVENTMSGTYPE=4)
            rebounder_id = np.nan
            second_chance = 0
            
            next_events = filtered_pbp.iloc[idx+1:]
            
            for i, r in next_events.iterrows():
                if r['EVENTMSGTYPE'] == REBOUND:
                    # ADDITION: Check if PLAYER1_NAME is present and not empty
                    if pd.isnull(r['PLAYER1_NAME']) or r['PLAYER1_NAME'].strip() == '':
                        # If no player name, this is a team rebound, skip
                        continue
                    # Found the rebound with a player name
                    rebounder_id = r['PLAYER1_ID']
                    # Offensive or defensive?
                    if rebounder_id in off_players:
                        second_chance = 1
                    else:
                        second_chance = 0
                    break
            
            data = {
                "OFF_PLAYER1_ID": off_players[0],
                "OFF_PLAYER2_ID": off_players[1],
                "OFF_PLAYER3_ID": off_players[2],
                "OFF_PLAYER4_ID": off_players[3],
                "OFF_PLAYER5_ID": off_players[4],
                "DEF_PLAYER1_ID": def_players[0],
                "DEF_PLAYER2_ID": def_players[1],
                "DEF_PLAYER3_ID": def_players[2],
                "DEF_PLAYER4_ID": def_players[3],
                "DEF_PLAYER5_ID": def_players[4],
                "OUTCOME": final_outcome,
                "SECOND_CHANCE": second_chance,
                "SHOOTER_ID": shooter_id,
                "ASSISTER_ID": np.nan,   # Not applicable for missed shots
                "BLOCKER_ID": blocker_id,
                "STEALER_ID": np.nan,
                "REBOUNDER_ID": rebounder_id,
                "TURNOVER_ID": np.nan
            }
            rows.append(data)

        elif event_type == FREE_THROW_ATTEMPT:
            # Parse the number of attempts from the first FT in the sequence
            shooter_id = row['PLAYER1_ID']
            desc = (row['HOMEDESCRIPTION'] if pd.notnull(row['HOMEDESCRIPTION']) else '') + \
                (row['VISITORDESCRIPTION'] if pd.notnull(row['VISITORDESCRIPTION']) else '')
            desc_up = desc.upper()

            match = re.search(r'FREE THROW.*?(\d+)\s+OF\s+(\d+)', desc_up)
            if not match:
                continue
            total_fts = int(match.group(2))
            if total_fts not in [2, 3]:
                # Not a scenario we care about
                continue

            # We'll accumulate all consecutive FTs for this shooter
            made_count = 0
            final_attempt_idx = None

            j = idx
            # Use a while loop to iterate through consecutive free throws of the same shooter
            while j in filtered_pbp.index:
                fr = filtered_pbp.loc[j]
                if fr['EVENTMSGTYPE'] != FREE_THROW_ATTEMPT or fr['PLAYER1_ID'] != shooter_id:
                    # Different event or different shooter means end of FT sequence
                    break

                f_desc = (fr['HOMEDESCRIPTION'] if pd.notnull(fr['HOMEDESCRIPTION']) else '') + \
                        (fr['VISITORDESCRIPTION'] if pd.notnull(fr['VISITORDESCRIPTION']) else '')
                f_desc_up = f_desc.upper()
                m2 = re.search(r'FREE THROW.*?(\d+)\s+OF\s+(\d+)', f_desc_up)
                if not m2:
                    # Can't parse this attempt, stop
                    break

                c_ft_num = int(m2.group(1))
                t_fts = int(m2.group(2))
                if t_fts not in [2, 3]:
                    # Different scenario than expected
                    break

                # Check if made
                if "MISS" not in f_desc_up:
                    made_count += 1

                if c_ft_num == t_fts:
                    # final attempt found
                    final_attempt_idx = j
                    break

                j += 1

            if final_attempt_idx is None:
                # No final attempt found
                continue

            # Determine outcome based on made_count
            # 0 made -> 10
            # 1 made -> 11
            # 2 made -> 12
            # 3 made -> 13
            if made_count == 0:
                final_outcome = 10
            elif made_count == 1:
                final_outcome = 11
            elif made_count == 2:
                final_outcome = 12
            elif made_count == 3:
                final_outcome = 13
            else:
                # Unexpected
                continue

            # Check if final attempt missed
            second_chance = 0
            rebounder_id = np.nan
            if made_count < total_fts:
                # last attempt missed
                next_after_ft = filtered_pbp.loc[final_attempt_idx+1:]
                for k, rb in next_after_ft.iterrows():
                    if rb['EVENTMSGTYPE'] == REBOUND:
                        if pd.isnull(rb['PLAYER1_NAME']) or rb['PLAYER1_NAME'].strip() == '':
                            # team rebound, skip
                            continue
                        rebounder_id = rb['PLAYER1_ID']
                        # If offensive rebound
                        if rebounder_id in off_players:
                            second_chance = 1
                        break

            data = {
                "OFF_PLAYER1_ID": off_players[0],
                "OFF_PLAYER2_ID": off_players[1],
                "OFF_PLAYER3_ID": off_players[2],
                "OFF_PLAYER4_ID": off_players[3],
                "OFF_PLAYER5_ID": off_players[4],
                "DEF_PLAYER1_ID": def_players[0],
                "DEF_PLAYER2_ID": def_players[1],
                "DEF_PLAYER3_ID": def_players[2],
                "DEF_PLAYER4_ID": def_players[3],
                "DEF_PLAYER5_ID": def_players[4],
                "OUTCOME": final_outcome,
                "SECOND_CHANCE": second_chance,
                "SHOOTER_ID": shooter_id,
                "ASSISTER_ID": np.nan,
                "BLOCKER_ID": np.nan,
                "STEALER_ID": np.nan,
                "REBOUNDER_ID": rebounder_id,
                "TURNOVER_ID": np.nan
            }
            rows.append(data)
        elif event_type == TURNOVER:
            # Filter out team turnover: if PLAYER1_NAME is empty, it's a team turnover
            if pd.isnull(row['PLAYER1_NAME']) or row['PLAYER1_NAME'].strip() == '':
                continue
            else:
                # We have a player turnover
                # 3. Document who steals: If there's PLAYER2_ID and a player name => stolen
                # If pd.notnull(PLAYER2_ID) and PERSON2TYPE in [4,5], that indicates a steal
                if pd.notnull(row['PLAYER2_ID']) and row['PLAYER2_ID'] != 0 and row['PERSON2TYPE'] in [4,5]:
                    final_outcome = 8  # Turnover (stolen)
                    stealer_id = row['PLAYER2_ID']
                else:
                    final_outcome = 9  # Turnover (unforced)
                    stealer_id = np.nan

            turnover_player_id = row['PLAYER1_ID']

            data = {
                "OFF_PLAYER1_ID": off_players[0],
                "OFF_PLAYER2_ID": off_players[1],
                "OFF_PLAYER3_ID": off_players[2],
                "OFF_PLAYER4_ID": off_players[3],
                "OFF_PLAYER5_ID": off_players[4],
                "DEF_PLAYER1_ID": def_players[0],
                "DEF_PLAYER2_ID": def_players[1],
                "DEF_PLAYER3_ID": def_players[2],
                "DEF_PLAYER4_ID": def_players[3],
                "DEF_PLAYER5_ID": def_players[4],
                "OUTCOME": final_outcome,
                "SECOND_CHANCE": 0,  # Not applicable for turnovers
                "SHOOTER_ID": np.nan,  # Not applicable
                "ASSISTER_ID": np.nan, # Not applicable
                "BLOCKER_ID": np.nan,  # Not applicable
                "STEALER_ID": stealer_id,
                "REBOUNDER_ID": np.nan,
                "TURNOVER_ID": turnover_player_id
            }
            rows.append(data)
        else:
            # Should not reach here because we filtered out other types
            pass
    output_folder = "dataset"
    os.makedirs(output_folder, exist_ok=True)
    processed_pbp = pd.DataFrame(rows, columns=columns)
    processed_pbp.to_csv(os.path.join(output_folder, f"{game_id}.csv"))

# generate_single_game_data('0022100001')

seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20']
# seasons = ['2023-24']

# Initialize an empty list to store game IDs
all_game_ids = []

# Loop through each season and collect game IDs
for season in seasons:
    # Specify 'Regular Season' directly
    game_finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    games = game_finder.get_data_frames()[0]
    game_ids = games['GAME_ID'].unique()
    all_game_ids.extend(game_ids)

num_workers = 9

failed_games = []  # List to keep track of game IDs that failed
num_workers = 8  # Example value, adjust as needed

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(generate_single_game_data, game_id): game_id for game_id in all_game_ids}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Games"):
        game_id = futures[future]
        try:
            # If generate_single_game_data doesn't return a value, .result() is None
            future.result()
        except Exception as e:
            print(f"Game {game_id} generated an exception: {e}")
            failed_games.append(game_id)

# After the loop, failed_games will contain all game IDs that raised exceptions.
print("Failed games:", failed_games)