from datetime import datetime

import pandas as pd

import mongo_connector

from api import get_all_data, get_df_from_endpoint_data, save_character_images
from entities.action import Action
from entities.character import Character
from entities.enemy import Enemy
from entities.marker import Marker
from entities.place import Place
from entities.race import Race

CURRENT_DATE = None

fantasy_months = {
    1: 'Eismond',
    2: 'Frostmond',
    3: 'Saatmond',
    4: 'Blütenmond',
    5: 'Wonnemond',
    6: 'Heumond',
    7: 'Sonnemond',
    8: 'Erntemond',
    9: 'Fruchtmond',
    10: 'Weinmond',
    11: 'Nebelmond',
    12: 'Julmond'
}


def extract_month_and_apply_fantasy_name(birthday):
    if pd.isna(birthday) or birthday == '':
        return None, None
    try:
        date_parts = birthday.split('.')
        month = int(date_parts[1])
        fantasy_month = fantasy_months.get(month, "Unknown Month")
        return fantasy_month, month
    except Exception as e:
        return None, None


def set_current_date(general_data):
    global CURRENT_DATE
    current_ingame_date = general_data.get('currentIngameDate', '')
    CURRENT_DATE = current_ingame_date


def calculate_age(birthday):
    current_year = int(CURRENT_DATE.split('-')[2])
    try:
        parts = birthday.split(".")
        birth_year = int(parts[2])
        return current_year - birth_year
    except (ValueError, IndexError, AttributeError):
        return None


def get_day_of_year(birthday: str):
    birthday_only_day_and_month = ".".join(birthday.split(".")[0:2])
    date = datetime.strptime(birthday_only_day_and_month, "%d.%m")
    return date.timetuple().tm_yday


def get_birthdays_grouped_by_month(df_characters):
    # Apply the extraction function to get the fantasy month and numerical month
    df_characters[['fantasy_month', 'month_number']] = df_characters['birthday'].apply(
        lambda x: pd.Series(extract_month_and_apply_fantasy_name(x)))

    # Combine character name and birthday for output
    df_characters['character_info'] = df_characters.apply(
        lambda row: f"{row['name']} ({row['birthday']})", axis=1)

    # Group by 'fantasy_month', keeping track of those with no birthday set
    no_birthday_set = \
    df_characters[df_characters['birthday'].isna() | (df_characters['birthday'] == '')][
        'name'].tolist()
    grouped = \
        df_characters[
            df_characters['birthday'].notna() & (df_characters['birthday'] != '')].groupby(
            'fantasy_month')[
            'character_info'].apply(list).reset_index()

    # Sort the grouped data based on the numerical month extracted
    grouped['month_number'] = grouped['fantasy_month'].map(
        {v: k for k, v in fantasy_months.items()})
    grouped = grouped.sort_values(by='month_number')

    # Output the results in the correct order
    for index, row in grouped.iterrows():
        if row['fantasy_month']:
            print(f"{row['fantasy_month']}: {', '.join(row['character_info'])}")

    if no_birthday_set:
        print("\nNo birthday set: " + ', '.join(no_birthday_set))


def get_next_birthday(df_characters):
    current_day, current_month, current_year = map(int, CURRENT_DATE.split('-'))
    current_date = datetime(year=1, month=current_month, day=current_day)

    upcoming_birthday = None
    closest_delta = None

    for index, row in df_characters.iterrows():
        try:
            day, month, year = map(int, row['birthday'].split('.'))
            birthday_this_year = datetime(year=1, month=month, day=day)

            if birthday_this_year < current_date:
                birthday_this_year = datetime(year=2, month=month, day=day)

            delta = (birthday_this_year - current_date).days

            if closest_delta is None or delta < closest_delta:
                closest_delta = delta
                upcoming_birthday = row
        except (ValueError, AttributeError):
            continue

    if upcoming_birthday is not None:
        print(
            f"The next upcoming birthday is: {upcoming_birthday['name']} {upcoming_birthday['surname']} born on {upcoming_birthday['birthday']}")
    else:
        print("No upcoming birthdays found.")


def get_tierlist_df():
    tierlists = mongo_connector.fetch_tierlists()
    return pd.DataFrame(tierlists)


def get_evaluated_tierlist_df(tierlists):
    filtered_tierlists = tierlists.loc[tierlists.groupby('author')['sessionNr'].idxmax()]
    list_of_characters = pd.DataFrame(columns=['sum_value', 'appearance', 'all_values'])
    list_of_characters.index.name = 'name'
    rank_of_characters = []
    tiers_mapping = {'D': 0, 'C': 1, 'B': 2, 'A': 3, 'S': 4, 'SS': 5}
    value_to_tier = {v: k for k, v in tiers_mapping.items()}

    def update_character(name, value_increment):
        if name not in list_of_characters.index:
            list_of_characters.loc[name] = {'sum_value': value_increment, 'appearance': 1,
                                            'all_values': [value_increment]}
        else:
            list_of_characters.loc[name, 'sum_value'] += value_increment
            list_of_characters.loc[name, 'appearance'] += 1
            list_of_characters.at[name, 'all_values'].append(value_increment)

    for index, row in filtered_tierlists.iterrows():
        for tier in tiers_mapping:
            current_tier_name_list = row.get(tier, [])
            if isinstance(current_tier_name_list, list):
                for name in current_tier_name_list:
                    update_character(name, tiers_mapping[tier])
    for index, row in list_of_characters.iterrows():
        average_rating = row['sum_value'] / row['appearance']
        std_dev = pd.Series(row['all_values']).std()
        rank_of_characters.append((index, average_rating, std_dev))
    rank_df = pd.DataFrame(rank_of_characters, columns=['name', 'average_rating', 'std_dev'])
    rank_df['rounded_rating'] = rank_df['average_rating'].round().astype(int)
    rank_df['tier'] = rank_df['rounded_rating'].map(value_to_tier)
    rank_df = rank_df[rank_df['rounded_rating'].between(0, 5)]
    rank_df = rank_df.sort_values(by='average_rating', ascending=True)
    return rank_df


def get_joined_tierlists_characters_df(characters: pd.DataFrame, tierlists: pd.DataFrame):
    characters['name'] = characters['name'].str.lower()
    rating_df = get_evaluated_tierlist_df(tierlists)
    combined_df = pd.merge(rating_df, characters, on='name', how='inner')
    return combined_df


def get_dataframes(faergria_map_url: str, faergria_map_data_skip):
    data = get_all_data(faergria_map_url, faergria_map_data_skip)
    set_current_date(data["general_data"])
    classes = {
        "actions_data": Action,
        "characters_data": Character,
        "enemies_data": Enemy,
        "places_data": Place,
        "races_data": Race,
        "markers_data": Marker
    }
    # "Calculate" dataframes for the respective object data from the endpoints,
    # but assign keys without "_data"
    dataframes = {
        key[:-5]: get_df_from_endpoint_data(endpoint_data, classes[key]) for key, endpoint_data
        in data.items()
        if key not in ["general_data", "effect_data"]
    }
    save_character_images(dataframes["characters"])
    tierlist_df = get_tierlist_df()
    dataframes['tierlists'] = tierlist_df
    return dataframes
