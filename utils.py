from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd

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
    no_birthday_set = df_characters[df_characters['birthday'].isna() | (df_characters['birthday'] == '')][
        'name'].tolist()
    grouped = \
        df_characters[df_characters['birthday'].notna() & (df_characters['birthday'] != '')].groupby('fantasy_month')[
            'character_info'].apply(list).reset_index()

    # Sort the grouped data based on the numerical month extracted
    grouped['month_number'] = grouped['fantasy_month'].map({v: k for k, v in fantasy_months.items()})
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
    import pandas as pd
    import json
    import os

    data_list = []
    root_path = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(root_path, 'data', 'tierlists')

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            basename = filename[len('tierlist_'):-len('.json')]
            try:
                author, sessionNr = basename.rsplit('_', 1)
            except ValueError:
                print(f"Filename {filename} does not match expected format 'tierlist_author_sessionNr.json'")
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entry = {
                'author': str(author),
                'sessionNr': int(sessionNr),
                'SS': data.get('SS', []),
                'S': data.get('S', []),
                'A': data.get('A', []),
                'B': data.get('B', []),
                'C': data.get('C', []),
                'D': data.get('D', [])
            }

            data_list.append(entry)

    df = pd.DataFrame(data_list)
    return df
