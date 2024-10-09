from api import get_all_data, get_character_df, get_races_df
from utils import set_current_date
from plots import create_gender_distribution, create_age_distribution_200y_focus, create_age_distribution_normalized

def setup():
    characters_data, general_data, races_data, places_data = get_all_data()
    set_current_date(general_data)
    return characters_data, general_data, races_data, places_data

def main():
    characters_data, general_data, races_data, places_data = setup()
    df_characters = get_character_df(characters_data)
    df_races = get_races_df(races_data)
    create_gender_distribution(df_characters)
    create_age_distribution_200y_focus(df_characters)
    create_age_distribution_normalized(df_characters, df_races)

if __name__ == '__main__':
    main()
