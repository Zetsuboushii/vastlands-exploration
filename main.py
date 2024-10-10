from api import get_all_data, get_character_df, get_races_df, get_places_df, get_actions_df, get_enemies_df
from utils import set_current_date, get_birthdays_grouped_by_month, get_next_birthday
from plots import create_gender_distribution, create_age_distribution_200y_focus, \
    create_age_distribution_normalized, create_birthday_data_presence_pie_chart, \
    create_birthday_distribution_clock_diagram, create_weakness_distribution_pie_chart, \
    create_resistance_distribution_pie_chart, create_ability_score_distribution_plot, \
    create_immunities_distribution_pie_chart, \
    create_combined_pie_charts, create_stats_distribution_plot, create_grouping_pie_chart, \
    create_character_class_bar_chart


def setup():
    characters_data, general_data, races_data, places_data, actions_data, enemies_data = get_all_data()
    set_current_date(general_data)
    return characters_data, general_data, races_data, places_data, actions_data, enemies_data

def main():
    characters_data, general_data, races_data, places_data, actions_data, enemies_data = setup()
    df_characters = get_character_df(characters_data)
    df_races = get_races_df(races_data)
    df_places = get_places_df(places_data)
    df_actions = get_actions_df(actions_data)
    df_enemies = get_enemies_df(enemies_data)
    create_gender_distribution(df_characters)
    create_age_distribution_200y_focus(df_characters)
    create_age_distribution_normalized(df_characters, df_races)
    create_birthday_data_presence_pie_chart(df_characters)
    create_birthday_distribution_clock_diagram(df_characters)
    create_combined_pie_charts(df_enemies)
    create_ability_score_distribution_plot(df_enemies)
    create_stats_distribution_plot(df_enemies)
    get_birthdays_grouped_by_month(df_characters)
    get_next_birthday(df_characters)
    create_grouping_pie_chart(df_characters, "character_class", "Character class distribution", legend=False)
    create_character_class_bar_chart(df_characters)


if __name__ == '__main__':
    main()
