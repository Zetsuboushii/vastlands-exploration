import plots
from api import get_all_data, get_df_from_endpoint_data
from entities.action import Action
from entities.character import Character
from entities.enemy import Enemy
from entities.place import Place
from entities.race import Race
from utils import set_current_date


def setup():
    data = get_all_data()
    set_current_date(data["general_data"])
    classes = {
        "actions_data": Action,
        "characters_data": Character,
        "enemies_data": Enemy,
        "places_data": Place,
        "races_data": Race
    }
    # "Calculate" dataframes for the respective object data from the endpoints,
    # but assign keys without "_data"
    dataframes = {
        key[:-5]: get_df_from_endpoint_data(endpoint_data, classes[key]) for key, endpoint_data
        in data.items()
        if key != "general_data"
    }
    return dataframes


def main():
    data = setup()
    plot_gen_methods = filter(lambda method: method.startswith("create_"), dir(plots))
    for method_name in plot_gen_methods:
        method = getattr(plots, method_name)
        method(**data)


if __name__ == '__main__':
    main()
