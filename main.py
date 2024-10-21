import click
import decorators
import plots
from api import get_all_data, get_df_from_endpoint_data, fetch_faergria_map, save_character_images
from entities.action import Action
from entities.character import Character
from entities.enemy import Enemy
from entities.marker import Marker
from entities.place import Place
from entities.race import Race
from utils import set_current_date, get_tierlist_df


def setup(faergria_map_url: str, faergria_map_data_skip):
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
        if key != "general_data"
    }
    save_character_images(dataframes["characters"])
    tierlist_df = get_tierlist_df()
    dataframes['tierlists'] = tierlist_df
    return dataframes


def _method_is_included(name: str):
    return (name.startswith("create_") and (
            decorators.included_method_names is None or name in decorators.included_method_names))



@click.command
@click.option("--faergria-map-url", "-u", default="http://localhost:1338")
@click.option("--faergria-map-data-skip", "-s", default=False, is_flag=True)

def main(faergria_map_url: str, faergria_map_data_skip: bool):
    data = setup(faergria_map_url,faergria_map_data_skip)
    plot_gen_methods = filter(_method_is_included, dir(plots))
    for method_name in plot_gen_methods:
        method = getattr(plots, method_name)
        method(**data)

if __name__ == '__main__':
    main()
