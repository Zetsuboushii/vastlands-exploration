import click
import decorators
import plots
from utils import get_dataframes


def _method_is_included(name: str):
    return (name.startswith("create_") and (
            decorators.included_method_names is None or name in decorators.included_method_names))



@click.command
@click.option("--faergria-map-url", "-u", default="http://localhost:1338")
@click.option("--faergria-map-data-skip", "-s", default=False, is_flag=True)

def main(faergria_map_url: str, faergria_map_data_skip: bool):
    data = get_dataframes(faergria_map_url, faergria_map_data_skip)
    plot_gen_methods = filter(_method_is_included, dir(plots))
    faergria_map_dependend_methods = ["create_population_distribution_map"]
    for method_name in plot_gen_methods:
        if faergria_map_data_skip and method_name in faergria_map_dependend_methods:
            continue
        method = getattr(plots, method_name)
        method(**data)

if __name__ == '__main__':
    main()
