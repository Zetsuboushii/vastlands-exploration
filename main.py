import click
from matplotlib import pyplot as plt

import decorators
import plots
import ui
from utils import get_dataframes


def _method_is_included(name: str):
    return (name.startswith("create_") and (
            decorators.included_method_names is None or name in decorators.included_method_names))


@click.command("plot")
@click.pass_context
def render_plots(ctx):
    faergria_map_data_skip = ctx.obj['faergria_map_data_skip']  # Get the shared option
    data = ctx.obj["data"]
    faergria_map_dependend_methods = ["create_population_distribution_map"]
    plot_gen_methods = filter(_method_is_included, dir(plots))
    faergria_map_dependend_methods = ["create_population_distribution_map"]
    for method_name in plot_gen_methods:
        if faergria_map_data_skip and method_name in faergria_map_dependend_methods:
            continue
        method = getattr(plots, method_name)
        return_value = method(**data)
        if isinstance(return_value, plt.Figure):
            return_value.show()

@click.command
@click.pass_context
def serve(ctx):
    faergria_map_data_skip = ctx.obj['faergria_map_data_skip']
    data = ctx.obj["data"]
    ui.run(data, faergria_map_data_skip)


@click.group()
@click.option("--faergria-map-url", "-u", default="http://localhost:1338")
@click.option("--faergria-map-data-skip", "-s", default=False, is_flag=True)
@click.pass_context
def main(ctx, faergria_map_url: str, faergria_map_data_skip: bool):
    ctx.ensure_object(dict)
    ctx.obj["faergria_map_url"] = faergria_map_url
    ctx.obj["faergria_map_data_skip"] = faergria_map_data_skip
    data = get_dataframes(faergria_map_url, faergria_map_data_skip)
    ctx.obj["data"] = data


main.add_command(render_plots)
main.add_command(serve)

if __name__ == '__main__':
    main()
