from typing import Dict

import pandas as pd
import panel as pn

import holoview_plots
from plots import create_birthday_distribution_clock_diagram, create_population_distribution_map


def setup(data: Dict[str, pd.DataFrame], faergria_map_data_skip: bool):
    pn.extension()

    age_widget = pn.widgets.IntSlider(name="age", value=200, start=10, end=1000)
    sex_filter = pn.widgets.Select(name="sex", options=['Male', 'Female', 'Combined'])

    age_distribution = pn.bind(holoview_plots.create_age_distribution_focus, age=age_widget,
                               sex=sex_filter, **data)

    app = pn.template.MaterialTemplate(
        site="Panel",
        title="Getting Started App"
    )

    age_dist_card = pn.Card(
        pn.Column(
            pn.Row(sex_filter, age_widget),
            age_distribution
        ),
        title="Age distribution"
    )

    gender_dist_card = pn.Card(
        holoview_plots.create_gender_distribution(**data),
        title="Gender distribution"
    )

    birthday_dist_card = pn.Card(
        create_birthday_distribution_clock_diagram(**data),
        title="Birthday distribution"
    )

    cards = [
        age_dist_card,
        gender_dist_card,
        birthday_dist_card,
    ]

    if not faergria_map_data_skip:
        population_distribution_map = pn.Card(
            create_population_distribution_map(**data),
            title="Population distribution"
        )
        cards.append(population_distribution_map)

    return app, cards


def run(data: Dict[str, pd.DataFrame], faergria_map_data_skip: bool):
    app, cards = setup(data, faergria_map_data_skip)
    for card in cards:
        app.main.append(card)
    app.servable()
    pn.serve(app, port=8182, show=False)
