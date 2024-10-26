import panel as pn
import plots
from utils import get_dataframes
from panel.pane import Matplotlib
import holoview_plots

pn.extension()

age_widget = pn.widgets.IntSlider(name="age", value=200, start=10, end=1000)
sex_filter = pn.widgets.Select(name="sex", options=['Male', 'Female', 'Combined'])

data = get_dataframes("", True)
cup_rating = pn.bind(holoview_plots.create_age_distribution_focus, age=age_widget, sex = sex_filter, **data)

app = pn.template.MaterialTemplate(
    site="Panel",
    title="Getting Started App"
)

demo_card = pn.Card(
    pn.Column(
        pn.Row(sex_filter, age_widget),
        cup_rating
    ),
    title="Demo card"
)

app.main.append(
    demo_card
)

app.servable()

if __name__ == "__main__":
    pn.serve(app)
