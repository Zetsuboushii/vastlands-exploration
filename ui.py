import panel as pn
import plots
from utils import get_dataframes
from panel.pane import Matplotlib
import holoview_plots

pn.extension()

age_widget = pn.widgets.IntSlider(name="age", value=200, start=10, end=1000)

data = get_dataframes("", True)
cup_rating = pn.bind(holoview_plots.create_age_distribution_focus, age=age_widget, **data)

app = pn.template.MaterialTemplate(
    site="Panel",
    title="Getting Started App"
)

demo_card = pn.Card(
    cup_rating,
    title="Demo card"
)

app.main.append(
    pn.Column(
        age_widget,
        cup_rating
    )
)

app.servable()

if __name__ == "__main__":
    pn.serve(app)
