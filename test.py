from matplotlib.figure import Figure
import panel as pn
from panel.pane import Matplotlib

pn.extension("ipywidgets")

fig1 = Figure()
ax1 = fig1.subplots()
ax1.plot([1, 2, 3], [1, 2, 3])

fig2 = Figure()
ax2 = fig2.subplots()
ax2.plot([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7])

pane1 = Matplotlib(fig1, interactive=True, dpi=144, tight=True)
pane2 = Matplotlib(fig2, interactive=True, dpi=144, tight=True)
pn.Tabs(
    ("title 1", pane1),
    ("title 2", pane2)
).show()
