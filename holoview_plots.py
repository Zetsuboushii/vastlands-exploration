import numpy as np
import holoviews as hv
from holoviews import opts
from bokeh.palettes import Category10

from utils import calculate_age


def create_age_distribution_focus(characters, age, **kwargs):
    # Calculate the age of each character
    characters['age'] = characters['birthday'].apply(calculate_age)

    # Filter out characters with NaN ages
    df_age = characters.dropna(subset=['age'])

    # Filter characters based on the years_focus parameter
    df_age_under_focus = df_age[df_age['age'] <= age]

    # Separate by gender
    male_characters = df_age_under_focus[df_age_under_focus['sex'] == 'm']
    female_characters = df_age_under_focus[df_age_under_focus['sex'] == 'w']
    min_age = df_age_under_focus["age"].min()
    max_age = df_age_under_focus["age"].max()
    import pandas as pd

    # Create histograms using Holoviews
    bin_count = 20
    bin_edges = np.linspace(min_age, max_age, bin_count)
    hist_male = hv.Histogram(np.histogram(male_characters['age'], bins=bin_edges), label='Male')
    hist_female = hv.Histogram(np.histogram(female_characters['age'], bins=bin_edges), label='Female')

    # Combine both histograms into a single plot
    overlay = (hist_male * hist_female).opts(
        opts.Histogram(alpha=0.3,
                       line_color='black',
                       color=hv.Cycle(list(Category10[10])),
                       xlabel='Age',
                       ylabel='Number of Characters',
                       title=f'Distribution of Ages (Up to {age}) by Gender',
                       width=800,
                       height=600)
    )

    return overlay
