import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts
from holoviews.streams import Pipe
from pandas import DataFrame

from utils import calculate_age

hv.extension("bokeh")


def create_gender_distribution(characters: DataFrame, **kwargs):
    sex_counts = characters['sex'].value_counts()
    labels = sex_counts.index.tolist()
    sizes = sex_counts.values.tolist()

    data = pd.DataFrame({'Gender': labels, 'Count': sizes})

    histogram = hv.Bars(data, kdims='Gender', vdims='Count').opts(
        cmap="Category10",
        color=hv.dim('Gender'),
        width=500,
        height=500
    )
    return histogram


def create_age_distribution_focus(characters, age, sex, **kwargs):
    characters['age'] = characters['birthday'].apply(calculate_age)
    df_age = characters.dropna(subset=['age'])
    df_age_under_focus = df_age[df_age['age'] <= age]

    male_characters = df_age_under_focus[df_age_under_focus['sex'] == 'm']
    female_characters = df_age_under_focus[df_age_under_focus['sex'] == 'w']
    min_age = df_age_under_focus["age"].min()
    max_age = df_age_under_focus["age"].max()

    bin_count = 20
    bin_edges = np.linspace(min_age, max_age, bin_count)

    def map_characters_to_bins(characters, bin_edges):
        hist, _ = np.histogram(characters['age'], bins=bin_edges)
        char_bins = {
            i: characters[
                (characters['age'] >= bin_edges[i]) & (characters['age'] <= bin_edges[i + 1])][
                'name'].tolist()
            for i in range(len(bin_edges) - 1)
        }
        return hist, char_bins

    hist_male, male_char_bins = map_characters_to_bins(male_characters, bin_edges)
    hist_female, female_char_bins = map_characters_to_bins(female_characters, bin_edges)

    hist_male_plot = hv.Histogram((bin_edges, hist_male)).relabel('Male').opts(alpha=0.3,
                                                                               color='blue',
                                                                               tools=["hover"])
    hist_female_plot = hv.Histogram((bin_edges, hist_female)).relabel('Female').opts(alpha=0.3,
                                                                                     color='red',
                                                                                     tools=[
                                                                                         "hover"])

    combined_char_bins = {
        i: male_char_bins[i] + female_char_bins[i] for i in range(len(bin_edges) - 1)
    }

    def hover_callback(index_male=None, index_female=None):
        if not index_male and not index_female:
            return hv.Table({'Character Names': ["nothing selected"]}, ['Character Names'])
        selected_index = index_male[0] if index_male else index_female[0]
        data = {'Character Names': combined_char_bins.get(selected_index, ["nothing selected"])}
        return hv.Table(data, ['Character Names'])

    stream_male = hv.streams.Selection1D(source=hist_male_plot, index=[])
    stream_female = hv.streams.Selection1D(source=hist_female_plot, index=[])

    hover_map = hv.DynamicMap(hover_callback, streams=[stream_male.rename(index='index_male'),
                                                       stream_female.rename(index='index_female')])

    combined_hist = hist_male_plot * hist_female_plot

    plot_with_hover = combined_hist + hover_map

    plot_with_hover.opts(
        opts.Histogram(
            tools=['tap', 'hover'],
            xlabel='Age',
            ylabel='Number of Characters',
            width=800,
            height=600,
            title=f'Distribution of Ages (Up to {age}) by Gender',
            line_color='black',
        )
    )
    return plot_with_hover
