import ast
import os
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from adjustText import adjust_text
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import stats

from decorators import include_plot
from utils import calculate_age, get_day_of_year, get_evaluated_tierlist_df, \
    get_joined_tierlists_characters_df


def create_gender_distribution(characters, **kwargs):
    sex_counts = characters['sex'].value_counts()
    labels = sex_counts.index.tolist()
    sizes = sex_counts.values.tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribution of Male and Female Characters')
    return fig


def create_age_distribution_200y_focus(characters, **kwargs):
    characters['age'] = characters['birthday'].apply(calculate_age)
    df_age = characters.dropna(subset=['age'])
    df_age_under_200 = df_age[df_age['age'] <= 200]

    male_characters = df_age_under_200[df_age_under_200['sex'] == 'm']
    female_characters = df_age_under_200[df_age_under_200['sex'] == 'w']

    fig, ax = plt.subplots(figsize=(8, 6))

    min_age = df_age_under_200["age"].min()
    max_age = df_age_under_200["age"].max()
    bin_edges = np.linspace(min_age, max_age, 20)

    ax.hist(male_characters['age'], bins=bin_edges, edgecolor='black', alpha=0.5, label='Male',
            color='blue')
    ax.hist(female_characters['age'], bins=bin_edges, edgecolor='black', alpha=0.5, label='Female',
            color='pink')

    plt.xticks(range(0, 201, 10))
    ax.set_title('Distribution of Ages (Up to 200) by Gender')
    plt.xlabel('Age')
    plt.ylabel('Number of Characters')
    ax.legend(loc='upper right')
    return fig


def create_age_distribution_normalized(characters, races, **kwargs):
    characters['age'] = characters['birthday'].apply(calculate_age)
    characters = characters.dropna(subset=['age'])
    races['ageavg'] = pd.to_numeric(races['ageavg'], errors='coerce')
    races = races.dropna(subset=['ageavg'])
    races["norm_metric"] = races["ageavg"] / 100

    merge_df = pd.merge(characters, races, left_on="race", right_on="name", how="inner")
    merge_df["normed_age"] = merge_df["age"] / merge_df["norm_metric"]

    male_characters = merge_df[merge_df['sex'] == 'm']
    female_characters = merge_df[merge_df['sex'] == 'w']

    fig, ax = plt.subplots(figsize=(8, 6))

    min_age = merge_df["normed_age"].min()
    max_age = merge_df["normed_age"].max()
    bin_edges = np.linspace(min_age, max_age, 20)

    ax.hist(male_characters["normed_age"], bins=bin_edges, edgecolor='black', alpha=0.5,
            label='Male',
            color='blue')
    ax.hist(female_characters["normed_age"], bins=bin_edges, edgecolor='black', alpha=0.5,
            label='Female',
            color='pink')

    plt.xticks(range(0, 201, 10))
    ax.set_title('Distribution of Ages normalized on human age by gender')
    plt.xlabel('Age')
    plt.ylabel('Number of Characters')
    ax.legend(loc='upper right')
    return fig


def create_birthday_data_presence_pie_chart(characters: pd.DataFrame, **kwargs):
    birthday_counts = (characters["birthday"] != "").value_counts()
    labels = ["Given", "Not given"]
    sizes = [birthday_counts[True], birthday_counts[False]]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title("Characters with birthdays assigned or not")
    return fig


def create_birthday_distribution_clock_diagram(characters: pd.DataFrame, **kwargs):
    # Birthdays that are actually set
    characters_with_birthdays = characters[characters["birthday"] != ""]

    unique_races = characters["race"].unique()
    color_map = cm.get_cmap("Set2", len(unique_races))
    colors = {race: color_map(i) for i, race in enumerate(unique_races)}

    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": "polar"})

    grouping = characters_with_birthdays.groupby("race")
    for race, race_df in grouping:
        set_birthdays = race_df["birthday"]
        days_of_the_year = set_birthdays.apply(get_day_of_year)
        ax.scatter(days_of_the_year, np.ones(len(race_df)) * .97, label=race, color=colors[race],
                   linewidths=0.2)

    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(range(1, 13))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(-np.pi / 2)

    ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), title="Rassen")
    plt.ylim(0, 1)

    ax.set_title("Birthday distribution by race")
    return fig


def create_combined_pie_charts(enemies: pd.DataFrame, min_percentage: float = 2.0, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    def group_and_plot(data, column, ax, title):
        # Explode the column into separate rows
        df_normalized = data.explode(column).reset_index(drop=True)

        # Group by the column and count occurrences
        group = df_normalized.groupby(column).size().sort_values(ascending=False)
        total = group.sum()

        # Calculate percentages and separate small groups
        percentages = (group / total) * 100
        large_groups = group[percentages >= min_percentage]
        other_group = group[percentages < min_percentage].sum()

        # Combine large groups and the "Other" category
        if other_group > 0:
            grouped_data = pd.concat([large_groups, pd.Series({'Other': other_group})])
        else:
            grouped_data = large_groups

        # Plot pie chart
        ax.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(title)

    # Plot Weaknesses
    group_and_plot(enemies, "weaknesses", axes[0], 'Distribution of Weaknesses')

    # Plot Resistances
    group_and_plot(enemies, "resistances", axes[1], 'Distribution of Resistances')

    # Plot Immunities
    group_and_plot(enemies, "immunities", axes[2], 'Distribution of Immunities')

    # Adjust layout
    plt.tight_layout()
    return fig


def create_ability_score_distribution_plot(enemies: pd.DataFrame, **kwargs):
    as_avg = enemies[["str", "dex", "con", "int", "wis", "cha"]].mean()
    enemies['sum_stats'] = enemies[['str', 'dex', 'con', 'int', 'wis', 'cha']].sum(axis=1)
    mean_sum_stats = enemies['sum_stats'].mean().round(2)
    as_avg = as_avg.round(2)
    overall_avg = as_avg.mean().round(2)
    as_avg["overall_avg"] = overall_avg
    as_avg["as_avg"] = mean_sum_stats
    fig, ax = plt.subplots(figsize=(8, 6))
    as_avg.plot(kind='bar')
    ax.set_title('Average Ability Score of Enemies', fontsize=16)
    plt.xlabel('Ability Score', fontsize=14)
    plt.ylabel('Average Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    return fig


def create_stats_distribution_plot(enemies: pd.DataFrame, **kwargs):
    # TODO: add movement range, but this can be complicated because enemies can have multiple ways of moving ag. flying, swimming
    enemies["hp_ac_ratio"] = enemies["hp"] / enemies["ac"]
    stats_avg: pd.Series = enemies[["hp", "ac", "hp_ac_ratio"]].mean()
    stats_avg.round(2)
    fig, ax = plt.subplots()
    ax.bar(stats_avg.index, stats_avg.values)
    ax.set_title('Average Stats of Enemies', fontsize=16)
    plt.xlabel('Stats', fontsize=14)
    plt.ylabel('Average Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    max_value = stats_avg.max()
    ticks = np.around(np.linspace(0, max_value, num=12))
    plt.yticks(ticks)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    return fig


def create_character_class_bar_chart(characters: pd.DataFrame, **kwargs):
    character_classes = characters["character_class"].unique()
    sexes = characters["sex"].unique()
    n_classes = len(character_classes)
    n_sexes = len(sexes)
    group_width = 0.8
    bar_width = group_width / n_sexes
    fig, ax = plt.subplots()
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_sexes))

    for i, sex in enumerate(sexes):
        sex_data = characters[characters["sex"] == sex]
        class_counts = sex_data["character_class"].value_counts()
        x = np.arange(n_classes) + (i - (n_sexes - 1) / 2) * bar_width

        ax.bar(x, [class_counts.get(c, 0) for c in character_classes],
               width=bar_width, alpha=0.7, color=colors[i],
               label=f'Sex: {sex}', edgecolor='black')

    plt.xlabel('Character Class')
    plt.ylabel('Count')
    ax.set_title('Character Class Distribution by Sex')
    plt.xticks(np.arange(n_classes), character_classes, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()


def _create_grouping_pie_chart(df: pd.DataFrame, group_column: str, title: str, legend: bool = True,
                               legend_title: Optional[str] = None, min_percentage=.01, obj=None):
    using_default_plt = False
    if obj is None:
        obj = plt
        using_default_plt = True
    df[group_column] = df[group_column].apply(
        lambda value: value if value != "" else "No " + group_column)
    value_counts = df[group_column].value_counts()
    value_counts_sorted = value_counts.sort_values(ascending=False)

    total = value_counts_sorted.sum()
    percentages = value_counts_sorted / total

    main_slices = percentages[percentages >= min_percentage]
    small_slices = percentages[percentages < min_percentage]

    if not small_slices.empty:
        main_slices['Other'] = small_slices.sum()

    labels = main_slices.index.tolist()
    values = main_slices.values.tolist()

    base_colors = plt.cm.Set3.colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    wedges, texts, autotexts = obj.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       pctdistance=0.85, startangle=90)

    if using_default_plt:
        obj.title(title)
    else:
        obj.set_title(title)
    if legend:
        obj.legend(wedges, labels,
                   title=legend_title,
                   loc="center right"
                   )

    if using_default_plt:
        obj.tight_layout()
    return obj


def _create_subclasses_bar_chart(characters: pd.DataFrame, obj=None, show_no_subclass=True):
    using_default_plt = False
    if obj is None:
        obj = plt
        using_default_plt = True
    all_subclasses = [subclass for subclasses in characters["subclasses"] for subclass in
                      subclasses]
    subclass_counts = pd.Series(all_subclasses).value_counts()

    if show_no_subclass:
        no_subclass_count = (characters['subclasses'].str.len() == 0).sum()
        no_subclass = "No Subclass"
        subclass_counts[no_subclass] = no_subclass_count

    tick_settings = {"rotation": 45, "ha": "right"}
    if using_default_plt:
        obj.figure(figsize=(12, 8))
        obj.xticks(**tick_settings)

    obj.bar(subclass_counts.index, subclass_counts.values)

    title = 'Distribution of Character Subclasses'
    if using_default_plt:
        obj.title(title)
        obj.tight_layout()
    else:
        obj.set_title(title)
    return obj


def create_subclasses_bar_chart_with_no_subclass(characters: pd.DataFrame, **kwargs):
    return _create_subclasses_bar_chart(characters, obj=None, show_no_subclass=True)


def create_subclasses_bar_chart_without_no_subclass(characters: pd.DataFrame, **kwargs):
    return _create_subclasses_bar_chart(characters, obj=None, show_no_subclass=False)


def create_character_classes_combined_pie_charts(characters: pd.DataFrame, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    _create_grouping_pie_chart(characters, "character_class", "Character class distribution",
                               legend=False, obj=axes[0])
    _create_grouping_pie_chart(characters, "masterclass", "Masterclass distribution",
                               min_percentage=0, obj=axes[1])
    return fig


def create_relationship_web(characters: pd.DataFrame, **kwargs):
    # Initialize data structures
    df_characters = characters
    characters = {}
    relationships = []
    output_filename = "data/plots/character_relationships.svg"

    # Iterate over each row in the DataFrame
    for index, row in df_characters.iterrows():
        # Extract character information
        name = str(row['name']).strip()
        surname = str(row['surname']).strip()
        full_name = f"{name} {surname}".strip()
        title = str(row['title']).strip()
        race = str(row['race']).strip()
        sex = str(row['sex']).strip()
        birthday = str(row['birthday']).strip()
        height = str(row['height']).strip()
        functions = row['function']
        character_class = str(row['character_class']).strip()
        subclasses = row['subclasses']
        masterclass = str(row['masterclass']).strip()
        homes = row['homes']
        alignment = str(row['alignment']).strip()
        status = str(row['status']).strip()
        relationships_data = row['relationships']
        lover = str(row['lover']).strip()

        # Store character information
        characters[name] = {
            'name': name,
            'title': title,
            'race': race,
            'sex': sex,
            'alignment': alignment,
            'status': status,
        }

        # Initialize relationships_list
        relationships_list = []

        # Check if relationships_data is not None and not empty
        if relationships_data is not None:
            if isinstance(relationships_data, (list, tuple, np.ndarray, pd.Series)):
                if len(relationships_data) > 0:
                    relationships_list = relationships_data
            elif isinstance(relationships_data, (str, np.str_)):
                relationships_data = relationships_data.strip()
                if relationships_data not in ['', '[]', 'nan']:
                    try:
                        # Replace single quotes with double quotes if necessary
                        relationships_data = relationships_data.replace("'", '"')
                        relationships_list = ast.literal_eval(relationships_data)
                    except (ValueError, SyntaxError):
                        print(f"Error parsing relationships for {name}: {relationships_data}")
            else:
                # relationships_list remains empty
                pass

        # Process relationships_list
        for rel in relationships_list:
            if isinstance(rel, (list, tuple)) and len(rel) == 2:
                rel_name = rel[0].strip()
                rel_desc = rel[1].strip()
                relationships.append((name, rel_name, rel_desc))
            else:
                continue  # Skip invalid entries

        # Parse lover
        if lover not in ['', '[]', 'nan']:
            lover_name = lover.strip()
            relationships.append((name, lover_name, 'Partner'))

    # Build the graph
    G = nx.Graph()

    # Add nodes with attributes
    for char_name, char_attrs in characters.items():
        G.add_node(char_name, **char_attrs)

    # Add edges with relationship types
    for source, target, relation in relationships:
        # Only add edge if both nodes exist
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, relation=relation)
        else:
            print(f"One of the nodes '{source}' or '{target}' does not exist in the graph.")

    # Check if graph has nodes and edges
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("The graph is empty. Please check the data parsing logic.")
        return

    # Define edge colors based on relationship types
    edge_colors = {
        'Bruder': 'blue',
        'Schwester': 'pink',
        'Freund': 'green',
        'Freundin': 'green',
        'Partner': 'purple',
        'Vater': 'brown',
        'Mutter': 'brown',
        'Sohn': 'orange',
        'Tochter': 'orange',
        'Cousin': 'cyan',
        'Feind': 'black',
        'Herr': 'red',
        'Ziehvater': 'brown',
        'Ziehmutter': 'brown',
        'Ziehsohn': 'orange',
        # Add other specific relationship types as needed
    }
    default_color = 'gray'

    # Assign colors to edges based on relationship
    edge_colors_list = []
    edge_labels = {}
    for u, v in G.edges():
        relation = G[u][v].get('relation', '')
        # Extract relationship type from relation description
        relation_type = relation.split(',')[0].strip()
        color = edge_colors.get(relation_type, default_color)
        edge_colors_list.append(color)
        edge_labels[(u, v)] = relation  # Store full relation description for edge labels

    # Adjust the layout to spread out nodes
    pos = nx.spring_layout(G, k=2, iterations=200)

    # Increase figure size and resolution
    fig, ax = plt.subplots(figsize=(40, 40), dpi=300)

    # Scale node sizes based on degree (number of connections)
    node_sizes = [max(G.degree(n) * 50, 200) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')

    # Draw edges with reduced width
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors_list, width=0.5)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=4)

    # Reduce label font size and adjust label positions
    texts = []
    for node, (x, y) in pos.items():
        texts.append(plt.text(x, y, node, fontsize=6))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Get unique relationship types and their colors
    unique_relations = set([rel_type.split(',')[0].strip() for rel_type in edge_colors.keys()])
    legend_handles = []
    for rel_type in unique_relations:
        color = edge_colors.get(rel_type, default_color)
        legend_handles.append(mlines.Line2D([], [], color=color, label=rel_type))

    # Add the legend to the plot
    ax.legend(handles=legend_handles, title='Relationship Types', fontsize=12, title_fontsize=12)

    plt.axis('off')
    ax.set_title('Character Relationships Clustered by Type', fontsize=15)
    return fig


def create_population_distribution_map(places: pd.DataFrame, markers: pd.DataFrame, **kwargs):
    population_counts = []
    for index, place in places.iterrows():
        marker = markers[markers["name"] == place["name"]]
        if not marker.empty:
            mark = marker.iloc[0]
            population_counts.append(
                (place["name"], place["demography"], mark["lat"], mark["long"]))

    img = plt.imread(os.path.join("data", "faergria-map.png"))
    height, width, _ = img.shape
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, extent=(0, width, 0, height))

    for name, count, lat, long in population_counts:
        coeff = 2.9
        x, y = long * coeff, lat * coeff
        circle = plt.Circle((x, y), max(count / 500, 5), fill=False, edgecolor='red')
        ax.add_artist(circle)

        ax.annotate(f"{name}: {count}", (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a title
    ax.set_title("Population Distribution Map")
    fig.savefig(os.path.join("data", "population-distribution-map.png"), dpi=600,
                bbox_inches="tight")
    return fig


def offset_image(y, character_name, ax, target_height):
    img_path = os.path.join('data', 'images', f"{character_name.lower()}.png")

    if not os.path.exists(img_path):
        print(f"Image for {character_name} not found at {img_path}")
        return

    img = Image.open(img_path)
    img_width, img_height = img.size

    zoom_factor = target_height / img_height

    im = OffsetImage(img, zoom=zoom_factor)
    im.image.axes = ax

    scaled_width = img_width * zoom_factor
    x_offset = -scaled_width

    ab = AnnotationBbox(im, (0, y), xybox=(x_offset, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)


def create_height_distribution_chart(characters: pd.DataFrame, target_image_height=100,
                                     bar_spacing=1,
                                     aspect_ratio=0.05, **kwargs):
    characters = characters.dropna(subset=["height"])
    characters = characters.sort_values(by=["height"])
    # characters['age'] = characters['birthday'].apply(calculate_age)
    # characters = characters[(characters["age"] >= 18) & (characters["age"] <= 30)]
    output_dir = "data/plots/height_distribution.svg"
    total_bar_height = target_image_height + bar_spacing
    plot_height = len(characters) * (total_bar_height / 70)
    plot_width = plot_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    bars = ax.barh(range(len(characters)), characters["height"], height=0.9,
                   color='skyblue', align='center', alpha=0.8)

    # Define the reference plot size and corresponding font sizes
    # dividing by 100 because width= 1 -> 100pixel
    ref_plot_height, ref_plot_width = 52700 / 100, 5270 / 100
    ref_x_tick_labelsize = 80
    ref_x_label_fontsize = 100
    ref_text_fontsize = 100
    ref_title_fontsize = 120

    # Calculate scaling factors based on plot size
    width_scale_factor = plot_width / ref_plot_width
    height_scale_factor = plot_height / ref_plot_height
    overall_scale_factor = (width_scale_factor + height_scale_factor) / 2

    # Adjust font sizes dynamically
    x_tick_labelsize = ref_x_tick_labelsize * overall_scale_factor
    x_label_fontsize = ref_x_label_fontsize * overall_scale_factor
    text_fontsize = ref_text_fontsize * overall_scale_factor
    title_fontsize = ref_title_fontsize * overall_scale_factor

    for i, (character_name, height_value) in enumerate(
            zip(characters["name"], characters["height"])):
        offset_image(i, character_name, ax=ax, target_height=target_image_height)

        # Add the height of the character at the end of each bar
        ax.text(height_value + 0.5, i, f'{height_value:.2f}', va='center', fontsize=text_fontsize,
                color='black')

    ax.set_yticks([])
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)

    ax.set_xlabel('Height', fontsize=x_label_fontsize)
    ax.set_title('Character Height Distribution', fontsize=title_fontsize)
    ax.tick_params(axis='x', labelsize=x_tick_labelsize)

    plt.margins(y=0)
    plt.tight_layout()

    output_dir = Path(output_dir)

    plt.savefig(output_dir / "", format='svg')

    return fig


def create_character_ranking_barchart(tierlists: pd.DataFrame, target_image_height=108,
                                      bar_spacing=0.1,
                                      aspect_ratio=0.05, **kwargs):
    rank_df = get_evaluated_tierlist_df(tierlists)

    character_names = rank_df['name'].tolist()
    average_rating = rank_df['average_rating'].tolist()
    std_devs = rank_df['std_dev'].tolist()

    total_bar_height = target_image_height + bar_spacing
    plot_height = len(character_names) * (total_bar_height / 70)
    plot_width = plot_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    y_positions = range(len(character_names))
    bars = ax.barh(y_positions, average_rating, height=0.9, color='skyblue', align='center',
                   alpha=0.8)

    ref_plot_height, ref_plot_width = 52700 / 100, 5270 / 100
    ref_x_tick_labelsize = 80
    ref_x_label_fontsize = 100
    ref_text_fontsize = 100
    ref_title_fontsize = 120

    width_scale_factor = plot_width / ref_plot_width
    height_scale_factor = plot_height / ref_plot_height
    overall_scale_factor = (width_scale_factor + height_scale_factor) / 2

    x_tick_labelsize = ref_x_tick_labelsize * overall_scale_factor
    x_label_fontsize = ref_x_label_fontsize * overall_scale_factor
    text_fontsize = ref_text_fontsize * overall_scale_factor
    title_fontsize = ref_title_fontsize * overall_scale_factor

    for i, (character_name, avg_value, std_dev) in enumerate(
            zip(character_names, average_rating, std_devs)):
        offset_image(i, character_name, ax=ax, target_height=target_image_height)
        ax.text(avg_value + 0.05, i, f'{avg_value:.2f} (SD: {std_dev:.2f})', va='center',
                fontsize=text_fontsize,
                color='black')

    ax.set_yticks([])
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    ax.set_xlabel('Tier', fontsize=x_label_fontsize)
    ax.set_title('Character Ranking Distribution', fontsize=title_fontsize)
    ax.tick_params(axis='x', labelsize=x_tick_labelsize)
    ax.set_xlim(0, 5)
    ax.set_xticks(range(6))
    tiers_ordered = ['D', 'C', 'B', 'A', 'S', 'SS']
    ax.set_xticklabels(tiers_ordered)

    plt.margins(y=0)
    plt.tight_layout()

    output_dir = Path('data/plots/character_ranking_distribution.svg')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir, format='svg')

    return fig


def create_character_ranking_barchart_no_image(tierlists: pd.DataFrame, **kwargs):
    rank_df = get_evaluated_tierlist_df(tierlists)

    character_names = rank_df['name'].tolist()
    average_rating = rank_df['average_rating'].tolist()
    std_devs = rank_df['std_dev'].tolist()

    fig_height = len(character_names) * 0.1
    fig_width = 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_positions = range(len(character_names))
    bars = ax.barh(y_positions, average_rating, height=0.8, color='skyblue', align='center',
                   alpha=0.8)

    # Set font sizes
    x_tick_labelsize = 8
    x_label_fontsize = 10
    y_tick_labelsize = 8
    text_fontsize = 6
    title_fontsize = 9

    # Add value labels next to the bars
    for i, (character_name, avg_value, std_dev) in enumerate(
            zip(character_names, average_rating, std_devs)):
        ax.text(avg_value + 0.05, i, f'{avg_value:.2f} (SD: {std_dev:.2f})', va='center',
                fontsize=text_fontsize,
                color='black')
    # Set the y-axis to display character names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(character_names, fontsize=y_tick_labelsize)

    # Configure the rest of the plot
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    ax.set_xlabel('Tier', fontsize=x_label_fontsize)
    ax.set_title('Character Ranking Distribution', fontsize=title_fontsize)
    ax.tick_params(axis='x', labelsize=x_tick_labelsize)
    ax.set_xlim(0, 5)
    ax.set_xticks(range(6))
    tiers_ordered = ['D', 'C', 'B', 'A', 'S', 'SS']
    ax.set_xticklabels(tiers_ordered)

    plt.margins(y=0)
    plt.tight_layout()
    return fig


def _create_grouped_boxplots(characters: pd.DataFrame, x_grouping: str, y_values: str, ylabel: str,
                             title: str):
    data = [characters[characters[x_grouping] == group_name][y_values] for group_name in
            characters[x_grouping].unique()]
    labels = characters[x_grouping].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    ax.set_title(title)
    return fig


def create_muscle_mass_boxplots_by_race(characters: pd.DataFrame, **kwargs):
    characters_copy = characters.copy()
    characters_copy = characters_copy[
        (characters_copy['muscle_mass'] != 0) & (characters_copy['muscle_mass'].notnull())]
    characters_copy = characters_copy.reset_index(drop=True)
    _create_grouped_boxplots(characters_copy, "race", "muscle_mass", "Muscle mass",
                             "Muscle mass distribution by race")


def create_weight_boxplots_by_race(characters: pd.DataFrame, **kwargs):
    _create_grouped_boxplots(characters, "race", "weight", "Weight", "Weight distribution by race")


def _create_correlation_plot(characters: pd.DataFrame, x_key: str, y_key: str, title: str,
                             filter_y_zeros=False, filter_x_zeros=False):
    fig, ax = plt.subplots()
    df = characters.copy()
    if filter_y_zeros:
        df = df[df[y_key] != 0]
    if filter_x_zeros:
        df = df[df[x_key] != 0]
    x_values = df[x_key]
    y_values = df[y_key]
    ax.scatter(x_values, y_values)

    slope, intercept, r, p, stdErr = stats.linregress(x_values, y_values)
    line = slope * x_values + intercept
    ax.plot(x_values, line, color="red", label=f"f(x) = {slope:.3f}x + {intercept:.3f}")

    correlation_text = (f'R = {r:.4f}\n'
                        f'R² = {r ** 2:.4f}\n'
                        f'p = {p:.4e}\n'
                        f'p% = {p * 100:.2f}')

    ax.text(0.05, 0.95, correlation_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    ax.set_title(title)
    ax.legend()
    return fig


def create_weight_height_correlation_plot_with_zero_weights(characters: pd.DataFrame, **kwargs):
    return _create_correlation_plot(characters, "height", "weight", "Weight by height (w/ 0s)",
                                    filter_y_zeros=False)


def create_weight_height_correlation_plot_without_zero_weights(characters: pd.DataFrame, **kwargs):
    return _create_correlation_plot(characters, "height", "weight", "Weight by height (w/o 0s)",
                                    filter_y_zeros=True, filter_x_zeros=True)


def create_weight_muscle_mass_correlation_plot(characters: pd.DataFrame, **kwargs):
    return _create_correlation_plot(characters, "muscle_mass", "weight", "Weight by muscle mass", filter_y_zeros=True,
                                    filter_x_zeros=True)


def create_muscle_mass_height_correlation_plot(characters: pd.DataFrame, **kwargs):
    return _create_correlation_plot(characters, "muscle_mass", "height", "Height by muscle mass",
                                    filter_y_zeros=True, filter_x_zeros=True)


def create_cup_rating_plot(characters: pd.DataFrame, tierlists: pd.DataFrame, **kwargs):
    combined_df = get_joined_tierlists_characters_df(characters, tierlists)
    combined_df = combined_df[combined_df["sex"] == "w"]
    combined_df = combined_df[combined_df["underbust"] != 0]
    combined_df["cup"] = combined_df["bust"] - combined_df["underbust"]
    return _create_correlation_plot(combined_df, "cup", "average_rating", "Rating by cup size",
                                    filter_y_zeros=True, filter_x_zeros=True)


def create_muscle_mass_rating_correlation_plot(characters: pd.DataFrame, tierlists: pd.DataFrame,
                                               **kwargs):
    combined_df = get_joined_tierlists_characters_df(characters, tierlists)
    return _create_correlation_plot(combined_df, "muscle_mass", "average_rating", "Rating by muscle mass",
                                    filter_y_zeros=True, filter_x_zeros=True)


def create_height_rating_correlation_plot(characters: pd.DataFrame, tierlists: pd.DataFrame,
                                          **kwargs):
    combined_df = get_joined_tierlists_characters_df(characters, tierlists)
    return _create_correlation_plot(combined_df, "height", "average_rating", "Rating by height",
                                    filter_y_zeros=True, filter_x_zeros=True)


def create_weight_rating_correlation_plot(characters: pd.DataFrame, tierlists: pd.DataFrame,
                                          **kwargs):
    combined_df = get_joined_tierlists_characters_df(characters, tierlists)
    return _create_correlation_plot(combined_df, "weight", "average_rating", "Rating by weight",
                                    filter_y_zeros=True, filter_x_zeros=True)


def create_race_class_correlation_plot(characters: pd.DataFrame, **kwargs):
    df = characters.copy()
    df["character_class"] = df["character_class"].replace({None: "Unknown", "": "Unknown"})

    race_class_matrix = df.pivot_table(index="race", columns="character_class", aggfunc='size', fill_value=0)

    race_class_normalized = race_class_matrix.div(race_class_matrix.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = sns.heatmap(race_class_normalized, cmap='coolwarm', fmt='d', cbar=True, ax=ax)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.set_ticklabels([f"{int(t)}%" for t in colorbar.get_ticks()])

    ax.set_title('Correlation Between Character Race and Class in Percent')
    ax.set_xlabel('Class')
    ax.set_ylabel('Race')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


@include_plot
def create_character_ranking_trend(tierlists: pd.DataFrame, **kwargs):
    selected_authors = ["zetsu"]
    select_all_authors_flag = False
    selected_character = None  # name of character or None

    if select_all_authors_flag:
        filter_df = tierlists.copy()
    else:
        # If selected_authors is a single string, convert it to a list for uniformity
        if isinstance(selected_authors, str):
            selected_authors = [selected_authors]
        filter_df = tierlists[tierlists['author'].isin(selected_authors)].copy()

    # Find the global maximum session number among the filtered authors
    max_session = filter_df['sessionNr'].max()

    continuous_author_dfs = []
    for author, group in filter_df.groupby('author', group_keys=False):
        # Set sessionNr as the index
        group = group.set_index('sessionNr').sort_index()

        # We want to ensure that the range goes up to the global max_session
        start_session = group.index.min()
        all_sessions = range(start_session, max_session + 1)

        # Reindex to this full range of sessions
        group = group.reindex(all_sessions)

        # Forward fill missing rows
        group = group.ffill()

        # Reset the index back to a column
        group = group.reset_index().rename(columns={'index': 'sessionNr'})

        # Add back the author column
        group['author'] = author

        continuous_author_dfs.append(group)

    # Combine the processed DataFrames back
    filter_df = pd.concat(continuous_author_dfs, ignore_index=True)

    tier_mapping = {"D": 1, "C": 2, "B": 3, "A": 4, "S": 5, "SS": 6}
    tier_cols = ["D", "C", "B", "A", "S", "SS"]

    exploded = []
    for tier_col in tier_cols:
        tmp = filter_df[['sessionNr', 'author', tier_col]].copy()
        tmp = tmp.explode(tier_col)
        tmp['Character'] = tmp[tier_col]
        tmp['Tier'] = tier_col
        tmp.drop(columns=[tier_col], inplace=True)
        exploded.append(tmp)

    long_df = pd.concat(exploded, ignore_index=True).dropna(subset=["Character"])
    long_df['TierValue'] = long_df['Tier'].map(tier_mapping)

    # Sort before grouping/diffing
    long_df = long_df.sort_values(by=['Character', 'sessionNr'])

    authors_used = filter_df['author'].unique()
    # If multiple authors or all authors are selected, average the TierValue for each session and character
    if select_all_authors_flag or len(authors_used) > 1:
        long_df = (long_df
                   .groupby(['sessionNr', 'Character'], as_index=False)['TierValue']
                   .mean())

    # Re-sort after averaging
    long_df = long_df.sort_values(by=['Character', 'sessionNr'])

    # Identify changing characters
    changes_per_char = (
        long_df
        .groupby('Character')['TierValue']
        .apply(lambda x: (x.diff().fillna(0) != 0).any())
    )
    changing_characters = changes_per_char[changes_per_char].index

    if selected_character:
        if selected_character not in changing_characters:
            print(f"The character '{selected_character}' never changed ratings.")
            return None
        changing_characters = [selected_character]

    # Filter to only changing characters
    long_df = long_df[long_df['Character'].isin(changing_characters)]

    if long_df.empty:
        print("No characters have changed their rating based on the specified conditions.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Unique colors for each character
    unique_chars = long_df['Character'].unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_chars))

    # Optional: Different markers and line styles for better distinction
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'H']
    line_styles = ['-', '--', '-.', ':']

    # Optional: Add jitter to slightly separate overlapping lines
    # Set this to a small nonzero value if you'd like to visually separate overlapping lines
    jitter_strength = 0.0  # Try 0.05 or 0.1 if needed

    for i, char in enumerate(unique_chars):
        char_data = long_df[long_df['Character'] == char]
        marker = markers[i % len(markers)]
        style = line_styles[(i // len(markers)) % len(line_styles)]

        # Apply vertical jitter if desired
        if jitter_strength > 0:
            jitter = np.random.normal(0, jitter_strength, size=len(char_data))
        else:
            jitter = 0

        ax.plot(
            char_data['sessionNr'],
            char_data['TierValue'] + jitter,
            marker=marker,
            linestyle=style,
            label=char,
            color=cmap(i),
            alpha=0.8
        )

    # Title handling
    if select_all_authors_flag:
        title_authors = "All Authors"
    else:
        if len(authors_used) == 1:
            title_authors = authors_used[0]
        elif 1 < len(authors_used) < 5:
            title_authors = ", ".join(authors_used)
        else:
            title_authors = "Multiple Authors"

    ax.set_title(f"Character Tier Changes Over Sessions\n(Authors: {title_authors})")
    ax.set_xlabel("Session Number")
    ax.set_ylabel("Tier")
    ax.set_xticks(sorted(long_df['sessionNr'].unique()))
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["D", "C", "B", "A", "S", "SS"])
    ax.legend()
    ax.grid(True)

    return fig
